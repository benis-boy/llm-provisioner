"""A fenced, offline SmolLM adapter for one supervisor-owned Ollama daemon."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import tempfile
import time

import aiohttp

from services.llm.provisioning.volume import verify_current
from services.llm.resource_manager.contracts import CapacityProfile
from services.llm.resource_manager.protocol import ProviderResponse
from .config import SmolLMProviderConfig
from .gpu import (GPUMemoryObservation, OwnedOllamaSnapshot, ProcessIdentity,
                  ResidencyEvidence, _ResidencyPending, settle_residency)
from .input_bounds import (SMOLLM_MAX_RAW_BYTES, frame_smollm_prompt,
                           validate_smollm_input)

_GGUF = "SmolLM2-1.7B-Instruct-Q8_0.gguf"
_JSON_LIMIT = 128 * 1024
_CLI_LIMIT = 128 * 1024
_ABSENCE_TIMEOUT_SECONDS = 5.0
_ABSENCE_POLL_INTERVAL_SECONDS = 0.2


class SmolLMProvider:
    def __init__(self, config: SmolLMProviderConfig):
        self.config = config
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._model: str | None = None
        self._evidence: dict | None = None
        self._root: Path | None = None
        self._tasks: dict[str, asyncio.Task[ProviderResponse]] = {}
        self._profile: CapacityProfile | None = None
        self._cleanup_verified = False
        self._ready = False
        self._preload_memory: GPUMemoryObservation | None = None
        self._model_specific_ready = False
        # Keep the clock and wait operation injectable so cleanup polling can
        # be tested without making the bounded wait real-time.
        self._absence_clock = time.monotonic
        self._absence_sleep = asyncio.sleep
        self._residency_clock = time.monotonic
        self._residency_sleep = asyncio.sleep

    @property
    def _url(self) -> str:
        return f"http://127.0.0.1:{self.config.ollama_port}"

    @staticmethod
    async def _resolve(value):
        return await value if inspect.isawaitable(value) else value

    async def _gpu(self) -> str:
        if self.config.gpu_proof is None:
            raise RuntimeError("GPU proof is unavailable")
        value = await self._resolve(self.config.gpu_proof.identity())
        if not isinstance(value, str):
            raise RuntimeError("GPU identity proof is invalid")
        return value

    async def _residency_proof(self) -> ResidencyEvidence:
        proof = self.config.gpu_proof
        if proof is None or proof.residency is None or proof.expected_supervisor is None:
            raise RuntimeError("GPU residency proof is unavailable")
        value = await settle_residency(proof.residency, sleep=self._residency_sleep,
                                       monotonic=self._residency_clock)
        if type(value) is not ResidencyEvidence:
            raise RuntimeError("GPU residency proof is invalid")
        if value.gpu_uuid != self.config.gpu_uuid:
            raise RuntimeError("GPU residency identity mismatch")
        supervisor = value.supervisor
        expected = proof.expected_supervisor
        if type(supervisor) is not ProcessIdentity or type(supervisor.pid) is not int or supervisor.pid <= 0 or type(supervisor.start_time) is not int or supervisor.start_time <= 0:
            raise RuntimeError("GPU supervisor identity proof is invalid")
        if type(expected) is not ProcessIdentity or supervisor != expected:
            raise RuntimeError("GPU supervisor identity mismatch")
        runners = value.runners
        if type(runners) is not tuple or not runners:
            raise RuntimeError("GPU runner residency proof is invalid")
        for runner in runners:
            if (type(runner) is not ProcessIdentity or type(runner.pid) is not int or runner.pid <= 0 or
                    type(runner.start_time) is not int or runner.start_time <= 0 or runner.pid == supervisor.pid):
                raise RuntimeError("GPU runner residency proof is invalid")
        if len({runner.pid for runner in runners}) != len(runners):
            raise RuntimeError("GPU runner residency proof is invalid")
        return value

    async def _memory(self) -> GPUMemoryObservation:
        if self.config.gpu_proof is None or self.config.gpu_proof.memory is None:
            raise RuntimeError("GPU memory proof is unavailable")
        value = await self._resolve(self.config.gpu_proof.memory())
        if type(value) is not GPUMemoryObservation:
            raise RuntimeError("GPU memory proof is invalid")
        fields = (value.start_ns, value.end_ns, value.total_bytes, value.used_bytes, value.free_bytes)
        if (any(type(item) is not int for item in fields) or value.start_ns > value.end_ns or
                value.total_bytes <= 0 or value.used_bytes < 0 or value.free_bytes < 0 or
                value.used_bytes > value.total_bytes or value.free_bytes > value.total_bytes or
                value.used_bytes + value.free_bytes > value.total_bytes):
            raise RuntimeError("GPU memory proof is malformed")
        expected = self.config.gpu_proof.expected_supervisor
        if value.gpu_uuid != self.config.gpu_uuid or value.supervisor != expected:
            raise RuntimeError("GPU memory identity mismatch")
        return value

    async def _resident_model(self) -> tuple[str, int, int]:
        """Read and validate the exact private endpoint model state."""
        assert self._session is not None and self._model is not None
        async with self._session.get("/api/ps", allow_redirects=False) as response:
            if response.status != 200:
                raise RuntimeError("Ollama residency check failed")
            data = await self._json(response)
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
            raise RuntimeError("Ollama residency is not exclusively owned")
        item = models[0]
        name, size, size_vram = item.get("name"), item.get("size"), item.get("size_vram")
        if (name != self._model + ":latest" or type(size) is not int or
                type(size_vram) is not int or size <= 0 or size_vram != size):
            raise RuntimeError("intended model is not fully resident on GPU")
        return name, size, size_vram

    @staticmethod
    def _validate_owned_snapshot(value, expected_supervisor: ProcessIdentity) -> OwnedOllamaSnapshot:
        if type(value) is not OwnedOllamaSnapshot:
            raise RuntimeError("Ollama ownership proof is invalid")
        identities = (value.supervisor, value.daemon, *value.descendants)
        if value.broker is not None:
            identities = (value.supervisor, value.broker, value.daemon, *value.descendants)
        for identity in identities:
            if (type(identity) is not ProcessIdentity or type(identity.pid) is not int or
                    identity.pid <= 0 or type(identity.start_time) is not int or identity.start_time <= 0):
                raise RuntimeError("Ollama ownership identity is invalid")
        if (value.supervisor != expected_supervisor or value.daemon.pid in
                {value.supervisor.pid} or (value.broker is not None and
                value.broker.pid in {value.supervisor.pid, value.daemon.pid})):
            raise RuntimeError("Ollama ownership supervisor or daemon mismatch")
        if (type(value.descendants) is not tuple or not value.descendants or
                len(set(value.descendants)) != len(value.descendants) or
                len({item.pid for item in value.descendants}) != len(value.descendants) or
                any(item.pid in {value.daemon.pid, value.supervisor.pid} for item in value.descendants) or
                tuple(sorted(value.descendants, key=lambda item: (item.pid, item.start_time))) != value.descendants):
            raise RuntimeError("Ollama ownership descendants are invalid")
        return value

    async def _fallback_residency(self) -> None:
        proof = self.config.gpu_proof
        if proof is None or proof.memory is None or proof.ollama_ownership is None:
            raise RuntimeError("SmolLM model-specific residency proof is unavailable")
        expected = proof.expected_supervisor
        if expected is None:
            raise RuntimeError("GPU supervisor identity proof is unavailable")
        snapshot_a = self._validate_owned_snapshot(
            await self._resolve(proof.ollama_ownership()), expected)
        endpoint_a = await self._resident_model()
        post = await self._memory()
        endpoint_b = await self._resident_model()
        snapshot_b = self._validate_owned_snapshot(
            await self._resolve(proof.ollama_ownership()), expected)
        if snapshot_a != snapshot_b or endpoint_a != endpoint_b:
            raise RuntimeError("SmolLM ownership or endpoint changed during readiness fence")
        pre = self._preload_memory
        if (pre is None or post.gpu_uuid != pre.gpu_uuid or post.supervisor != pre.supervisor or
                post.total_bytes != pre.total_bytes or post.supervisor != expected):
            raise RuntimeError("SmolLM memory identity or capacity changed")
        if type(pre.used_bytes) is not int or type(post.used_bytes) is not int or post.used_bytes <= pre.used_bytes:
            raise RuntimeError("SmolLM GPU memory increase is not positive")
        self._model_specific_ready = True

    def accepted_model_specific_residency(self) -> bool:
        return self._model_specific_ready

    async def validate(self, profile: CapacityProfile) -> None:
        if (profile.model_id.value != "SmolLM" or
                profile.optimal_parallelism != self.config.parallelism or
                profile.context_size != 512 or profile.bucket_identity is not None):
            raise ValueError("profile is not the configured SmolLM capacity")
        if not profile.matches(profile.model_id, self.config.gpu_uuid,
                               self.config.manifest_sha256, self.config.model_sha256,
                               self.config.runtime_identity, self.config.adapter_identity):
            raise ValueError("capacity profile identities do not match server configuration")
        if await self._gpu() != self.config.gpu_uuid:
            raise ValueError("GPU identity proof mismatch")
        self._profile = profile

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _prove_artifact(self) -> tuple[dict, Path, dict]:
        evidence = verify_current(self.config.artifact_root)
        if not isinstance(evidence, dict) or evidence.get("manifestSha256") != self.config.manifest_sha256:
            raise ValueError("current artifact manifest mismatch")
        root = self.config.artifact_root / self.config.manifest_sha256
        model_root = root / "models" / "SmolLM"
        gguf = model_root / _GGUF
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        files = manifest.get("models", {}).get("SmolLM", {}).get("files", [])
        item = next((entry for entry in files if isinstance(entry, dict) and entry.get("path") == _GGUF), None)
        if not isinstance(item, dict) or item.get("sha256") != self.config.model_sha256:
            raise ValueError("configured model identity is not the selected GGUF")
        if not gguf.is_file() or self._sha256(gguf) != self.config.model_sha256:
            raise ValueError("selected model artifact mismatch")
        return evidence, model_root, validate_smollm_input(model_root, "probe")

    async def _run_create(self, name: str, gguf: Path) -> None:
        absolute = str(gguf.resolve())
        if not absolute.isascii() or any(ord(char) < 0x20 for char in absolute):
            raise ValueError("local model path must be bounded ASCII")
        # Ollama's Modelfile syntax accepts quoted paths; only the verified local
        # GGUF is interpolated, and it is escaped before the temporary file exists.
        contents = ('FROM "' + absolute.replace("\\", "\\\\").replace('"', '\\"') + '"\n'
                    "PARAMETER num_predict 64\nPARAMETER temperature 0\n")
        with tempfile.NamedTemporaryFile("w", encoding="ascii", delete=False) as handle:
            handle.write(contents)
            modelfile = handle.name
        proc = None
        drains: tuple[asyncio.Task[bytes], ...] = ()
        try:
            env = {"OLLAMA_HOST": self._url, "PATH": "/usr/bin:/bin"}
            if self.config.ollama_home is not None:
                env["HOME"] = str(self.config.ollama_home)
            proc = await asyncio.create_subprocess_exec(
                self.config.ollama_binary, "create", name, "-f", modelfile,
                env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True)
            async def drain(stream):
                output = bytearray()
                while chunk := await stream.read(8192):
                    output.extend(chunk)
                    if len(output) > _CLI_LIMIT:
                        raise RuntimeError("Ollama local import output exceeded bound")
                return bytes(output)
            drains = (asyncio.create_task(drain(proc.stdout)),
                      asyncio.create_task(drain(proc.stderr)))
            stdout, stderr, _ = await asyncio.wait_for(
                asyncio.gather(*drains, proc.wait()),
                self.config.request_timeout_seconds)
            if len(stdout) + len(stderr) > _CLI_LIMIT:
                raise RuntimeError("Ollama local import output exceeded aggregate 256 KiB bound")
            if proc.returncode != 0:
                raise RuntimeError("Ollama local import failed: " + stderr[-512:].decode(errors="replace"))
        except BaseException:
            # The leader may have exited while a descendant retains inherited
            # pipes.  Always kill the saved process group, not merely a live
            # leader.  Readers that stopped at their output bound cannot be
            # reused: settle them first, then discard both pipes through EOF.
            if proc is not None:
                async def terminate_and_drain():
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        for task in drains:
                            if not task.done():
                                task.cancel()
                        if drains:
                            await asyncio.gather(*drains, return_exceptions=True)

                        async def discard(stream):
                            while await stream.read(8192):
                                pass
                        discarded = (asyncio.create_task(discard(proc.stdout)),
                                     asyncio.create_task(discard(proc.stderr)))
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(proc.wait(), *discarded),
                                self.config.request_timeout_seconds)
                        except asyncio.TimeoutError:
                            # A killed descendant can retain a pipe forever.
                            # This is an owned asyncio transport fallback, not
                            # a success path: force both read transports closed
                            # and report a fail-closed import cleanup failure.
                            for fd in (1, 2):
                                pipe = proc._transport.get_pipe_transport(fd)  # noqa: SLF001
                                if pipe is not None:
                                    pipe.close()
                            await asyncio.gather(*discarded, return_exceptions=True)
                            await asyncio.wait_for(proc.wait(), self.config.request_timeout_seconds)
                            raise RuntimeError("Ollama import pipes did not close after group termination")
                    except BaseException:
                        # Cancellation during cleanup still owns all pipe
                        # transports; close them before the task can settle.
                        for fd in (1, 2):
                            pipe = proc._transport.get_pipe_transport(fd)  # noqa: SLF001
                            if pipe is not None:
                                pipe.close()
                        raise
                cleanup = asyncio.create_task(terminate_and_drain())
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                cleanup.result()
            raise
        finally:
            Path(modelfile).unlink(missing_ok=True)

    async def load(self, profile: CapacityProfile) -> None:
        async with self._lock:
            self._model_specific_ready = False
            if self._cleanup_verified is False and self._session is None and self._model is None and getattr(self, "_ownership_started", False):
                raise RuntimeError("previous Ollama ownership has not been cleaned up")
            self._ownership_started = True
            await self.validate(profile)
            if self._session is not None or self._model is not None:
                raise RuntimeError("provider is already loaded")
            # Resource Manager owns cleanup from the beginning of load, even
            # when artifact validation fails before a session is acquired.
            evidence, root, input_evidence = await asyncio.to_thread(self._prove_artifact)
            name = "smollm-" + self.config.model_sha256[:16]
            self._cleanup_verified = False
            self._ownership_started = True
            self._session = aiohttp.ClientSession(base_url=self._url, trust_env=False,
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds))
            self._model, self._root, self._evidence = name, root, input_evidence
            try:
                self._preload_memory = await self._memory()
                await self._run_create(name, root / _GGUF)
            except BaseException:
                # Import may have reached the daemon; retain ownership for RM cleanup.
                raise

    @staticmethod
    async def _json(response):
        data = bytearray()
        async for chunk in response.content.iter_chunked(16384):
            data.extend(chunk)
            if len(data) > _JSON_LIMIT:
                raise RuntimeError("Ollama response too large")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        try:
            return json.loads(bytes(data), object_pairs_hook=pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise RuntimeError("invalid Ollama JSON response") from exc

    async def _request(self, body, *, preload=False):
        assert self._session is not None
        async with self._session.post("/api/generate", json=body, allow_redirects=False) as response:
            if response.status != 200:
                raise RuntimeError("Ollama generation failed")
            value = await self._json(response)
        if not isinstance(value, dict) or value.get("done") is not True:
            raise RuntimeError("invalid Ollama response")
        if preload:
            return value
        if not isinstance(value.get("response"), str) or not value["response"]:
            raise RuntimeError("invalid Ollama response")
        count = value.get("prompt_eval_count")
        if type(count) is not int or not 1 <= count <= 448:
            raise RuntimeError("invalid prompt evaluation count")
        return value

    async def _check_residency(self) -> None:
        await self._resident_model()

    async def ready(self) -> None:
        self._ready = False
        self._model_specific_ready = False
        if self._session is None or self._model is None:
            raise RuntimeError("provider is not loaded")
        if await self._gpu() != self.config.gpu_uuid:
            raise RuntimeError("GPU identity proof mismatch")
        await self._request(self._body(""), preload=True)
        await self._check_residency()
        try:
            await self._residency_proof()
        except _ResidencyPending:
            await self._fallback_residency()
        self._ready = True

    def _body(self, text: str) -> dict:
        assert self._model is not None
        return {"model": self._model, "prompt": frame_smollm_prompt(text), "raw": True,
                "stream": False, "keep_alive": -1,
                "options": {"num_ctx": 512, "num_predict": 64, "temperature": 0}}

    def _validate_text(self, text: str) -> None:
        if self._evidence is None or not text.isascii() or any(ord(c) < 0x20 or ord(c) > 0x7e for c in text):
            raise ValueError("SmolLM input must use the proved printable-ASCII bucket")
        if len(text.encode("ascii")) > SMOLLM_MAX_RAW_BYTES or len(frame_smollm_prompt(text).encode("ascii")) > 448:
            raise ValueError("SmolLM input exceeds proved no-truncation bound")

    async def execute(self, request_id: str, payload: bytes) -> ProviderResponse:
        if self._session is None or self._model is None or not self._ready:
            raise RuntimeError("provider is not loaded")
        if request_id in self._tasks:
            raise RuntimeError("duplicate request id")
        if len(self._tasks) >= self.config.parallelism:
            raise RuntimeError("configured SmolLM parallelism is exhausted")
        text = payload.decode("utf-8")
        self._validate_text(text)
        async def run():
            value = await self._request(self._body(text))
            return ProviderResponse(value["response"].encode(), None, False)
        task = asyncio.create_task(run())
        self._tasks[request_id] = task
        def finish(done):
            if self._tasks.get(request_id) is done:
                self._tasks.pop(request_id, None)
            try: done.exception()
            except BaseException: pass
        task.add_done_callback(finish)
        # A cancelled caller must not detach RM's owned request from its live HTTP
        # task: wait for its terminal state before propagating cancellation.
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            raise

    async def cancel(self, request_id: str) -> None:
        return None

    async def _check_absent(self) -> None:
        assert self._session is not None
        deadline = self._absence_clock() + _ABSENCE_TIMEOUT_SECONDS
        while True:
            async with self._session.get("/api/ps", allow_redirects=False) as response:
                if response.status != 200:
                    raise RuntimeError("cannot verify Ollama cleanup")
                value = await self._json(response)
            if not isinstance(value, dict) or not isinstance(value.get("models"), list):
                raise RuntimeError("Ollama still has resident models")
            if not value["models"]:
                return
            remaining = deadline - self._absence_clock()
            if remaining <= 0:
                raise RuntimeError("Ollama still has resident models")
            await self._absence_sleep(min(_ABSENCE_POLL_INTERVAL_SECONDS, remaining))

    async def unload(self) -> None:
        async with self._lock:
            self._model_specific_ready = False
            self._ready = False
            if self._tasks:
                _, pending = await asyncio.wait(tuple(self._tasks.values()), timeout=self.config.request_timeout_seconds)
                if pending:
                    raise RuntimeError("provider requests did not drain")
            if self._session is not None and self._model is not None:
                async with self._session.post("/api/generate", json={"model": self._model, "prompt": "", "keep_alive": 0, "stream": False}, allow_redirects=False) as response:
                    if response.status != 200:
                        raise RuntimeError("Ollama unload failed")
                    terminal = await self._json(response)
                    if not isinstance(terminal, dict) or terminal.get("done") is not True or terminal.get("error"):
                        raise RuntimeError("Ollama unload was not terminal")
                await self._check_absent()
                self._cleanup_verified = True
                await self._session.close()
            elif self._session is None and self._model is None:
                # Resource Manager may clean up after validation failed before a
                # client session was acquired.  Verify the daemon, rather than
                # treating the absence of our local state as cleanup.
                async with aiohttp.ClientSession(base_url=self._url, trust_env=False,
                        timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)) as probe:
                    async with probe.get("/api/ps", allow_redirects=False) as response:
                        if response.status != 200:
                            raise RuntimeError("cannot verify Ollama cleanup")
                        value = await self._json(response)
                    if not isinstance(value, dict) or not isinstance(value.get("models"), list) or value["models"]:
                        raise RuntimeError("Ollama still has resident models")
                self._cleanup_verified = True
            self._session = self._model = self._root = None
            self._evidence = self._profile = None
            self._preload_memory = None
            self._model_specific_ready = False
            self._ready = False

    async def verify_cleanup(self) -> bool:
        return (self._session is None and self._model is None and
                self._cleanup_verified)

    async def validate_input(self, payload: bytes, *, context_size: int | None, bucket_identity: str | None) -> None:
        if (not isinstance(payload, bytes) or self._profile is None or context_size != 512
                or bucket_identity is not None or not self._profile.accepts_request(context_size, bucket_identity)):
            raise ValueError("request does not match measured context profile")
        self._validate_text(payload.decode("utf-8"))
