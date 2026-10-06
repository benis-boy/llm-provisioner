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
from services.llm.resource_manager.protocol import Failure, ProviderResponse
from .config import SmolLMProviderConfig
from .gpu import (GPUMemoryObservation, OwnedOllamaSnapshot, ProcessIdentity,
                  ResidencyEvidence, _ResidencyPending, settle_residency)
from .input_bounds import (SMOLLM_MAX_RAW_BYTES, frame_smollm_prompt,
                            validate_smollm_input)
try:
    from tools.compatibility.debug_trace import lifecycle
except ImportError:
    def lifecycle(*args, **kwargs): return lambda function: function

_GGUF = "SmolLM2-1.7B-Instruct-Q8_0.gguf"
_JSON_LIMIT = 128 * 1024
_CLI_LIMIT = 128 * 1024
_ABSENCE_TIMEOUT_SECONDS = 5.0
_ABSENCE_POLL_INTERVAL_SECONDS = 0.2
_DONE_REASONS = frozenset(("stop", "length", "load", "unload"))

LIFECYCLE_PHASES = frozenset(("validate", "load", "ready", "cleanup"))
LIFECYCLE_SUBREASONS = frozenset((
    "profile_shape_identity", "gpu_identity_proof", "artifact_verification",
    "gpu_memory_proof", "ollama_create", "readiness_request",
    "endpoint_residency", "gpu_residency", "fallback_ownership_memory",
    "timeout", "cleanup_verification",
))


class LifecycleFailure(RuntimeError):
    """A closed, non-text lifecycle classification for startup and cleanup."""

    def __init__(self, phase: str, subreason: str, cause: BaseException | None = None):
        if phase not in LIFECYCLE_PHASES or subreason not in LIFECYCLE_SUBREASONS:
            raise ValueError("invalid lifecycle classification")
        super().__init__("provider lifecycle operation failed")
        self.lifecycle_phase = phase
        self.lifecycle_subreason = subreason
        if cause is not None:
            self.__cause__ = cause


def _lifecycle(phase: str, subreason: str, exc: BaseException) -> LifecycleFailure:
    if isinstance(exc, LifecycleFailure):
        return exc
    if isinstance(exc, asyncio.CancelledError):
        raise exc
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        subreason = "timeout"
    return LifecycleFailure(phase, subreason, exc)


class _RequestFailure(RuntimeError):
    """A bounded provider failure suitable for ResourceManager classification."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(failure.message)
        self.failure = failure
        self.failure_kind = "provider_execution_failed"
        self.failure_code = failure.code
        self.failure_message = failure.message


def _request_failure(code: str, message: str, *, retryable: bool = True) -> _RequestFailure:
    return _RequestFailure(Failure(code, message, retryable))


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
        # A failed pre-listener load is different from a failed unload: no
        # request can have reached the private daemon, so it may be safe to
        # discard the client once the independent ownership fence is clean.
        self._listener_reached = False
        self._pre_listener_failure = False
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

    @staticmethod
    def _connection_refused(exc: BaseException) -> bool:
        """Recognise refusal without treating exception text as a protocol."""
        pending = [exc]
        seen: set[int] = set()
        while pending and len(seen) < 16:
            current = pending.pop(0)
            if id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current, (aiohttp.ClientConnectorError,
                                    ConnectionRefusedError)):
                return True
            if isinstance(current, OSError) and getattr(current, "errno", None) == 111:
                return True
            if isinstance(current, BaseExceptionGroup):
                pending.extend(item for item in current.exceptions
                               if isinstance(item, BaseException))
            for related in (current.__cause__, current.__context__):
                if isinstance(related, BaseException):
                    pending.append(related)
        return False

    async def _pre_listener_absent(self) -> bool:
        """Require an ownership fence before accepting a refused first probe.

        A refused socket is only an observation.  The GPU cleanup probe (and,
        when supplied, the daemon's process-group probe) must also say that no
        owned work remains.  Unknown ownership errors stay fail-closed.
        """
        proof = self.config.gpu_proof
        if proof is None:
            return False
        try:
            clean = await self._resolve(proof.cleanup())
        except BaseException:
            return False
        if clean is not True:
            return False
        ownership = proof.ollama_ownership
        if ownership is None:
            # The GPU proof is the existing process absence fence for legacy
            # bindings.  It is combined with the refused private endpoint and
            # the fact that this provider never reached its listener.
            return True
        try:
            await self._resolve(ownership())
        except (ProcessLookupError, FileNotFoundError):
            return True
        except BaseException as exc:
            message = str(exc).lower()
            if any(word in message for word in ("not available", "not started", "disappeared", "group gone")):
                return True
            return False
        # A positive snapshot means an owned daemon may exist.
        return False

    async def _clear_local_state(self) -> None:
        session = self._session
        self._session = None
        self._model = self._root = self._evidence = None
        self._profile = None
        self._preload_memory = None
        self._model_specific_ready = False
        self._ready = False
        self._listener_reached = False
        self._pre_listener_failure = False
        if session is not None and not session.closed:
            await session.close()

    async def _close_session(self, session: aiohttp.ClientSession) -> None:
        """Close a client even when load is cancelled, with an owned deadline."""
        if session.closed:
            return
        close_task = asyncio.create_task(session.close())
        deadline = time.monotonic() + self.config.request_timeout_seconds
        cancelled = False
        while not close_task.done() and time.monotonic() < deadline:
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                cancelled = True
                continue
        if not close_task.done():
            # Do not detach a live owned close.  Cancellation is the only
            # bounded escape hatch for a connector whose transport is stuck.
            close_task.cancel()
            await asyncio.gather(close_task, return_exceptions=True)
            if cancelled:
                raise asyncio.CancelledError
            raise RuntimeError("provider client close timed out")
        close_task.result()
        if cancelled:
            raise asyncio.CancelledError

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

    @lifecycle("provider.smollm")
    async def validate(self, profile: CapacityProfile) -> None:
        try:
            if (profile.model_id.value != "SmolLM" or
                    profile.optimal_parallelism != self.config.parallelism or
                    profile.context_size != 512 or profile.bucket_identity is not None):
                raise ValueError("profile is not the configured SmolLM capacity")
            if not profile.matches(profile.model_id, self.config.gpu_uuid,
                                   self.config.manifest_sha256, self.config.model_sha256,
                                   self.config.runtime_identity, self.config.adapter_identity):
                raise ValueError("capacity profile identities do not match server configuration")
        except BaseException as exc:
            raise _lifecycle("validate", "profile_shape_identity", exc)
        try:
            if await self._gpu() != self.config.gpu_uuid:
                raise ValueError("GPU identity proof mismatch")
        except BaseException as exc:
            raise _lifecycle("validate", "gpu_identity_proof", exc)
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

    @lifecycle("provider.smollm")
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
            try:
                evidence, root, input_evidence = await asyncio.to_thread(self._prove_artifact)
            except BaseException as exc:
                raise _lifecycle("load", "artifact_verification", exc)
            name = "smollm-" + self.config.model_sha256[:16]
            self._cleanup_verified = False
            self._ownership_started = True
            self._session = aiohttp.ClientSession(base_url=self._url, trust_env=False,
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds))
            self._model, self._root, self._evidence = name, root, input_evidence
            try:
                try:
                    self._preload_memory = await self._memory()
                except BaseException as exc:
                    raise _lifecycle("load", "gpu_memory_proof", exc)
                try:
                    await self._run_create(name, root / _GGUF)
                except BaseException as exc:
                    raise _lifecycle("load", "ollama_create", exc)
                self._listener_reached = True
            except BaseException as exc:
                # Import may have reached the daemon; retain ownership for RM cleanup.
                self._pre_listener_failure = (not self._listener_reached and
                                              self._connection_refused(exc))
                session = self._session
                if session is not None:
                    await self._close_session(session)
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
        started = time.monotonic_ns()
        try:
            async with self._session.post("/api/generate", json=body, allow_redirects=False) as response:
                if response.status != 200:
                    raise _request_failure("ollama_http_status", "Ollama generation returned a non-success status",
                                           retryable=response.status >= 500)
                try:
                    value = await self._json(response)
                except _RequestFailure:
                    raise
                except Exception as exc:
                    raise _request_failure("ollama_json_response", "Ollama returned invalid JSON") from exc
        except _RequestFailure:
            raise
        except Exception as exc:
            raise _request_failure("ollama_transport", "Ollama generation transport failed") from exc
        if not isinstance(value, dict) or value.get("done") is not True:
            raise _request_failure("ollama_response_contract", "Ollama generation response contract failed", retryable=False)
        if preload:
            return value
        if not isinstance(value.get("response"), str) or not value["response"]:
            raise _request_failure("ollama_response_contract", "Ollama generation response contract failed", retryable=False)
        count = value.get("prompt_eval_count")
        if type(count) is not int or not 1 <= count <= 448:
            raise _request_failure("ollama_telemetry_contract", "Ollama generation telemetry contract failed", retryable=False)
        options = body.get("options") if isinstance(body, dict) else None
        configured = options.get("num_predict") if isinstance(options, dict) else None
        context = options.get("num_ctx") if isinstance(options, dict) else None
        if type(configured) is not int or configured < 1:
            raise _request_failure("ollama_response_contract", "Ollama generation response contract failed", retryable=False)
        if type(context) is not int or context < 1:
            raise _request_failure("ollama_response_contract", "Ollama generation response contract failed", retryable=False)
        eval_count = value.get("eval_count")
        if type(eval_count) is not int or not 1 <= eval_count <= configured:
            raise _request_failure("ollama_telemetry_contract", "Ollama generation telemetry contract failed", retryable=False)
        done_reason = value.get("done_reason")
        if done_reason is not None and (not isinstance(done_reason, str) or
                                        done_reason not in _DONE_REASONS):
            raise _request_failure("ollama_telemetry_contract", "Ollama generation telemetry contract failed", retryable=False)
        value["_native_started_ns"] = started
        value["_native_ended_ns"] = time.monotonic_ns()
        return value

    async def _check_residency(self) -> None:
        await self._resident_model()

    @lifecycle("provider.smollm")
    async def ready(self) -> None:
        self._ready = False
        self._model_specific_ready = False
        if self._session is None or self._model is None:
            raise RuntimeError("provider is not loaded")
        try:
            if await self._gpu() != self.config.gpu_uuid:
                raise RuntimeError("GPU identity proof mismatch")
        except BaseException as exc:
            raise _lifecycle("ready", "gpu_identity_proof", exc)
        try:
            await self._request(self._body(""), preload=True)
        except BaseException as exc:
            raise _lifecycle("ready", "readiness_request", exc)
        try:
            await self._check_residency()
        except BaseException as exc:
            raise _lifecycle("ready", "endpoint_residency", exc)
        try:
            await self._residency_proof()
        except _ResidencyPending:
            try:
                await self._fallback_residency()
            except BaseException as exc:
                raise _lifecycle("ready", "fallback_ownership_memory", exc)
        except BaseException as exc:
            raise _lifecycle("ready", "gpu_residency", exc)
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

    @staticmethod
    def _execute_internal_failure() -> _RequestFailure:
        """Close unexpected adapter faults at the provider execution boundary."""
        return _request_failure("smollm_internal", "SmolLM execution failed")

    @staticmethod
    def _observation_contract_failure() -> _RequestFailure:
        return _request_failure("smollm_observation_contract",
                                "SmolLM execution observation contract failed",
                                retryable=False)

    @lifecycle("provider.smollm", failures_only=True)
    async def execute(self, request_id: str, payload: bytes) -> ProviderResponse:
        try:
            if self._session is None or self._model is None or not self._ready:
                raise self._execute_internal_failure()
            if request_id in self._tasks:
                raise self._execute_internal_failure()
            if len(self._tasks) >= self.config.parallelism:
                raise self._execute_internal_failure()
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise _request_failure("smollm_input_decode", "SmolLM input decoding failed",
                                       retryable=False) from exc
            try:
                self._validate_text(text)
            except (TypeError, ValueError, UnicodeError) as exc:
                raise _request_failure("smollm_input_validation",
                                       "SmolLM input validation failed", retryable=False) from exc
        except asyncio.CancelledError:
            raise
        except _RequestFailure:
            raise
        except BaseException as exc:
            raise self._execute_internal_failure() from exc

        async def run():
            # Keep the witness coupled to the exact native request sent over
            # the wire.  A second hard-coded extractor is not evidence of the
            # request that Ollama actually received.
            try:
                body = self._body(text)
                value = await self._request(body)
                try:
                    started = value.pop("_native_started_ns")
                    ended = value.pop("_native_ended_ns")
                    options = body["options"]
                    response = value["response"]
                    if (type(started) is not int or type(ended) is not int or started < 0
                            or ended < started or not isinstance(options, dict)
                            or not isinstance(response, str) or not response):
                        raise ValueError("invalid native observation")
                    observation = {"kind": "ollama_generate", "request_id": request_id,
                                   "execution_started": started, "execution_ended": ended,
                                   "configured_num_predict": options["num_predict"],
                                   "configured_num_ctx": options["num_ctx"],
                                   "configured_temperature": options["temperature"],
                                   "prompt_eval_count": value.get("prompt_eval_count"),
                                   "eval_count": value.get("eval_count"),
                                   "done_reason": value.get("done_reason"),
                                   "prompt_eval_duration": value.get("prompt_eval_duration"),
                                   "eval_duration": value.get("eval_duration"),
                                   "load_duration": value.get("load_duration"),
                                   "total_duration": value.get("total_duration")}
                except (KeyError, TypeError, ValueError, UnicodeError) as exc:
                    raise self._observation_contract_failure() from exc
                return ProviderResponse(response.encode(), None, False, observation)
            except asyncio.CancelledError:
                raise
            except _RequestFailure:
                raise
            except BaseException as exc:
                raise self._execute_internal_failure() from exc
        try:
            task = asyncio.create_task(run())
            self._tasks[request_id] = task
        except asyncio.CancelledError:
            raise
        except _RequestFailure:
            raise
        except BaseException as exc:
            raise self._execute_internal_failure() from exc
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

    @lifecycle("provider.smollm")
    async def unload(self) -> None:
        try:
            await self._unload()
        except BaseException as exc:
            raise _lifecycle("cleanup", "cleanup_verification", exc)

    async def _unload(self) -> None:
        async with self._lock:
            self._model_specific_ready = False
            self._ready = False
            if self._tasks:
                _, pending = await asyncio.wait(tuple(self._tasks.values()), timeout=self.config.request_timeout_seconds)
                if pending:
                    raise RuntimeError("provider requests did not drain")
            if self._pre_listener_failure:
                if not await self._pre_listener_absent():
                    # The endpoint refusal does not prove absence.  We can
                    # still release our HTTP transport, but deliberately keep
                    # cleanup unverified so Resource Manager remains fenced.
                    await self._clear_local_state()
                    raise RuntimeError("cannot prove pre-listener Ollama absence")
                self._cleanup_verified = True
                await self._clear_local_state()
                return
            replaced_session = False
            if self._session is not None and self._session.closed:
                self._session = aiohttp.ClientSession(base_url=self._url, trust_env=False,
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds))
                replaced_session = True
            if self._session is not None and self._model is not None:
                try:
                    async with self._session.post("/api/generate", json={"model": self._model, "prompt": "", "keep_alive": 0, "stream": False}, allow_redirects=False) as response:
                        if response.status != 200:
                            raise RuntimeError("Ollama unload failed")
                        terminal = await self._json(response)
                        if not isinstance(terminal, dict) or terminal.get("done") is not True or terminal.get("error"):
                            raise RuntimeError("Ollama unload was not terminal")
                    await self._check_absent()
                except BaseException:
                    # A replacement exists only to make a closed client usable
                    # for verification.  It must never survive a failed probe;
                    # retain the closed session and model as the fail-closed
                    # witness rather than clearing state that proves cleanup is
                    # still unverified.
                    if replaced_session and self._session is not None:
                        await self._close_session(self._session)
                    raise
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
            await self._clear_local_state()

    @lifecycle("provider.smollm")
    async def verify_cleanup(self) -> bool:
        return (self._session is None and self._model is None and
                self._cleanup_verified)

    @lifecycle("provider.smollm", failures_only=True)
    async def validate_input(self, payload: bytes, *, context_size: int | None, bucket_identity: str | None) -> None:
        if (not isinstance(payload, bytes) or self._profile is None or context_size != 512
                or bucket_identity is not None or not self._profile.accepts_request(context_size, bucket_identity)):
            raise ValueError("request does not match measured context profile")
        self._validate_text(payload.decode("utf-8"))
