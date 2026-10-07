"""One-shot composition and lifecycle owner for the offline runtime.

This module intentionally composes existing authorities; it is not a second
ResourceManager or scheduler.  The runtime owns the parent GPU proof, private
Ollama daemon, HTTP listener, health boundary, and reverse-order cleanup.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aiohttp import web

from services.llm.health import register_routes
from services.llm.providers.gpu import LinuxGPUProof
from services.llm.providers.config import GPUProof
from services.llm.queue.results import ResultStore
from services.llm.queue.contracts import ModelId
from services.llm.resource_manager.core import ResourceManager
from services.llm.resource_manager.health import ResourceManagerHealthBoundary
from services.llm.resource_manager.http import ResourceManagerHttpServer
from .bindings import PreparedBindings, _read_manifest, observe_runtime_identities, prepare_bindings
from .config import BootstrapConfig
from .supervisor import OwnedOllama


@dataclass(frozen=True)
class RuntimeOptions:
    state_dir: Path
    result_dir: Path
    host: str = "127.0.0.1"
    port: int = 8080
    sqlite_min_free_bytes: int = 5 * 1024**3
    shutdown_grace_seconds: float = 60.0
    host_pid_namespace: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.state_dir, Path) or not self.state_dir.is_absolute()
                or not isinstance(self.result_dir, Path) or not self.result_dir.is_absolute()):
            raise ValueError("state and result directories must be absolute paths")
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host is required")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port is out of bounds")
        if type(self.sqlite_min_free_bytes) is not int or self.sqlite_min_free_bytes < 0:
            raise ValueError("sqlite minimum free space is invalid")
        if (isinstance(self.shutdown_grace_seconds, bool)
                or not isinstance(self.shutdown_grace_seconds, (int, float))
                or not 0 < self.shutdown_grace_seconds <= 300):
            raise ValueError("shutdown grace is out of bounds")
        if type(self.host_pid_namespace) is not bool:
            raise ValueError("host PID namespace attestation must be boolean")


class BootstrapRuntime:
    """A one-shot runtime.  Failed cleanup keeps admission permanently fenced."""

    def __init__(self, config: BootstrapConfig, options: RuntimeOptions, *,
                 proof_capture: Callable[..., Any] = LinuxGPUProof.capture,
                 daemon_factory: Callable[..., OwnedOllama] = OwnedOllama,
                 core: ResourceManager | None = None) -> None:
        if not isinstance(config, BootstrapConfig) or not isinstance(options, RuntimeOptions):
            raise TypeError("validated config and runtime options are required")
        self.config, self.options = config, options
        self._capture, self._daemon_factory, self.core = proof_capture, daemon_factory, core or ResourceManager()
        self.proof = None
        self.daemon = None
        self._daemon_started = False
        self.prepared: PreparedBindings | None = None
        self.result_store: ResultStore | None = None
        self.http: ResourceManagerHttpServer | None = None
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self.health: ResourceManagerHealthBoundary | None = None
        self._monitor: asyncio.Task[Any] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._fenced = False
        self._started = False
        self._cleanup_errors: list[BaseException] = []
        self._startup_task: asyncio.Task | None = None
        self._retained: set[asyncio.Task] = set()

    def _state_probe(self) -> bool:
        probe: Path | None = None
        try:
            self.options.state_dir.mkdir(parents=True, exist_ok=True)
            self.options.result_dir.mkdir(parents=True, exist_ok=True)
            if (shutil.disk_usage(self.options.state_dir).free < self.options.sqlite_min_free_bytes or
                    shutil.disk_usage(self.options.result_dir).free < self.options.sqlite_min_free_bytes):
                return False
            with tempfile.NamedTemporaryFile(prefix=".runtime-probe-", suffix=".sqlite3",
                                             dir=self.options.state_dir, delete=False) as handle:
                probe = Path(handle.name)
            db = sqlite3.connect(probe, timeout=2)
            try:
                if db.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() != "wal":
                    return False
                db.execute("PRAGMA synchronous=FULL")
                if db.execute("PRAGMA synchronous").fetchone()[0] != 2:
                    return False
                db.execute("CREATE TABLE IF NOT EXISTS probe (id INTEGER PRIMARY KEY, value TEXT)")
                db.execute("INSERT OR REPLACE INTO probe VALUES (1, 'ok')")
                db.commit()
                if db.execute("SELECT value FROM probe WHERE id=1").fetchone() != ("ok",):
                    return False
                return db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone() == (0, 0, 0)
            finally:
                db.close()
        except (OSError, sqlite3.Error):
            return False
        finally:
            if probe is not None:
                try:
                    probe.unlink()
                    probe.with_suffix(probe.suffix + "-wal").unlink(missing_ok=True)
                    probe.with_suffix(probe.suffix + "-shm").unlink(missing_ok=True)
                except OSError:
                    pass

    def _external(self) -> dict[str, Any]:
        if self._fenced or self.proof is None or self.daemon is None or self.prepared is None:
            return {name: False for name in ("sqlite", "gpu", "artifacts", "adapter", "ollama", "profile", "cleanup")}
        lifecycle = self.core.snapshot()
        return {
            "sqlite": True, "gpu": True, "artifacts": True,
            "profile": True, "ollama": True,
            "cleanup": not self._cleanup_errors,
            "adapter": lifecycle.phase == "stable" and lifecycle.available and lifecycle.session_present,
        }

    def _probes(self) -> dict[str, Callable[[], Any]]:
        def artifacts() -> bool:
            try:
                current = self.config.artifact_root / "current"
                selected = os.readlink(current)
                if selected != self.config.manifest_sha256:
                    return False
                document = _read_manifest(self.config.artifact_root, selected)
                return document["manifest_sha256"] == self.config.manifest_sha256
            except (OSError, ValueError, KeyError):
                return False

        def gpu() -> bool:
            value = self.proof.identity()
            if hasattr(value, "__await__"):
                return asyncio.run(value) == self.config.gpu_uuid
            return value == self.config.gpu_uuid

        async def ollama() -> bool:
            try:
                await self.daemon.health()
                return True
            except Exception:
                return False

        def profile() -> bool:
            # Bindings pin exact profile identity/shape.  Re-run the startup
            # preflight in a bounded worker so SQLite never blocks aiohttp.
            if self.prepared is None:
                return False
            try:
                # Existing read-only connection is thread-affine; opening a
                # temporary read-only binding in the worker is the safe probe.
                from services.llm.resource_manager.profiles import ProfileStore
                if observe_runtime_identities(self._ollama_version) != {
                        model: config.runtime_identity for model, config in self.config.models.items()}:
                    return False
                prepared = self.prepared
                if prepared is None:
                    return False
                store = ProfileStore.open_readonly(self.config.profile_db)
                try:
                    for model in prepared.bindings:
                        expected = prepared.profiles[model]
                        kwargs = ({"context_size": expected.context_size} if expected.context_size is not None
                                  else {"bucket_identity": expected.bucket_identity})
                        observed = store.lookup(model, expected.gpu_uuid, expected.artifact_manifest_hash,
                                                expected.model_hash, expected.runtime_identity,
                                                expected.adapter_identity, **kwargs)
                        if observed != expected:
                            return False
                finally:
                    store.close()
                return True
            except Exception:
                return False

        async def adapter() -> bool:
            state, revision, ok = await self.core.probe_active_dependency()
            return ok and self.core.snapshot().revision == revision

        return {"sqlite": self._state_probe, "gpu": gpu, "ollama": ollama,
                "artifacts": artifacts, "profile": profile, "adapter": adapter}

    def _typed_proof(self) -> GPUProof:
        if self.proof is None:
            raise RuntimeError("GPU proof has not been captured")
        def daemon_ownership_snapshot():
            # The daemon receives this same proof object.  Its ownership probe
            # must fail closed until startup has established daemon authority.
            if self.daemon is None or not self._daemon_started:
                raise RuntimeError("Ollama ownership snapshot is unavailable before daemon startup")
            return self.daemon.ownership_snapshot()

        return GPUProof(self.proof.identity, self.proof.cleanup, self.proof.residency,
                        self.proof.supervisor_identity, self.proof.residency_for_runner,
                        self.proof.memory, daemon_ownership_snapshot)

    async def _fence(self, reason: str) -> None:
        self._fenced = True
        self.core.fence_shutdown(reason=reason)

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("runtime is one-shot")
        self._started = True
        self._startup_task = asyncio.current_task()
        try:
            if not await asyncio.to_thread(self._state_probe):
                raise RuntimeError("state SQLite/free-space preflight failed")
            if self._fenced: raise RuntimeError("runtime is fenced")
            # This is deliberately the first provider-adjacent operation.
            if not self.options.host_pid_namespace:
                raise RuntimeError("explicit host PID namespace attestation is required")
            self.proof = await asyncio.to_thread(self._capture, self.config.gpu_uuid, os.getpid(),
                                                  host_pid_namespace=self.options.host_pid_namespace)
            if self._fenced: raise RuntimeError("runtime is fenced")
            typed_proof = self._typed_proof()
            # Preflight the exact immutable profiles before constructing either
            # the owned daemon or runtime providers.  In particular, Ollama's
            # process concurrency is selected from the measured profile, not
            # from a speculative/default p=1 setting.
            configured_ollama = self.config.models["SmolLM"].runtime_identity
            if not configured_ollama.startswith("ollama:") or not configured_ollama[7:]:
                raise ValueError("configured SmolLM runtime identity is not an Ollama version")
            expected_identities = observe_runtime_identities(configured_ollama[7:])
            self.prepared = await prepare_bindings(self.config, typed_proof, expected_identities)
            if self._fenced: raise RuntimeError("runtime is fenced")
            smollm_capacity = self.prepared.profiles[ModelId.SMOLLM].optimal_parallelism
            self.daemon = self._daemon_factory(self.config, typed_proof, num_parallel=smollm_capacity)
            version = await self.daemon.start()
            self._daemon_started = True
            self._ollama_version = version
            if self._fenced: raise RuntimeError("runtime is fenced")
            observed = observe_runtime_identities(version)
            if observed != expected_identities:
                raise ValueError("runtime identity changed after measured profile preflight")
            self.result_store = ResultStore(self.options.result_dir)
            self.http = ResourceManagerHttpServer(self.core, bindings=dict(self.prepared.bindings),
                                                  result_store=self.result_store)
            self.health = ResourceManagerHealthBoundary(self.core, self._external, probes=self._probes(), max_concurrent=8)
            register_routes(self.http.app, self.health)
            self._install_admission_gate()
            self.runner = web.AppRunner(self.http.app)
            await self.runner.setup()
            if self._fenced: raise RuntimeError("runtime is fenced")
            self.site = web.TCPSite(self.runner, self.options.host, self.options.port)
            await self.site.start()
            if self._fenced: raise RuntimeError("runtime is fenced")
            self._monitor = asyncio.create_task(self._monitor_daemon())
        except BaseException:
            # ``stop`` waits for a concurrent startup owner before releasing
            # resources.  Once this task has received its startup failure or
            # cancellation, it can no longer acquire another resource, so it
            # must relinquish that ownership before joining shared cleanup.
            # Otherwise a cancelled start awaiting stop and stop awaiting the
            # start reference form a shutdown-grace-length cycle.
            self._startup_task = None
            await self.stop()
            raise
        finally:
            self._startup_task = None

    async def _monitor_daemon(self) -> None:
        try:
            while await self.daemon.alive():
                await asyncio.sleep(.05)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        if not self._fenced:
            await self._fence("ollama_lost")
            # Initiate cleanup even for callers using start() rather than run().
            # Do not join it here: cleanup collects this monitor.
            if self._stop_task is None:
                self._stop_task = asyncio.create_task(self._stop_impl())
                self._stop_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)

    def _install_admission_gate(self) -> None:
        @web.middleware
        async def gate(request: web.Request, handler):
            is_start = request.method == "POST" and request.path == "/resource-manager/sessions"
            is_submit = request.method == "POST" and "/submissions" in request.path
            if not (is_start or is_submit):
                return await handler(request)
            before = self.core.snapshot()
            states = await self.health.dependencies()
            required = ("sqlite", "gpu", "artifacts", "profile", "ollama")
            if is_submit: required += ("adapter",)
            after = self.core.snapshot()
            if (self._fenced or not after.available or before.revision != after.revision or
                    any(not states[name].ok for name in required)):
                return web.json_response({"error": {"code": "dependencies_unavailable",
                    "message": "runtime dependencies are unavailable", "retryable": False}}, status=503)
            response = await handler(request)
            if self._fenced:
                return web.json_response({"error": {"code": "dependencies_unavailable",
                    "message": "runtime dependencies changed", "retryable": False}}, status=503)
            return response
        self.http.app.middlewares.append(gate)

    async def stop(self) -> None:
        self._fenced = True
        self.core.fence_shutdown(reason="runtime_stopped")
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop_impl())
        cancelled = False
        while True:
            try:
                await asyncio.shield(self._stop_task)
                break
            except asyncio.CancelledError:
                if self._stop_task.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _stop_impl(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.options.shutdown_grace_seconds

        async def stage(operation) -> None:
            task = asyncio.ensure_future(operation)
            self._retained.add(task)
            def finished(done):
                self._retained.discard(done)
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(finished)
            try:
                await asyncio.wait_for(asyncio.shield(task), max(0, deadline - asyncio.get_running_loop().time()))
            except BaseException as exc:
                self._cleanup_errors.append(exc)

        # A concurrent start must finish acquiring (or fail) before resources
        # are released. Its exception path clears this reference before stop.
        startup = self._startup_task
        if startup is not None and not startup.done():
            startup.cancel()
            # Do not await startup here: its cancellation path awaits this task.
            while self._startup_task is not None and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(.005)
            if self._startup_task is not None:
                self._cleanup_errors.append(TimeoutError("startup ownership remains pending"))
        if self.site is not None:
            await stage(self.site.stop())
        if self._monitor and self._monitor is not asyncio.current_task():
            self._monitor.cancel()
            await stage(asyncio.gather(self._monitor, return_exceptions=True))
        await stage(self.core.shutdown())
        if self.runner is not None:
            await stage(self.runner.cleanup())
        if self.health is not None:
            await stage(self.health.close())
        if self.daemon is not None:
            await stage(self.daemon.close())
            self._daemon_started = False
        if self.prepared is not None:
            try:
                self.prepared.close()
                self.prepared = None
            except BaseException as exc:
                self._cleanup_errors.append(exc)
        if self._cleanup_errors:
            raise RuntimeError("runtime cleanup remains unproved") from self._cleanup_errors[0]

    async def run(self) -> None:
        await self.start()
        try:
            if self._monitor is None:
                raise RuntimeError("daemon monitor was not started")
            await self._monitor
        finally:
            await self.stop()
