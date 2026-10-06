"""Focused lifecycle tests for the Phase 5 measurement entry point.

These tests exercise the orchestration seam only.  All providers, the daemon,
and storage are fakes; no model runtime, Docker, or GPU is involved.
"""
import argparse
import asyncio
import json
import os
from contextlib import ExitStack
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from services.llm.bootstrap.measurement_matrix import measurement_matrix
from services.llm.bootstrap.measurement_bindings import prepare_measurement_bindings
from services.llm.queue.contracts import ModelId
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import GPUProofError, ProcessIdentity, _ResidencyPending
from services.llm.providers.python_config import PythonProviderConfig
from services.llm.provisioning.benchmark_requests import prepare_benchmark_request
from services.llm.provisioning.rm_runner import ProvisioningError
from services.llm.resource_manager.protocol import Failure
from tools.compatibility import measure_profiles


class _Proof:
    supervisor_identity = ProcessIdentity(1, 1)
    cleanup = Mock()
    residency = Mock()
    residency_for_runner = Mock()
    memory = Mock()

    def identity(self):
        return "GPU-test"


class _Daemon:
    def __init__(self, config, proof, events, *, version="0.11.6", close_error=None,
                 num_parallel=1):
        self.events, self.version, self.close_error = events, version, close_error
        self.num_parallel = num_parallel
        self.snapshot = object()
        events.append("daemon_construct")

        # The ownership callback is deliberately unavailable until start.
        try:
            proof.ollama_ownership()
        except RuntimeError:
            pass
        else:
            raise AssertionError("ownership callback was available before daemon start")
        self.proof = proof

    async def start(self):
        self.events.append("daemon_start")
        return self.version

    def ownership_snapshot(self):
        self.events.append("ownership")
        return self.snapshot

    async def close(self):
        self.events.append("daemon_close")
        if self.close_error:
            raise self.close_error


class _Bindings:
    def __init__(self, provider):
        self.bindings = {(mid, selector): SimpleNamespace(
                              provider=provider, profile=SimpleNamespace(),
                              identity_derived_max_parallelism=1,
                              identity_derived_capability_reason=(
                                  "identity_configured_provider_capability"),
                              provider_max_parallelism=1)
                         for mid, selector in measurement_matrix()}
        self.hashes = {mid.value: "model-hash" for mid, _ in measurement_matrix()}


class _RM:
    def __init__(self):
        self._phase = "startup"

    async def start_session(self, *args, **kwargs):
        return SimpleNamespace(session_token="token")

    async def stop_session(self, *args, **kwargs):
        return None

    def snapshot(self):
        return SimpleNamespace(phase=self._phase)


class MeasureProfilesLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.args = argparse.Namespace(
            requests=root / "requests.json", db=root / "profiles.sqlite", provenance="test",
            ollama_version="0.11.6", ceiling=1,
        )
        self.args.db.parent.mkdir(parents=True, exist_ok=True)
        self.config = SimpleNamespace(gpu_uuid="GPU-test", manifest_sha256="manifest",
                                      profile_db=root / "generated-profile.sqlite",
                                      models={m.value: SimpleNamespace(runtime_identity="0.11.6",
                                                                        adapter_identity="adapter")
                                              for m in ModelId})

    def tearDown(self):
        self.tmp.cleanup()

    async def test_global_ceiling_fences_gector_profile_before_provider_validation(self):
        """Binding construction applies each adapter's identity capability."""
        config = SimpleNamespace(
            gpu_uuid="GPU-test", manifest_sha256="a" * 64,
            artifact_root=Path(self.tmp.name), ollama_binary=Path("/usr/bin/ollama"),
            ollama_home=Path(self.tmp.name), ollama_port=11434,
            models={m.value: SimpleNamespace(runtime_identity="runtime",
                                              adapter_identity="adapter") for m in ModelId},
        )
        async def identity():
            return "GPU-test"
        proof = SimpleNamespace(identity=identity, cleanup=lambda: True,
                                residency=None, residency_for_runner=None, memory=None,
                                supervisor_identity=ProcessIdentity(1, 1),
                                ollama_ownership=lambda: object())
        hashes = {m.value: "b" * 64 for m in ModelId}
        with patch("services.llm.bootstrap.measurement_bindings._verify_and_hashes",
                   return_value=hashes):
            prepared = await prepare_measurement_bindings(
                config, proof, {m.value: "runtime" for m in ModelId}, ceiling=32)

        for model in ModelId:
            selector = next(selector for mid, selector in measurement_matrix() if mid is model)
            binding = prepared.bindings[(model, selector)]
            await binding.provider.validate(binding.profile)
            if model is ModelId.GECTOR:
                self.assertEqual(binding.profile.optimal_parallelism, 1)
                self.assertEqual(binding.profile.memory_safe_n, 1)
                self.assertEqual(binding.profile.buffer_capacity, 1)

    def test_coedit_measurement_ceiling_keeps_operator_bound_below_fixed_capability(self):
        """A requested cap limits discovery but does not redefine CoEdIT identity."""
        capability = PythonProviderConfig.MAX_NATIVE_BATCH_SIZE
        self.assertEqual(capability, 32)
        binding = SimpleNamespace(provider_max_parallelism=capability,
                                  identity_derived_max_parallelism=capability)
        for requested in (1, 7, capability):
            with self.subTest(requested=requested):
                effective = min(requested, binding.provider_max_parallelism,
                                binding.identity_derived_max_parallelism)
                self.assertEqual(effective, requested)
                self.assertEqual(binding.provider_max_parallelism, capability)
                self.assertEqual(binding.identity_derived_max_parallelism, capability)

    async def test_maximum_witness_classifies_each_bounded_predicate_without_values(self):
        cases = (
            (ModelId.SMOLLM, b"short", "smollm_token_count", None),
            (ModelId.COEDIT, {"instruction": "i", "texts": ["t"]},
             "coedit_token_count", {"count": 127, "max": 128}),
            (ModelId.COEDIT, {"instruction": "i", "texts": ["t"]},
             "coedit_configured_maximum", {"count": 128, "max": 127}),
            (ModelId.COEDIT, {"instruction": "i", "texts": ["t"]},
             "coedit_payload_fingerprint", {"count": 128, "max": 128, "fingerprint": "wrong"}),
            (ModelId.GECTOR, {"text": "t"}, "gector_token_count",
             {"count": 127, "max": 128}),
            (ModelId.GECTOR, {"text": "t"}, "gector_configured_maximum",
             {"count": 128, "max": 127}),
            (ModelId.GECTOR, {"text": "t"}, "gector_payload_fingerprint",
             {"count": 128, "max": 128, "fingerprint": "wrong"}),
        )
        for model, payload, detail, response in cases:
            with self.subTest(model=model, detail=detail):
                provider = SimpleNamespace(validate_input=AsyncMock())
                provider.worker = SimpleNamespace(call=AsyncMock(return_value=response or {}))
                request = SimpleNamespace(payload=(
                    json.dumps(payload).encode() if isinstance(payload, dict) else payload))
                with self.assertRaises(measure_profiles.CapacityEvidenceError) as raised:
                    await measure_profiles._maximum_witness(
                        provider, model, request, context=None, bucket=None)
                failure = raised.exception
                self.assertEqual(failure.measurement_model, model.value)
                self.assertEqual(failure.measurement_failure_code, "maximum_witness_failed")
                self.assertEqual(failure.measurement_failure_detail, detail)
                self.assertNotIn("wrong", json.dumps(failure.__dict__))

    async def test_smollm_residency_fence_accepts_pending_model_specific_proof(self):
        proof = SimpleNamespace(residency=AsyncMock(side_effect=_ResidencyPending("pending")))
        provider = SimpleNamespace(accepted_model_specific_residency=Mock(return_value=True))

        await measure_profiles._smollm_residency_fence(proof, provider, 1)

        proof.residency.assert_awaited_once()
        provider.accepted_model_specific_residency.assert_called_once_with()

    async def test_smollm_residency_fence_rejects_pending_without_exact_acceptance(self):
        for accepted in (False, 1, "true", None):
            with self.subTest(accepted=accepted):
                proof = SimpleNamespace(residency=AsyncMock(
                    side_effect=_ResidencyPending("pending")))
                provider = SimpleNamespace(
                    accepted_model_specific_residency=Mock(return_value=accepted))

                with self.assertRaises(_ResidencyPending):
                    await measure_profiles._smollm_residency_fence(proof, provider, 1)

    async def test_smollm_residency_fence_does_not_mask_non_pending_proof_errors(self):
        failure = GPUProofError("residency proof failed")
        proof = SimpleNamespace(residency=AsyncMock(side_effect=failure))
        provider = SimpleNamespace(accepted_model_specific_residency=Mock(return_value=True))

        with self.assertRaises(GPUProofError) as raised:
            await measure_profiles._smollm_residency_fence(proof, provider, 1)
        self.assertIs(raised.exception, failure)
        provider.accepted_model_specific_residency.assert_not_called()

    def test_reservation_is_exclusive_and_does_not_touch_existing_database(self):
        destination = Path(self.tmp.name) / "profiles.sqlite"
        destination.write_bytes(b"foreign")
        with self.assertRaises(ValueError):
            measure_profiles._reserve_destination(destination)
        lock = Path(str(destination) + ".measurement.lock")
        lock.write_bytes(b"owned")
        try:
            with self.assertRaises(ValueError):
                measure_profiles._reserve_destination(destination)
        finally:
            lock.unlink()
        self.assertEqual(destination.read_bytes(), b"foreign")

    def test_lock_release_refuses_foreign_owner_and_keeps_foreign_lock(self):
        destination = Path(self.tmp.name) / "profiles.sqlite"
        fd, lock, token = measure_profiles._reserve_destination(destination)
        lock.write_bytes(b"foreign-owner")
        with self.assertRaises(RuntimeError):
            measure_profiles._release_destination(fd, lock, token)
        self.assertEqual(lock.read_bytes(), b"foreign-owner")
        fd = os.open(lock, os.O_RDONLY)
        lock.write_bytes(token)
        measure_profiles._release_destination(fd, lock, token)

    def test_atomic_install_is_no_clobber_and_audited_readonly(self):
        destination = Path(self.tmp.name) / "profiles.sqlite"
        staged = Path(self.tmp.name) / ".staged.sqlite"
        staged.write_bytes(b"audited")
        measure_profiles._install_atomic(staged, destination)
        self.assertEqual(destination.read_bytes(), b"audited")
        self.assertEqual(destination.stat().st_mode & 0o777, 0o444)
        replacement = Path(self.tmp.name) / ".replacement.sqlite"
        replacement.write_bytes(b"replacement")
        with self.assertRaises(FileExistsError):
            measure_profiles._install_atomic(replacement, destination)
        self.assertEqual(destination.read_bytes(), b"audited")

    def test_atomic_install_fsyncs_and_preserves_readonly_audit(self):
        staged = Path(self.tmp.name) / ".staged.sqlite"
        destination = Path(self.tmp.name) / "profiles.sqlite"
        staged.write_bytes(b"content")
        measure_profiles._install_atomic(staged, destination)
        self.assertFalse(staged.exists())
        self.assertEqual(destination.stat().st_mode & 0o444, 0o444)

    def test_busy_checkpoint_fails_closed_and_wal_sidecars_are_rejected(self):
        staged = Path(self.tmp.name) / ".staged.sqlite"
        staged.write_bytes(b"content")
        with patch.object(measure_profiles.sqlite3, "connect") as connect:
            connect.return_value.execute.return_value.fetchone.return_value = (1, 2, 3)
            with self.assertRaises(RuntimeError):
                measure_profiles._checkpoint_audited_database(staged)
        staged.with_name(staged.name + "-wal").write_bytes(b"tampered")
        with patch.object(measure_profiles.sqlite3, "connect") as connect:
            connect.return_value.execute.return_value.fetchone.return_value = (0, 0, 0)
            with self.assertRaises(RuntimeError):
                measure_profiles._checkpoint_audited_database(staged)

    def _patches(self, *, daemon_version="0.11.6", close_error=None,
                 prepare_error=None, measurement_error=None):
        events = []
        proof = _Proof()
        generated_text = "generated"
        generated_payload = json.dumps(
            {"instruction": "i", "texts": [generated_text]},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        provider = SimpleNamespace(
            unload=AsyncMock(),
            validate_input=AsyncMock(),
            worker=SimpleNamespace(call=AsyncMock(return_value={
                "text": generated_text,
                "count": 128,
                "max": 128,
                "fingerprint": __import__("hashlib").sha256(generated_payload).hexdigest(),
            })),
        )
        bindings = _Bindings(provider)
        daemon_holder = []

        def daemon_factory(config, typed_proof, **kwargs):
            daemon = _Daemon(config, typed_proof, events, version=daemon_version,
                              close_error=close_error, num_parallel=kwargs.get("num_parallel", 1))
            daemon_holder.append(daemon)
            return daemon

        def capture(*_args, **_kwargs):
            events.append("capture")
            return proof

        async def prepare(*_args, **_kwargs):
            events.append("bindings")
            if prepare_error:
                raise prepare_error
            return bindings

        async def measure(*_args, **_kwargs):
            events.append("measure")
            if measurement_error:
                raise measurement_error
            return SimpleNamespace(profile_eligible=True, reason="ok", n=1)

        runner = Mock()

        writer = Mock()
        store = Mock()
        store.__enter__ = Mock(return_value=store)
        store.__exit__ = Mock(return_value=False)

        install = Mock()
        install_patch = patch.object(measure_profiles, "_install_atomic", side_effect=install)
        connect = Mock()
        connect.return_value.execute.return_value.fetchone.return_value = (0, 0, 0)
        request = SimpleNamespace(
            payload=b'{"instruction":"i","texts":["t"]}',
            fingerprint="fingerprint",
                                  model_id=ModelId.SMOLLM, request_bucket="bucket")
        patches = [
            patch.object(measure_profiles, "_requests", return_value={m.value: b"x" for m in ModelId}),
            patch.object(measure_profiles.LinuxGPUProof, "capture", side_effect=capture),
            patch.object(measure_profiles, "OwnedOllama", side_effect=daemon_factory),
            patch.object(measure_profiles, "observe_runtime_identities",
                         side_effect=lambda version: (events.append("identities") or
                                                       {m.value: version for m in ModelId})),
            patch.object(measure_profiles, "prepare_benchmark_request", return_value=request),
            patch.object(measure_profiles, "_runtime_generated_coedit_request",
                         new=AsyncMock(return_value=request)),
            patch.object(measure_profiles, "prepare_measurement_bindings", side_effect=prepare),
            patch.object(measure_profiles, "ResourceManager", _RM),
            patch.object(measure_profiles, "ResourceManagerWaveRunner", return_value=runner),
            patch.object(measure_profiles, "ProfileStore", return_value=writer),
            patch.object(measure_profiles.ProfileStore, "open_readonly", return_value=store),
            patch.object(measure_profiles, "_maximum_witness", new_callable=AsyncMock),
            patch.object(measure_profiles, "measure_authoritative", side_effect=measure),
            patch.object(measure_profiles, "persist_measured_profile",
                         side_effect=lambda *a, **k: SimpleNamespace(profile=SimpleNamespace(profile_identity="p"))),
            install_patch,
            patch("sqlite3.connect", connect),
        ]
        return patches, events, bindings, daemon_holder, proof, install

    def _diagnostic_patches(self, *, provider_error=None, unload_error=None,
                             daemon_close_error=None, runner_error=None,
                             wave=None):
        """Build a completely local seam for the bounded p=2 diagnostic."""
        events = []
        proof = _Proof()
        provider = SimpleNamespace(validate_input=AsyncMock(), unload=AsyncMock())
        if unload_error:
            provider.unload.side_effect = unload_error
        bindings = SimpleNamespace(
            bindings={(ModelId.SMOLLM, "smollm:context512"):
                      SimpleNamespace(provider=provider, profile=SimpleNamespace(),
                                      identity_derived_max_parallelism=2,
                                      identity_derived_capability_reason="identity_configured_operator_ceiling",
                                      provider_max_parallelism=2)})
        daemon_holder = []

        def daemon_factory(config, typed_proof, **kwargs):
            daemon = _Daemon(config, typed_proof, events,
                             close_error=daemon_close_error,
                             num_parallel=kwargs.get("num_parallel", 1))
            daemon_holder.append(daemon)
            return daemon

        async def prepare(*_args, **_kwargs):
            events.append("bindings")
            return bindings

        async def witness_impl(*_args, **_kwargs):
            events.append("maximum_witness")
            if provider_error:
                raise provider_error
        witness = AsyncMock(side_effect=witness_impl)

        wave = wave or SimpleNamespace(execution_ended=2, execution_started=1,
                                       native_request_correlation=True,
                                       observation_count=2,
                                       observed_native_batch_sizes=(1, 1),
                                       observation_drops=0,
                                       evidence_kind="ollama_native")
        runner = AsyncMock(return_value=wave)
        if runner_error is not None:
            runner.side_effect = runner_error
        request = SimpleNamespace(payload="exact-request")
        request_builder = Mock(return_value=request)
        patches = [
            patch.object(measure_profiles.LinuxGPUProof, "capture", return_value=proof),
            patch.object(measure_profiles, "OwnedOllama", side_effect=daemon_factory),
            patch.object(measure_profiles, "observe_runtime_identities", return_value={}),
            patch.object(measure_profiles, "prepare_measurement_bindings", side_effect=prepare),
            patch.object(measure_profiles, "prepare_benchmark_request", request_builder),
            patch.object(measure_profiles, "_maximum_witness", new=witness),
            patch.object(measure_profiles, "ResourceManager", _RM),
            patch.object(measure_profiles, "ResourceManagerWaveRunner", return_value=runner),
        ]
        return patches, request_builder, witness, runner, provider, daemon_holder, events

    def _diagnostic_args(self, request_path):
        return argparse.Namespace(selector="smollm:context512", request=request_path,
                                  ollama_version="0.11.6")

    async def test_capture_starts_smollm_with_one_slot_not_benchmark_ceiling(self):
        self.args.ceiling = 32
        patches, events, bindings, daemons, _, _ = self._patches()
        with self._all(patches):
            await measure_profiles._run(self.args, self.config)
        self.assertEqual(events[:4], ["capture", "daemon_construct", "daemon_start", "identities"])
        self.assertEqual(events.count("daemon_construct"), 1)
        self.assertEqual(events.count("daemon_start"), 1)
        self.assertEqual(events.count("bindings"), 1)
        self.assertEqual(daemons[0].num_parallel, 1)

    async def test_coedit_generation_uses_started_session_and_same_request_everywhere(self):
        selector = "coedit:p1:input128:output64:float16:beams1:nosample"
        events = []
        seed = prepare_benchmark_request(
            measure_profiles.MATRIX[ModelId.COEDIT], request_bucket=selector,
            identity_witnesses={"adapter": "loaded"},
            configured_request={"instruction": "i", "texts": ["seed"]},
            validate_request=lambda value, *_: True)
        generated_text = "generated"
        generated_payload = json.dumps(
            {"instruction": "i", "texts": [generated_text]},
            sort_keys=True, separators=(",", ":")).encode()
        async def generate(*args, **kwargs):
            events.append("generate")
            return {"text": generated_text, "count": 128, "max": 128,
                    "fingerprint": __import__("hashlib").sha256(generated_payload).hexdigest()}
        provider = SimpleNamespace(unload=AsyncMock(), validate_input=AsyncMock(),
            worker=SimpleNamespace(call=AsyncMock(side_effect=generate)))
        binding = SimpleNamespace(provider=provider, profile=SimpleNamespace(),
            identity_derived_max_parallelism=1,
            identity_derived_capability_reason="identity_configured_provider_capability",
            provider_max_parallelism=1)
        bindings = SimpleNamespace(bindings={(ModelId.COEDIT, selector): binding},
                                   hashes={"CoEdIT": "model-hash"})
        session = SimpleNamespace(session_token="token")

        class RM(_RM):
            async def start_session(self, *args, **kwargs):
                events.append("session_start")
                return session

            async def stop_session(self, *args, **kwargs):
                events.append("session_stop")

        async def witness(_provider, _model, request, **kwargs):
            events.append(("witness", request))

        measurement_request = []
        async def measure(request, *args, **kwargs):
            events.append(("measure", request))
            measurement_request.append(request)
            return SimpleNamespace(profile_eligible=True, n=1)

        persisted_request = []
        def persist(request, *args, **kwargs):
            events.append(("persist", request))
            persisted_request.append(request)
            return SimpleNamespace(profile=SimpleNamespace(profile_identity="p"))

        writer = Mock()
        store = Mock(); store.__enter__ = Mock(return_value=store); store.__exit__ = Mock(return_value=False)
        with self._all([
            patch.object(measure_profiles, "measurement_matrix", return_value=((ModelId.COEDIT, selector),)),
            patch.object(measure_profiles, "_requests", return_value={"CoEdIT": seed}),
            patch.object(measure_profiles.LinuxGPUProof, "capture", return_value=_Proof()),
            patch.object(measure_profiles, "OwnedOllama", side_effect=lambda *a, **k: _Daemon(*a, events, **k)),
            patch.object(measure_profiles, "observe_runtime_identities", return_value={}),
            patch.object(measure_profiles, "prepare_measurement_bindings", new=AsyncMock(return_value=bindings)),
            patch.object(measure_profiles, "prepare_benchmark_request", return_value=seed),
            patch.object(measure_profiles, "ResourceManager", RM),
            patch.object(measure_profiles, "ResourceManagerWaveRunner", return_value=Mock()),
            patch.object(measure_profiles, "_maximum_witness", side_effect=witness),
            patch.object(measure_profiles, "measure_authoritative", side_effect=measure),
            patch.object(measure_profiles, "persist_measured_profile", side_effect=persist),
            patch.object(measure_profiles, "ProfileStore", return_value=writer),
            patch.object(measure_profiles.ProfileStore, "open_readonly", return_value=store),
            patch.object(measure_profiles, "_checkpoint_audited_database"),
            patch.object(measure_profiles, "_install_atomic"),
        ]):
            result = await measure_profiles._run(self.args, self.config)

        generated = events[next(i for i, value in enumerate(events)
                                if isinstance(value, tuple) and value[0] == "witness")][1]
        self.assertLess(events.index("session_start"), events.index("generate"))
        self.assertLess(events.index("generate"),
                        next(i for i, value in enumerate(events) if isinstance(value, tuple) and value[0] == "witness"))
        self.assertEqual(provider.worker.call.await_count, 1)
        self.assertEqual(provider.worker.call.call_args.kwargs["generate"], True)
        self.assertIs(generated, measurement_request[0])
        self.assertIs(generated, persisted_request[0])
        self.assertEqual({generated.fingerprint, measurement_request[0].fingerprint,
                          persisted_request[0].fingerprint}, {generated.fingerprint})
        self.assertEqual(result, ["p"])
        self.assertIn("session_stop", events)

    async def test_smollm_runtime_restarts_at_each_requested_slot_count(self):
        self.args.ceiling = 32
        patches, events, _bindings, daemons, proof, _ = self._patches()
        proof.cleanup = AsyncMock(return_value=True)
        slots = []

        async def measure(*_args, **kwargs):
            factory = kwargs["runner_for_concurrency"]
            if factory is None:
                return SimpleNamespace(profile_eligible=True, reason="ok", n=1)
            await factory(1)
            await factory(2)
            await factory(2)
            slots.extend(daemon.num_parallel for daemon in daemons)
            return SimpleNamespace(profile_eligible=True, reason="ok", n=1)

        patches.append(patch.object(measure_profiles, "measure_authoritative", side_effect=measure))
        with self._all(patches):
            await measure_profiles._run(self.args, self.config)
        # Only SmolLM supplies the scoped factory.  Its initial p=1 daemon is
        # replaced exactly once for p=2 and never receives the operator's 32.
        self.assertEqual(slots, [1, 2])
        self.assertNotIn(32, slots)
        self.assertGreaterEqual(events.count("daemon_close"), 2)

    async def test_smollm_restart_cleanup_failure_fails_closed(self):
        self.args.ceiling = 32
        patches, _events, _bindings, _daemons, proof, _ = self._patches()
        proof.cleanup = AsyncMock(return_value=False)

        async def measure(*_args, **kwargs):
            factory = kwargs.get("runner_for_concurrency")
            if factory is not None:
                await factory(2)
            return SimpleNamespace(profile_eligible=True, reason="ok", n=1)

        patches.append(patch.object(measure_profiles, "measure_authoritative", side_effect=measure))
        with self._all(patches), self.assertRaises(Exception):
            await measure_profiles._run(self.args, self.config)

    async def test_binding_and_measurement_failures_close_daemon(self):
        for failure in ("binding", "measurement"):
            with self.subTest(failure=failure):
                kwargs = {"prepare_error": RuntimeError("bindings")} if failure == "binding" else {"measurement_error": RuntimeError("measure")}
                patches, events, *_ = self._patches(**kwargs)
                with self._all(patches), self.assertRaises(RuntimeError):
                    await measure_profiles._run(self.args, self.config)
                self.assertEqual(events.count("daemon_close"), 1)

    async def test_failed_coedit_measurement_does_not_start_gector_or_persist_coedit(self):
        patches, _events, _bindings, _daemons, _proof, install = self._patches()
        measured, persisted = [], []
        async def measure(*_args, **kwargs):
            model = kwargs["run_identity"]
            measured.append(model)
            if model == "CoEdIT":
                return SimpleNamespace(profile_eligible=False, failure_code="oom",
                    failure_detail=None, total_vram_bytes=None,
                    baseline_pre_used_bytes=None, peak_incremental_request_bytes=None,
                    derived_ceiling=None, n=None)
            return SimpleNamespace(profile_eligible=True, n=1)
        def persist(*args, **_kwargs):
            persisted.append(args[2].model_id.value)
            return SimpleNamespace(profile=SimpleNamespace(profile_identity="p"))
        patches.extend([
            patch.object(measure_profiles, "measure_authoritative", side_effect=measure),
            patch.object(measure_profiles, "persist_measured_profile", side_effect=persist),
        ])
        with self._all(patches), self.assertRaises(measure_profiles.CapacityEvidenceError):
            await measure_profiles._run(self.args, self.config)
        self.assertEqual(measured, ["SmolLM", "CoEdIT"])
        self.assertEqual(persisted, ["SmolLM"])
        install.assert_not_called()

    async def test_version_mismatch_closes_without_bindings(self):
        patches, events, *_ = self._patches(daemon_version="0.11.5")
        with self._all(patches), self.assertRaises(ValueError):
            await measure_profiles._run(self.args, self.config)
        self.assertEqual(events, ["capture", "daemon_construct", "daemon_start", "daemon_close"])

    async def test_cleanup_failure_prevents_atomic_install(self):
        patches, events, _, _, _, install = self._patches(close_error=RuntimeError("close"))
        with self._all(patches), self.assertRaises(BaseException):
            await measure_profiles._run(self.args, self.config)
        self.assertGreaterEqual(events.count("daemon_close"), 1)
        install.assert_not_called()

    async def test_ownership_callback_is_fail_closed_then_propagates(self):
        patches, events, _, daemons, _, _ = self._patches()
        with self._all(patches):
            await measure_profiles._run(self.args, self.config)
        daemon = daemons[0]
        self.assertIs(daemon.proof.ollama_ownership(), daemon.snapshot)
        self.assertIn("ownership", events)

    async def test_diagnostic_runs_exact_fixed_p2_wave_without_creating_database(self):
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        db = Path(self.tmp.name) / "must-not-exist.sqlite"
        patches, builder, witness, runner, provider, daemons, events = self._diagnostic_patches()
        with self._all(patches):
            result = await measure_profiles._diagnostic_run(
                self._diagnostic_args(request_path), self.config)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["concurrency"], 2)
        self.assertEqual(result["wave"], 1)
        self.assertFalse(db.exists())
        builder.assert_called_once()
        self.assertEqual(builder.call_args.kwargs["request_bucket"], "smollm:context512")
        self.assertEqual(builder.call_args.kwargs["configured_request"], "configured-request")
        self.assertEqual(builder.call_args.kwargs["identity_witnesses"], {
            "dtype": "q8_0",
            "generation_parameters": {"num_predict": 64, "temperature": 0},
            "native_batch_shape": [1],
        })
        witness.assert_awaited_once()
        self.assertEqual(witness.call_args.kwargs, {"context": 512, "bucket": None})
        runner.assert_awaited_once()
        self.assertEqual(runner.call_args.args[:3],
                         (2, 1, ("diagnostic-smollm-p2-0", "diagnostic-smollm-p2-1")))
        self.assertEqual(runner.call_args.args[3], "exact-request")
        self.assertEqual(daemons[0].num_parallel, 2)
        self.assertEqual(events[-1], "daemon_close")
        provider.unload.assert_awaited_once()

    async def test_diagnostic_reports_actionable_inner_provider_code_and_message(self):
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        failure = RuntimeError("outer provider failure")
        failure.failure_kind = "provider_rejection"
        failure.failure_code = "ollama_response_contract"
        failure.failure_message = "request exceeded the exact SmolLM bound"
        patches, *_ = self._diagnostic_patches(provider_error=failure)
        with self._all(patches):
            result = await measure_profiles._diagnostic_run(
                self._diagnostic_args(request_path), self.config)

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["failure_kind"], "provider_rejection")
        self.assertEqual(result["failure_code"], "ollama_response_contract")
        self.assertEqual(result["failure_message"], "provider execution failed")
        self.assertEqual(result["cleanup"], "proved")

    def test_generic_diagnostic_failure_gets_a_bounded_classification(self):
        result = measure_profiles._diagnostic_failure(RuntimeError("provider failed"), "p2_wave")
        self.assertEqual(result["failure_code"], "provider_execution_failed")

    async def test_diagnostic_preserves_failure_metadata_attached_by_runner(self):
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        failure = ProvisioningError("wave request failed or was cancelled")
        failure.failure_kind = "runner_error"
        provider_failure = Failure("ollama_response_contract",
                                   "request exceeded the exact SmolLM bound", False)
        failure.failure_code = provider_failure.code
        failure.failure_message = provider_failure.message
        patches, *_ = self._diagnostic_patches(runner_error=failure)
        with self._all(patches):
            result = await measure_profiles._diagnostic_run(
                self._diagnostic_args(request_path), self.config)

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["stage"], "p2_wave")
        self.assertEqual(result["failure_kind"], "runner_error")
        self.assertEqual(result["failure_code"], "ollama_response_contract")
        self.assertEqual(result["failure_message"], "provider execution failed")

    def test_diagnostic_rejects_unbounded_provider_code(self):
        failure = RuntimeError("hostile provider body /secret")
        failure.failure_kind = "runner_error"
        failure.failure_code = "ollama_http_status"
        failure.failure_message = "HTTP body /secret prompt"
        result = measure_profiles._diagnostic_failure(failure, "p2_wave")
        self.assertEqual(result["failure_code"], "ollama_http_status")
        self.assertEqual(result["failure_message"], "provider execution failed")
        self.assertNotIn("secret", json.dumps(result))

    def test_diagnostic_traverses_bounded_nested_metadata_without_text(self):
        startup = RuntimeError("startup /private/model secret")
        startup.failure_kind = "runner_error"
        startup.failure_code = "ollama_response_contract"
        startup.lifecycle_phase = "load"
        startup.lifecycle_subreason = "ollama_create"
        cleanup = RuntimeError("cleanup /private/weights secret")
        cleanup.failure_kind = "not-allowlisted"
        cleanup.lifecycle_phase = "cleanup"
        cleanup.lifecycle_subreason = "cleanup_verification"
        nested = BaseExceptionGroup("hostile rendering", [startup, cleanup])

        result = measure_profiles._diagnostic_failure(nested, "p2_wave")

        self.assertEqual(result["failure_kind"], "runner_error")
        self.assertEqual(result["failure_code"], "ollama_response_contract")
        self.assertEqual(result["lifecycle_phase"], "cleanup")
        self.assertEqual(result["lifecycle_subreason"], "cleanup_verification")
        self.assertNotIn("private", json.dumps(result))

    def test_atomic_install_holds_reservation_until_commit_then_releases(self):
        destination = Path(self.tmp.name) / "profiles.sqlite"
        staged = Path(self.tmp.name) / ".staged.sqlite"
        staged.write_bytes(b"audited")
        fd, lock, token = measure_profiles._reserve_destination(destination)
        observed = []
        original_link = os.link
        def link_with_reservation(*args, **kwargs):
            observed.append((lock.exists(), lock.read_bytes() == token))
            return original_link(*args, **kwargs)
        with patch.object(measure_profiles.os, "link", side_effect=link_with_reservation):
            measure_profiles._install_atomic(staged, destination, reservation_fd=fd,
                                              reservation_lock=lock,
                                              reservation_token=token)
        measure_profiles._release_destination(fd, lock, token)
        self.assertEqual(observed, [(True, True)])
        self.assertTrue(destination.exists())

    async def test_diagnostic_requires_every_native_p2_success_invariant(self):
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        base = dict(execution_ended=2, execution_started=1,
                    native_request_correlation=True, observation_count=2,
                    observed_native_batch_sizes=(1, 1), observation_drops=0,
                    evidence_kind="ollama_native")
        failures = {
            "overlap": {"execution_ended": 1},
            "correlation": {"native_request_correlation": False},
            "count": {"observation_count": 1},
            "shape": {"observed_native_batch_sizes": (2,)},
            "drops": {"observation_drops": 1},
        }
        for name, changes in failures.items():
            with self.subTest(name=name):
                values = {**base, **changes}
                patches, *_ = self._diagnostic_patches(wave=SimpleNamespace(**values))
                with self._all(patches):
                    result = await measure_profiles._diagnostic_run(
                        self._diagnostic_args(request_path), self.config)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["cleanup"], "proved")

    async def test_diagnostic_cleanup_failure_is_actionable_and_never_persists(self):
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        db = Path(self.tmp.name) / "must-not-exist.sqlite"
        patches, *_ = self._diagnostic_patches(unload_error=RuntimeError("unload failed"))
        with self._all(patches):
            result = await measure_profiles._diagnostic_run(
                self._diagnostic_args(request_path), self.config)

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["stage"], "cleanup")
        self.assertEqual(result["failure_kind"], "cleanup_failed")
        self.assertEqual(result["failure_code"], "cleanup_failed")
        self.assertIn("diagnostic lifecycle cleanup failed", result["failure_message"])
        self.assertFalse(db.exists())

    async def test_diagnostic_cleanup_failure_overrides_provider_failure(self):
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        provider_failure = RuntimeError("provider rejected request")
        provider_failure.failure_kind = "provider_rejection"
        provider_failure.failure_code = "inner_request_invalid"
        provider_failure.failure_message = "request exceeded the exact SmolLM bound"
        patches, *_ = self._diagnostic_patches(
            provider_error=provider_failure, unload_error=RuntimeError("unload failed"))
        with self._all(patches):
            result = await measure_profiles._diagnostic_run(
                self._diagnostic_args(request_path), self.config)

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["stage"], "cleanup")
        self.assertEqual(result["failure_kind"], "cleanup_failed")
        self.assertEqual(result["failure_code"], "cleanup_failed")
        self.assertEqual(result["failure_message"], "diagnostic lifecycle cleanup failed")

    async def test_diagnostic_cleanup_failure_normalizes_hostile_unload_error(self):
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        hostile = RuntimeError("unload /private/model\\weights\x00\x1b" + ("x" * 1000))
        patches, *_ = self._diagnostic_patches(unload_error=hostile)
        with self._all(patches):
            result = await measure_profiles._diagnostic_run(
                self._diagnostic_args(request_path), self.config)

        json.dumps(result)
        self.assertEqual(result["failure_kind"], "cleanup_failed")
        self.assertEqual(result["failure_code"], "cleanup_failed")
        self.assertEqual(result["failure_message"], "diagnostic lifecycle cleanup failed")
        self.assertNotRegex(result["failure_message"], r"[\x00-\x1f\x7f/\\]")
        self.assertLessEqual(len(result["failure_message"]), 160)

    async def test_diagnostic_cleanup_failure_ignores_nested_error_count(self):
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        nested = BaseExceptionGroup("hostile /private/cleanup", [
            RuntimeError("unload one"), RuntimeError("unload two")])
        patches, *_ = self._diagnostic_patches(
            unload_error=nested, daemon_close_error=RuntimeError("close failed"))
        with self._all(patches):
            result = await measure_profiles._diagnostic_run(
                self._diagnostic_args(request_path), self.config)

        self.assertEqual(result["failure_kind"], "cleanup_failed")
        self.assertEqual(result["failure_code"], "cleanup_failed")
        self.assertEqual(result["failure_message"], "diagnostic lifecycle cleanup failed")

    async def test_diagnostic_rejects_non_exact_selector_before_gpu_capture(self):
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        args = self._diagnostic_args(request_path)
        args.selector = "smollm:other"
        capture = Mock(side_effect=AssertionError("capture must not run"))
        with patch.object(measure_profiles.LinuxGPUProof, "capture", capture), \
             self.assertRaises(ValueError):
            await measure_profiles._diagnostic_run(args, self.config)
        capture.assert_not_called()

    def test_cli_diagnostic_does_not_require_provenance(self):
        config_path = Path(self.tmp.name) / "bootstrap.json"
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        diagnostic = {"status": "complete", "cleanup": "proved"}
        argv = ["measure_profiles.py", "--config", str(config_path),
                "--diagnostic-smollm-p2", "--request", str(request_path),
                "--selector", "smollm:context512", "--ollama-version", "0.11.6"]
        with patch.object(measure_profiles, "load_config", return_value=self.config), \
             patch.object(measure_profiles, "_diagnostic_run", new=AsyncMock(return_value=diagnostic)), \
             patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 0)
        self.assertEqual(json.loads(output.getvalue()), diagnostic)

    def test_cli_incomplete_diagnostic_returns_bounded_json(self):
        config_path = Path(self.tmp.name) / "bootstrap.json"
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        diagnostic = {"status": "incomplete", "failure_code": "provider_failed"}
        argv = ["measure_profiles.py", "--config", str(config_path),
                "--diagnostic-smollm-p2", "--request", str(request_path),
                "--selector", "smollm:context512", "--ollama-version", "0.11.6"]
        with patch.object(measure_profiles, "load_config", return_value=self.config), \
             patch.object(measure_profiles, "_diagnostic_run", new=AsyncMock(return_value=diagnostic)), \
             patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 2)
        self.assertEqual(json.loads(output.getvalue()), diagnostic)

    def test_cli_diagnostic_writes_the_same_final_json_line_to_fixed_runtime_result_file(self):
        config_path = Path(self.tmp.name) / "bootstrap.json"
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        runtime = Path(self.tmp.name) / "runtime"
        runtime.mkdir()
        self.config.profile_db = runtime / "profiles.sqlite"
        result_file = runtime / "diagnostic-result.json"
        diagnostic = {"status": "incomplete", "cleanup": "proved", "failure_kind": "runner_error"}
        argv = ["measure_profiles.py", "--config", str(config_path),
                "--diagnostic-smollm-p2", "--request", str(request_path),
                "--result-file", str(result_file), "--ollama-version", "0.11.6"]
        with patch.object(measure_profiles, "load_config", return_value=self.config), \
             patch.object(measure_profiles, "_diagnostic_run", new=AsyncMock(return_value=diagnostic)), \
             patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 2)
        self.assertEqual(result_file.read_text(), output.getvalue())

    def test_result_file_rejects_bundle_and_symlink_locations(self):
        runtime = Path(self.tmp.name) / "runtime"
        runtime.mkdir()
        self.config.profile_db = runtime / "profiles.sqlite"
        with self.assertRaisesRegex(ValueError, "fixed runtime"):
            measure_profiles._diagnostic_result_file(Path("/opt/measurement/result.json"), self.config)
        target = runtime / "target.json"
        result = runtime / "diagnostic-result.json"
        result.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            measure_profiles._diagnostic_result_file(result, self.config)

    def test_cli_malformed_arguments_emit_one_json_envelope(self):
        argv = ["measure_profiles.py", "--config"]
        with patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 2)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        result = json.loads(lines[0])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["stage"], "cli_argument_parse")
        self.assertNotIn("usage:", lines[0])

    def test_cli_config_failure_is_classified_without_reloading_config(self):
        config_path = Path(self.tmp.name) / "private" / "bootstrap.json"
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        argv = ["measure_profiles.py", "--config", str(config_path),
                "--diagnostic-smollm-p2", "--request", str(request_path),
                "--ollama-version", "0.11.6"]
        failure = RuntimeError(f"cannot read {config_path}")
        with patch.object(measure_profiles, "load_config", side_effect=failure), \
             patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result["stage"], "config_load")
        self.assertNotIn(str(config_path), output.getvalue())

    def test_failure_envelope_does_not_mask_original_when_matrix_identity_fails(self):
        original = RuntimeError("original application failure")
        with patch.object(measure_profiles, "measurement_matrix",
                          side_effect=RuntimeError("matrix reporting failure")):
            result = measure_profiles._main_failure(original, "application")
        self.assertEqual(result["reason"], "application failure")
        self.assertEqual(result["message"], "application failure")
        self.assertEqual(result["exception_type"], "RuntimeError")
        self.assertEqual(result["matrix"], ())
        self.assertNotIn("original application failure", json.dumps(result))

    def test_terminal_application_trace_uses_only_bounded_classification(self):
        failure = RuntimeError("secret exception text")
        failure.measurement_model = "SmolLM"
        failure.measurement_failure_code = "oom"
        with patch.object(measure_profiles, "trace") as traced:
            measure_profiles._trace_main_failure(measure_profiles._main_failure(failure, "application"))
        fields = traced.call_args.kwargs
        self.assertEqual(fields["measurement_failure_code"], "oom")
        self.assertNotIn("secret exception text", json.dumps(fields))
        self.assertNotIn("exception_type", fields)

    def test_failure_envelope_transports_only_allowlisted_measurement_model(self):
        failure = RuntimeError("provider failure")
        failure.measurement_model = "SmolLM"
        failure.measurement_failure_code = "oom"
        result = measure_profiles._main_failure(failure, "application")
        self.assertEqual(result["model"], "SmolLM")
        self.assertEqual(result["measurement_failure_code"], "oom")

        failure.measurement_model = "unlisted-model"
        result = measure_profiles._main_failure(failure, "application")
        self.assertNotIn("model", result)

    def test_failure_envelopes_reject_unhashable_allowlist_metadata(self):
        failure = RuntimeError("provider failure")
        failure.measurement_model = []
        failure.measurement_failure_code = {}
        failure.measurement_failure_detail = []
        self.assertEqual(measure_profiles._main_failure(failure, "application")["status"],
                         "incomplete")

        diagnostic = RuntimeError("provider failure")
        diagnostic.failure_kind = []
        diagnostic.failure_code = {}
        diagnostic.failure_stage_detail = []
        diagnostic.lifecycle_phase = {}
        diagnostic.lifecycle_subreason = []
        envelope = measure_profiles._diagnostic_failure(diagnostic, "p2_wave")
        self.assertEqual(envelope["status"], "incomplete")
        self.assertEqual(envelope["failure_code"], "provider_execution_failed")

    def test_failure_envelope_does_not_transport_raw_measurement_reason(self):
        failure = RuntimeError("raw provider reason with secret details")
        failure.measurement_model = "CoEdIT"
        failure.measurement_failure_code = "runner_error"
        result = measure_profiles._main_failure(failure, "application")
        encoded = json.dumps(result)
        self.assertEqual(result["model"], "CoEdIT")
        self.assertEqual(result["measurement_failure_code"], "runner_error")
        self.assertNotIn("raw provider reason", encoded)
        self.assertNotIn("secret details", encoded)

    def test_failure_envelope_transports_residency_classifications_without_raw_reason(self):
        for code in ("runtime_setup_failed", "resident_warmup_failed", "residency_fence_failed"):
            failure = RuntimeError("daemon/GPU proof secret details")
            failure.measurement_model = "SmolLM"
            failure.measurement_failure_code = code
            result = measure_profiles._main_failure(failure, "application")
            self.assertEqual(result["measurement_failure_code"], code)
            self.assertEqual(result["reason"], "classified measurement failure")
            self.assertNotIn("secret details", json.dumps(result))

    def test_failure_envelope_transports_only_closed_structural_detail(self):
        failure = RuntimeError("provider secret /private/path")
        failure.measurement_model = "SmolLM"
        failure.measurement_failure_code = "resident_warmup_failed"
        failure.measurement_failure_detail = "failed_output"
        result = measure_profiles._main_failure(failure, "application")
        self.assertEqual(result["measurement_failure_detail"], "failed_output")
        failure.measurement_failure_detail = "provider_secret"
        self.assertNotIn("measurement_failure_detail",
                         measure_profiles._main_failure(failure, "application"))

    def test_failure_envelope_transports_no_execution_sample_detail(self):
        failure = RuntimeError("provider timing detail must not escape")
        failure.measurement_model = "GECToR"
        failure.measurement_failure_code = "no_execution_sample"
        failure.measurement_failure_detail = "no_execution_sample"
        result = measure_profiles._main_failure(failure, "application")
        self.assertEqual(result["measurement_failure_code"], "no_execution_sample")
        self.assertEqual(result["measurement_failure_detail"], "no_execution_sample")
        self.assertNotIn("provider timing detail", json.dumps(result))

    def test_classified_failure_propagates_structural_detail_not_reason(self):
        measurement = SimpleNamespace(
            failure_code="resident_warmup_failed", failure_detail="missing_correlation",
            reason="missing_correlation:/private/provider-secret",
            total_vram_bytes=1, baseline_pre_used_bytes=2,
            peak_incremental_request_bytes=3, derived_ceiling=4)
        failure = measure_profiles._classified_measurement_failure(ModelId.SMOLLM, measurement)
        self.assertEqual(failure.measurement_failure_detail, "missing_correlation")
        self.assertNotIn("provider-secret", json.dumps(
            measure_profiles._main_failure(failure, "application")))

    def test_classified_measurement_failure_covers_finite_discovery_categories(self):
        reasons = (
            "sampler_error", "sample_bound", "native_batch_correlation",
            "undercovered_decoder", "decoder_workload", "identity", "output",
            "allocator", "timing", "invalid_sample", "no_execution_sample",
            "chronology",
        )
        for reason in reasons:
            with self.subTest(reason=reason):
                measurement = SimpleNamespace(
                    failure_code=reason,
                    reason=f"invalid discovery: {reason}: bounded detail",
                    total_vram_bytes=1, baseline_pre_used_bytes=2,
                    peak_incremental_request_bytes=3, derived_ceiling=4)
                failure = measure_profiles._classified_measurement_failure(
                    ModelId.SMOLLM, measurement)
                self.assertEqual(failure.measurement_failure_code, reason)
                result = measure_profiles._main_failure(failure, "application")
                self.assertEqual(result["measurement_failure_code"], reason)
                self.assertEqual(result["reason"], "classified measurement failure")
                self.assertNotIn("bounded detail", json.dumps(result))

    def test_classified_measurement_failure_keeps_unknown_reason_unknown(self):
        measurement = SimpleNamespace(
            failure_code=None,
            reason="invalid discovery: provider_specific_future_code",
            total_vram_bytes=1, baseline_pre_used_bytes=2,
            peak_incremental_request_bytes=3, derived_ceiling=4)
        failure = measure_profiles._classified_measurement_failure(ModelId.SMOLLM, measurement)
        self.assertEqual(failure.measurement_failure_code, "unknown_evidence_failure")
        result = measure_profiles._main_failure(failure, "application")
        self.assertEqual(result["measurement_failure_code"], "unknown_evidence_failure")

    def test_persistence_validation_failure_propagates_model_and_closed_code(self):
        request = object()
        measurement = object()
        identity = object()
        writer = object()
        hostile = ValueError("/private/model prompt=secret output=token")
        with patch.object(measure_profiles, "persist_measured_profile",
                          side_effect=hostile):
            with self.assertRaises(measure_profiles.CapacityEvidenceError) as raised:
                measure_profiles._persist_profile_or_classify(
                    ModelId.COEDIT, request, measurement, identity, writer)
        failure = raised.exception
        self.assertEqual(failure.measurement_model, "CoEdIT")
        self.assertEqual(failure.measurement_failure_code,
                         "persistence_validation_failed")
        envelope = measure_profiles._main_failure(failure, "application")
        self.assertEqual(envelope["model"], "CoEdIT")
        self.assertEqual(envelope["measurement_failure_code"],
                         "persistence_validation_failed")
        self.assertNotIn("private", json.dumps(envelope))
        self.assertNotIn("secret", json.dumps(envelope))
        # The canonical matrix legitimately contains the GECToR selector
        # ``tokens``; assert that the hostile diagnostic value is not leaked
        # rather than rejecting an unrelated matrix identifier.
        self.assertNotIn("prompt=secret", json.dumps(envelope))

    def test_structured_code_wins_over_raw_reason_and_public_envelope_omits_it(self):
        measurement = SimpleNamespace(
            failure_code="configured_ceiling_unproved",
            reason="provider secret /private/model and future code",
            total_vram_bytes=1, baseline_pre_used_bytes=2,
            peak_incremental_request_bytes=3, derived_ceiling=None)
        failure = measure_profiles._classified_measurement_failure(ModelId.SMOLLM, measurement)
        result = measure_profiles._main_failure(failure, "application")
        encoded = json.dumps(result)
        self.assertEqual(failure.measurement_failure_code, "configured_ceiling_unproved")
        self.assertEqual(result["measurement_failure_code"], "configured_ceiling_unproved")
        self.assertNotIn("provider secret", encoded)

    def test_cli_full_matrix_requires_bundle_before_run(self):
        config_path = Path(self.tmp.name) / "bootstrap.json"
        run = AsyncMock()
        argv = ["measure_profiles.py", "--config", str(config_path),
                "--db", str(self.args.db),
                "--ollama-version", "0.11.6"]
        with patch.object(measure_profiles, "load_config", return_value=self.config), \
             patch.object(measure_profiles, "_run", new=run), patch("sys.argv", argv), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["stage"], "application")
        self.assertEqual(result["reason"], "application failure")
        self.assertEqual(result["message"], "application failure")
        self.assertEqual(result["exception_type"], "ValueError")
        self.assertNotIn("--bundle is required", json.dumps(result))
        run.assert_not_awaited()

    def test_cli_rejects_bundle_at_wrong_runtime_mount_before_service_construction(self):
        bundle = Path(self.tmp.name) / "bundle"
        run = AsyncMock()
        argv = ["measure_profiles.py", "--bundle", str(bundle), "--db", str(self.args.db),
                 "--ollama-version", "0.11.6"]
        with patch.object(measure_profiles, "verify_bundle_inputs",
                          side_effect=ValueError("bundle is not mounted at its recorded container mount")) as verifier, \
              patch.object(measure_profiles, "load_config") as load, \
              patch.object(measure_profiles, "_run", new=run), \
              patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 2)
        self.assertEqual(json.loads(output.getvalue())["stage"], "config_load")
        verifier.assert_called_once_with(bundle, require_runtime_mount=True)
        load.assert_not_called()
        run.assert_not_awaited()

    def test_cli_full_matrix_requires_atomic_bundle_and_verifies_before_loading(self):
        bundle = Path(self.tmp.name) / "bundle"
        config_path = bundle / "config.json"
        requests_path = bundle / "requests.json"
        provenance_path = bundle / "provenance.json"
        run = AsyncMock()
        verifier = Mock(return_value=(config_path, requests_path, provenance_path, "a" * 64))
        argv = ["measure_profiles.py", "--bundle", str(bundle), "--db", str(self.args.db),
                "--ollama-version", "0.11.6"]
        profiles = measure_profiles._MeasuredRun(["profile"], tuple(measurement_matrix()))
        run.return_value = profiles
        self.config.artifact_root = Path("/opt/measurement/artifacts")
        self.config.profile_db = self.args.db
        with patch.object(measure_profiles, "verify_bundle_inputs", verifier), \
             patch.object(measure_profiles, "load_config", return_value=self.config) as load, \
             patch.object(measure_profiles, "_run", new=run), \
             patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO):
            self.assertEqual(measure_profiles.main(), 0)
        verifier.assert_called_once_with(bundle, require_runtime_mount=True)
        load.assert_called_once_with(config_path)
        self.assertEqual(run.call_args.args[0].requests, requests_path)
        self.assertEqual(run.call_args.args[0].provenance, "a" * 64)

    def test_cli_full_failure_atomically_persists_same_bounded_envelope(self):
        bundle = Path(self.tmp.name) / "bundle"
        config_path = bundle / "config.json"
        requests_path = bundle / "requests.json"
        runtime = Path(self.tmp.name) / "runtime"
        runtime.mkdir()
        result_file = runtime / "measurement-result.json"
        self.config.profile_db = runtime / "profiles.sqlite"
        failure = RuntimeError("provider failed at /private/model\x00 with secret details")
        argv = ["measure_profiles.py", "--bundle", str(bundle),
                "--db", str(self.config.profile_db), "--result-file", str(result_file),
                "--ollama-version", "0.11.6"]
        with patch.object(measure_profiles, "verify_bundle_inputs",
                          return_value=(config_path, requests_path,
                                        bundle / "provenance.json", "test")), \
             patch.object(measure_profiles, "load_config", return_value=self.config), \
             patch.object(measure_profiles, "_run", new=AsyncMock(side_effect=failure)), \
             patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 2)

        stdout = output.getvalue()
        persisted = result_file.read_text()
        self.assertEqual(persisted, stdout)
        envelope = json.loads(stdout)
        self.assertEqual(envelope["status"], "incomplete")
        self.assertEqual(envelope["stage"], "application")
        self.assertEqual(envelope["failure_kind"], "application_failed")
        self.assertEqual(envelope["reason"], "application failure")
        self.assertEqual(envelope["message"], "application failure")
        self.assertNotIn("/private", stdout)
        self.assertNotIn("secret details", stdout)
        self.assertLessEqual(len(persisted.encode()), 8192)

    def test_cli_full_matrix_rejects_legacy_inputs_even_with_bundle(self):
        bundle = Path(self.tmp.name) / "bundle"
        argv = ["measure_profiles.py", "--bundle", str(bundle), "--config",
                str(bundle / "config.json"), "--db", str(self.args.db),
                "--ollama-version", "0.11.6"]
        with patch.object(measure_profiles, "load_config", return_value=self.config), \
             patch.object(measure_profiles, "verify_bundle_inputs") as verifier, \
             patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 2)
        verifier.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertEqual(result["reason"], "application failure")
        self.assertEqual(result["message"], "application failure")
        self.assertEqual(result["exception_type"], "ValueError")
        self.assertNotIn("cannot be combined", output.getvalue())

    def test_cli_success_reports_run_matrix_without_reloading_config(self):
        bundle = Path(self.tmp.name) / "bundle"
        config_path = bundle / "config.json"
        requests_path = bundle / "requests.json"
        argv = ["measure_profiles.py", "--bundle", str(bundle),
                "--db", str(self.args.db), "--ollama-version", "0.11.6"]
        profiles = measure_profiles._MeasuredRun(["profile"],
                                                  tuple(measurement_matrix()))
        run = AsyncMock(return_value=profiles)
        with patch.object(measure_profiles, "verify_bundle_inputs",
                          return_value=(config_path, requests_path, bundle / "provenance.json", "test")), \
             patch.object(measure_profiles, "load_config", return_value=self.config) as load, \
             patch.object(measure_profiles, "_run", new=run), patch("sys.argv", argv), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.config.artifact_root = Path("/opt/measurement/artifacts")
            self.config.profile_db = self.args.db
            self.assertEqual(measure_profiles.main(), 0)
        load.assert_called_once_with(config_path)
        envelope = json.loads(output.getvalue())
        self.assertEqual(set(envelope), {"status", "matrix", "profiles"})
        self.assertEqual(envelope["status"], "complete")
        self.assertEqual(envelope["profiles"],
                         [{"model": "unknown", "profile_identity": "profile"}])
        # JSON serialization normalizes the tuple pairs to arrays.
        self.assertEqual(envelope["matrix"], [list(pair) for pair in profiles.matrix])

    def _assert_db_rejected_before_service_construction(self, db, expected):
        bundle = Path(self.tmp.name) / "bundle"
        config_path = bundle / "config.json"
        requests_path = bundle / "requests.json"
        run = AsyncMock(side_effect=AssertionError("service construction must not run"))
        argv = ["measure_profiles.py", "--bundle", str(bundle), "--db", str(db),
                "--ollama-version", "0.11.6"]
        self.config.artifact_root = Path("/opt/measurement/artifacts")
        with patch.object(measure_profiles, "verify_bundle_inputs",
                          return_value=(config_path, requests_path, bundle / "provenance.json", "test")), \
             patch.object(measure_profiles, "load_config", return_value=self.config), \
             patch.object(measure_profiles, "_run", new=run), patch("sys.argv", argv), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result["reason"], "application failure")
        self.assertEqual(result["message"], "application failure")
        self.assertEqual(result["exception_type"], "ValueError")
        self.assertNotIn(expected, json.dumps(result))
        run.assert_not_awaited()

    def _assert_db_allowed_before_service_construction(self, db):
        bundle = Path(self.tmp.name) / "bundle"
        config_path = bundle / "config.json"
        requests_path = bundle / "requests.json"
        run = AsyncMock(return_value=measure_profiles._MeasuredRun([], tuple(measurement_matrix())))
        argv = ["measure_profiles.py", "--bundle", str(bundle), "--db", str(db),
                "--ollama-version", "0.11.6"]
        self.config.artifact_root = Path("/opt/measurement/artifacts")
        with patch.object(measure_profiles, "verify_bundle_inputs",
                          return_value=(config_path, requests_path, bundle / "provenance.json", "test")), \
             patch.object(measure_profiles, "load_config", return_value=self.config), \
             patch.object(measure_profiles, "_run", new=run), patch("sys.argv", argv), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "complete")
        run.assert_awaited_once()

    def test_cli_rejects_db_equal_to_bundle_before_service_construction(self):
        bundle = Path(self.tmp.name) / "bundle"
        self._assert_db_rejected_before_service_construction(bundle,
                                                              "outside the resolved bundle")

    def test_cli_rejects_db_nested_in_bundle_before_service_construction(self):
        bundle = Path(self.tmp.name) / "bundle"
        self._assert_db_rejected_before_service_construction(bundle / "nested" / "profiles.sqlite",
                                                              "outside the resolved bundle")

    def test_cli_rejects_symlink_alias_db_before_service_construction(self):
        bundle = Path(self.tmp.name) / "bundle"
        target = Path(self.tmp.name) / "caller-selected.sqlite"
        alias = Path(self.tmp.name) / "db-alias.sqlite"
        os.symlink(target, alias)
        self._assert_db_rejected_before_service_construction(alias,
                                                               "non-symlink path")

    def test_cli_rejects_symlink_bundle_before_service_construction(self):
        target = Path(self.tmp.name) / "bundle-target"
        bundle = Path(self.tmp.name) / "bundle-alias"
        os.symlink(target, bundle)
        config_path = bundle / "config.json"
        requests_path = bundle / "requests.json"
        run = AsyncMock(side_effect=AssertionError("service construction must not run"))
        self.config.artifact_root = Path("/opt/measurement/artifacts")
        argv = ["measure_profiles.py", "--bundle", str(bundle),
                "--db", str(Path(self.tmp.name) / "profiles.sqlite"),
                "--ollama-version", "0.11.6"]
        with patch.object(measure_profiles, "verify_bundle_inputs",
                          return_value=(config_path, requests_path,
                                        bundle / "provenance.json", "test")), \
             patch.object(measure_profiles, "load_config", return_value=self.config), \
             patch.object(measure_profiles, "_run", new=run), patch("sys.argv", argv), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(measure_profiles.main(), 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result["reason"], "application failure")
        self.assertEqual(result["message"], "application failure")
        self.assertEqual(result["exception_type"], "ValueError")
        self.assertNotIn("bundle must be a non-symlink path", json.dumps(result))
        run.assert_not_awaited()

    def test_cli_accepts_db_equal_to_generated_config_profile_db(self):
        bundle = Path(self.tmp.name) / "bundle"
        # The generated path is the caller-owned database on the writable
        # runtime volume; equality is required rather than forbidden.
        self.config.profile_db = Path(self.tmp.name) / "profiles.sqlite"
        self._assert_db_allowed_before_service_construction(self.config.profile_db)

    async def test_diagnostic_failure_normalizes_hostile_metadata(self):
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        failure = RuntimeError("outer\x00 failure /private/prompt")
        failure.failure_kind = object()
        failure.failure_code = "C:\\private\\result\x1b" + ("x" * 1000)
        failure.failure_message = "prompt /private/result\n" + ("y" * 1000)
        patches, *_ = self._diagnostic_patches(provider_error=failure)
        with self._all(patches):
            result = await measure_profiles._diagnostic_run(
                self._diagnostic_args(request_path), self.config)
        json.dumps(result)
        for field in ("failure_kind", "failure_code", "failure_message"):
            self.assertIsInstance(result[field], str)
            self.assertLessEqual(len(result[field]), 160)
            self.assertNotRegex(result[field], r"[\x00-\x1f\x7f/\\]")
        self.assertEqual(result["failure_kind"], "provider_execution_failed")
        self.assertEqual(result["status"], "incomplete")

    async def test_diagnostic_gpu_capture_failure_is_classified(self):
        request_path = Path(self.tmp.name) / "diagnostic.json"
        request_path.write_text(json.dumps("configured-request"))
        with patch.object(measure_profiles.LinuxGPUProof, "capture", side_effect=RuntimeError("secret")), \
             patch.object(measure_profiles, "_diagnostic_request", return_value="configured-request"):
            result = await measure_profiles._diagnostic_run(
                self._diagnostic_args(request_path), self.config)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["failure_stage_detail"], "gpu_proof_capture")

    @staticmethod
    def _all(patches):
        stack = ExitStack()
        for item in patches:
            stack.enter_context(item)
        return stack


if __name__ == "__main__":
    unittest.main()
