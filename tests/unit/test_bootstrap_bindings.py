"""Bootstrap boundary tests (intentionally not run in the offline slice)."""
import json
import hashlib
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from services.llm.bootstrap.bindings import _BUCKETS, observe_runtime_identities, prepare_bindings
from services.llm.bootstrap.measurement_bindings import prepare_measurement_bindings
from services.llm.bootstrap.config import BootstrapConfig, ModelConfig, load_config
from services.llm.providers.config import GPUProof
from services.llm.providers.gpu import ProcessIdentity
from services.llm.queue.contracts import ModelId
from services.llm.provisioning.artifacts import SPECS
from services.llm.provisioning.volume import provision
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.profiles import BenchmarkMetadata, ProfileStore


def _sample(concurrency, wave, wall=10):
    return SampleMetadata(concurrency, wave, concurrency, wall, 100, (2,) * concurrency)


def _profile(model, manifest_hash, model_hash, runtime, adapter, *, context=None, bucket=None,
             fingerprint="fixture", draft=False, optimum=1):
    baseline = tuple(_sample(1, wave) for wave in range(4))
    warmup = (_sample(1, 0), _sample(2, 0))
    # N=2 is retained as diagnostic evidence, but its 20ms waves do not make it
    # the p=1 admission profile (p=1 is 10ms).
    measured = tuple(_sample(n, wave, 10 if n == 1 or optimum == 2 else 20)
                     for n in (1, 2) for wave in range(1, 5))
    identity = {"model_id": model.value, "gpu_uuid": "GPU-test",
                "artifact_manifest_hash": manifest_hash, "model_hash": model_hash,
                "runtime_identity": runtime, "adapter_identity": adapter,
                "context_size": context, "bucket_identity": bucket, "fingerprint": fingerprint}
    profile_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    profile = CapacityProfile(model, "GPU-test", manifest_hash, model_hash, runtime, adapter,
                               profile_id, optimum, 2, optimum, 20, baseline + warmup + measured,
                              context, bucket)
    metadata = BenchmarkMetadata(fingerprint, "2026-09-16T00:00:00Z", "bootstrap-fixture",
                                 baseline, warmup, measured,
                                 bucket if bucket else f"context:{context}")
    return profile, metadata


def _provisioned_config(directory):
    root = Path(directory)
    sources = {}
    for model, files in SPECS.items():
        source = root / "sources" / model
        source.mkdir(parents=True)
        for name in files:
            (source / name).write_bytes(f"{model}/{name}".encode())
        sources[model] = source
    artifacts = root / "artifacts"
    document = provision(sources, artifacts)
    runtime = {name: f"runtime-{name}" for name in SPECS}
    config = BootstrapConfig("GPU-test", artifacts, document["manifest_sha256"], root / "profiles.sqlite",
                             root / "ollama", root / "ollama-home", 11434,
                             {name: ModelConfig(runtime[name], f"adapter-{name}") for name in SPECS})
    hashes = {model: next(item["sha256"] for item in document["models"][model]["files"]
                          if item["path"] == ("SmolLM2-1.7B-Instruct-Q8_0.gguf" if model == "SmolLM" else "model.safetensors"))
              for model in SPECS}
    return config, document, hashes, runtime


class BootstrapConfigTests(unittest.TestCase):
    def _document(self):
        identity = {"runtime_identity": "approved-runtime", "adapter_identity": "approved-adapter"}
        return {
            "schema": 1, "gpu_uuid": "GPU-test", "artifact_root": "/srv/artifacts",
            "manifest_sha256": "0" * 64, "profile_db": "/srv/profiles.sqlite",
            "ollama_binary": "/usr/local/bin/ollama", "ollama_home": "/srv/ollama",
            "ollama_port": 11434, "models": {name: dict(identity) for name in ("SmolLM", "CoEdIT", "GECToR")},
        }

    def test_exact_document_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bootstrap.json"
            path.write_text(json.dumps(self._document()))
            self.assertEqual(load_config(path).gpu_uuid, "GPU-test")

    def test_unknown_and_duplicate_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bootstrap.json"
            document = self._document()
            document["unexpected"] = True
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_config(path)
            path.write_text('{"schema":1,"schema":1}')
            with self.assertRaises(ValueError):
                load_config(path)

    def test_relative_paths_and_boolean_port_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bootstrap.json"
            document = self._document()
            document["profile_db"] = "profiles.sqlite"
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_oversized_nofollow_and_control_inputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bootstrap.json"
            path.write_bytes(b"{" + b" " * (32 * 1024) + b"}")
            with self.assertRaises(ValueError):
                load_config(path)
            document = self._document()
            document["gpu_uuid"] = "GPU-test\x00"
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_models_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bootstrap.json"
            path.write_text(json.dumps(self._document()))
            config = load_config(path)
            with self.assertRaises(TypeError):
                config.models["SmolLM"] = config.models["CoEdIT"]

    def test_runtime_observation_is_metadata_only_and_complete(self):
        with patch("services.llm.bootstrap.bindings.importlib.metadata.version",
                   side_effect=lambda name: {"torch": "2", "transformers": "4", "tokenizers": "0", "safetensors": "1", "gector": "3"}[name]):
            observed = observe_runtime_identities("0.1")
        self.assertEqual(set(observed), {"SmolLM", "CoEdIT", "GECToR"})
        self.assertIn("safetensors=1", observed["GECToR"])
        self.assertIn("gector=3", observed["GECToR"])

    def test_runtime_observation_rejects_bounded_ascii_and_delimiter_inputs(self):
        valid = {"torch": "2", "transformers": "4", "tokenizers": "0", "safetensors": "1", "gector": "3"}
        for bad in ("", "x" * 65, "has whitespace", "has\tcontrol", "has\ncontrol", "has|delimiter", "has=delimiter", "café"):
            with self.subTest(bad=bad):
                with patch("services.llm.bootstrap.bindings.importlib.metadata.version",
                           side_effect=lambda name, bad=bad: bad if name == "torch" else valid[name]):
                    with self.assertRaises(ValueError):
                        observe_runtime_identities("0.1")
        for bad in ("", "x" * 65, "has whitespace", "has\tcontrol", "has|delimiter", "has=delimiter", "café"):
            with self.subTest(ollama=bad):
                with patch("services.llm.bootstrap.bindings.importlib.metadata.version", side_effect=valid.__getitem__):
                    with self.assertRaises(ValueError):
                        observe_runtime_identities(bad)

    def test_gector_metadata_version_is_scoped_to_gector_identity(self):
        versions = {"torch": "2", "transformers": "4", "tokenizers": "0", "safetensors": "1", "gector": "3"}
        with patch("services.llm.bootstrap.bindings.importlib.metadata.version", side_effect=versions.__getitem__):
            before = observe_runtime_identities("0.1")
        versions["gector"] = "4"
        with patch("services.llm.bootstrap.bindings.importlib.metadata.version", side_effect=versions.__getitem__):
            after = observe_runtime_identities("0.1")
        self.assertEqual(after["SmolLM"], before["SmolLM"])
        self.assertEqual(after["CoEdIT"], before["CoEdIT"])
        self.assertNotEqual(after["GECToR"], before["GECToR"])

    def test_prepare_constructs_three_unloaded_real_adapters_and_pins_resolution(self):
        async def identity():
            return "GPU-test"
        proof = GPUProof(identity, lambda: True, expected_supervisor=ProcessIdentity(1, 2))
        with tempfile.TemporaryDirectory() as directory:
            config, document, hashes, runtime = _provisioned_config(directory)
            with ProfileStore(config.profile_db) as registry:
                for model in ModelId:
                    name = model.value
                    profile, metadata = _profile(model, config.manifest_sha256, hashes[name], runtime[name],
                                                 f"adapter-{name}", context=512 if model is ModelId.SMOLLM else None,
                                                 bucket=None if model is ModelId.SMOLLM else _BUCKETS[model])
                    registry.save_measured(profile, metadata)
            prepared = __import__("asyncio").run(prepare_bindings(config, proof, runtime))
            try:
                self.assertEqual({type(binding.provider).__name__ for binding in prepared.bindings.values()},
                                 {"SmolLMProvider", "CoEdITProvider", "GECToRProvider"})
                with self.assertRaises(ValueError):
                    prepared.bindings[ModelId.SMOLLM].resolve(context_size=513, bucket_identity=None)
                with self.assertRaises(ValueError):
                    prepared.bindings[ModelId.COEDIT].resolve(context_size=None, bucket_identity="other")
                self.assertEqual(prepared.bindings[ModelId.SMOLLM].resolve(context_size=512, bucket_identity=None)[0].optimal_parallelism, 1)
                self.assertEqual(prepared.bindings[ModelId.COEDIT].resolve(context_size=None, bucket_identity=_BUCKETS[ModelId.COEDIT])[0].memory_safe_n, 2)
            finally:
                prepared.close()
            document = self._document()
            document["ollama_port"] = True
            path = Path(directory) / "bootstrap.json"
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_measured_parallelism_configures_runtime_adapters_and_preserves_selectors(self):
        async def identity(): return "GPU-test"
        proof = GPUProof(identity, lambda: True, expected_supervisor=ProcessIdentity(1, 2))
        with tempfile.TemporaryDirectory() as directory:
            config, _, hashes, runtime = _provisioned_config(directory)
            capacities = {ModelId.SMOLLM: 2, ModelId.COEDIT: 2, ModelId.GECTOR: 1}
            with ProfileStore(config.profile_db) as registry:
                for model in ModelId:
                    name = model.value
                    profile, metadata = _profile(
                        model, config.manifest_sha256, hashes[name], runtime[name], f"adapter-{name}",
                        context=512 if model is ModelId.SMOLLM else None,
                        bucket=None if model is ModelId.SMOLLM else _BUCKETS[model],
                        optimum=capacities[model])
                    registry.save_measured(profile, metadata)
            prepared = __import__("asyncio").run(prepare_bindings(config, proof, runtime))
            try:
                smollm = prepared.bindings[ModelId.SMOLLM].provider.config
                coedit = prepared.bindings[ModelId.COEDIT].provider.config
                gector = prepared.bindings[ModelId.GECTOR].provider.config
                self.assertEqual(smollm.parallelism, 2)
                self.assertEqual(coedit.max_native_batch_size, 2)
                self.assertEqual(coedit.bucket_batch_size, 1)
                self.assertEqual(coedit.bucket_identity, _BUCKETS[ModelId.COEDIT])
                self.assertIsNone(coedit.measurement_max_native_batch_size)
                self.assertEqual(gector.__class__.__name__, "GECToRProviderConfig")
                self.assertEqual(prepared.bindings[ModelId.COEDIT].resolve(
                    context_size=None, bucket_identity=_BUCKETS[ModelId.COEDIT])[0].admission_limit, 4)
                self.assertEqual(prepared.bindings[ModelId.SMOLLM].resolve(
                    context_size=512, bucket_identity=None)[0].admission_limit, 4)
            finally:
                prepared.close()

    def test_prepare_rejects_capacity_above_gector_capability(self):
        async def identity(): return "GPU-test"
        proof = GPUProof(identity, lambda: True, expected_supervisor=ProcessIdentity(1, 2))
        with tempfile.TemporaryDirectory() as directory:
            config, _, hashes, runtime = _provisioned_config(directory)
            with ProfileStore(config.profile_db) as registry:
                for model in ModelId:
                    name = model.value
                    profile, metadata = _profile(
                        model, config.manifest_sha256, hashes[name], runtime[name], f"adapter-{name}",
                        context=512 if model is ModelId.SMOLLM else None,
                        bucket=None if model is ModelId.SMOLLM else _BUCKETS[model],
                        optimum=2 if model is ModelId.GECTOR else 1)
                    registry.save_measured(profile, metadata)
            with self.assertRaisesRegex(ValueError, "supported measured capacity"):
                __import__("asyncio").run(prepare_bindings(config, proof, runtime))

    def test_prepare_rejects_profile_without_twenty_percent_reserve(self):
        async def identity(): return "GPU-test"
        proof = GPUProof(identity, lambda: True, expected_supervisor=ProcessIdentity(1, 2))
        with tempfile.TemporaryDirectory() as directory:
            config, _, hashes, runtime = _provisioned_config(directory)
            with ProfileStore(config.profile_db) as registry:
                for model in ModelId:
                    name = model.value
                    profile, metadata = _profile(
                        model, config.manifest_sha256, hashes[name], runtime[name], f"adapter-{name}",
                        context=512 if model is ModelId.SMOLLM else None,
                        bucket=None if model is ModelId.SMOLLM else _BUCKETS[model])
                    registry.save_measured(profile, metadata)
            lookup = ProfileStore.lookup
            def changed_reserve(store, model, *args, **kwargs):
                profile = lookup(store, model, *args, **kwargs)
                if model is ModelId.COEDIT:
                    return replace(profile, safety_reserve_percent=10)
                return profile
            with patch.object(ProfileStore, "lookup", changed_reserve):
                with self.assertRaisesRegex(ValueError, "supported measured capacity"):
                    __import__("asyncio").run(prepare_bindings(config, proof, runtime))

    def test_pinned_binding_rejects_changed_capacity_after_preflight(self):
        async def identity(): return "GPU-test"
        proof = GPUProof(identity, lambda: True, expected_supervisor=ProcessIdentity(1, 2))
        with tempfile.TemporaryDirectory() as directory:
            config, _, hashes, runtime = _provisioned_config(directory)
            with ProfileStore(config.profile_db) as registry:
                for model in ModelId:
                    name = model.value
                    profile, metadata = _profile(
                        model, config.manifest_sha256, hashes[name], runtime[name], f"adapter-{name}",
                        context=512 if model is ModelId.SMOLLM else None,
                        bucket=None if model is ModelId.SMOLLM else _BUCKETS[model],
                        optimum=2 if model is ModelId.SMOLLM else 1)
                    registry.save_measured(profile, metadata)
            prepared = __import__("asyncio").run(prepare_bindings(config, proof, runtime))
            try:
                binding = prepared.bindings[ModelId.SMOLLM]
                changed = replace(prepared.profiles[ModelId.SMOLLM], optimal_parallelism=1,
                                  buffer_capacity=1)
                with patch.object(prepared._store, "lookup", return_value=changed):
                    with self.assertRaisesRegex(ValueError, "pinned measured shape"):
                        binding.resolve(context_size=512, bucket_identity=None)
            finally:
                prepared.close()

    def test_larger_context_cannot_substitute_for_pinned_512_profile(self):
        async def identity(): return "GPU-test"
        proof = GPUProof(identity, lambda: True, expected_supervisor=ProcessIdentity(1, 2))
        with tempfile.TemporaryDirectory() as directory:
            config, _, hashes, runtime = _provisioned_config(directory)
            with ProfileStore(config.profile_db) as registry:
                for model in ModelId:
                    name = model.value
                    context = 1024 if model is ModelId.SMOLLM else None
                    bucket = None if model is ModelId.SMOLLM else _BUCKETS[model]
                    profile, metadata = _profile(model, config.manifest_sha256, hashes[name], runtime[name], f"adapter-{name}", context=context, bucket=bucket)
                    registry.save_measured(profile, metadata)
            with self.assertRaises(ValueError):
                __import__("asyncio").run(prepare_bindings(config, proof, runtime))

    def test_resolve_rejects_later_profile_that_expands_pinned_capacity(self):
        async def identity(): return "GPU-test"
        proof = GPUProof(identity, lambda: True, expected_supervisor=ProcessIdentity(1, 2))
        with tempfile.TemporaryDirectory() as directory:
            config, _, hashes, runtime = _provisioned_config(directory)
            with ProfileStore(config.profile_db) as registry:
                for model in ModelId:
                    name = model.value
                    profile, metadata = _profile(
                        model, config.manifest_sha256, hashes[name], runtime[name],
                        f"adapter-{name}", context=512 if model is ModelId.SMOLLM else None,
                        bucket=None if model is ModelId.SMOLLM else _BUCKETS[model])
                    registry.save_measured(profile, metadata)
            prepared = __import__("asyncio").run(prepare_bindings(config, proof, runtime))
            try:
                binding = prepared.bindings[ModelId.SMOLLM]
                expanded = replace(prepared.profiles[ModelId.SMOLLM], optimal_parallelism=2, buffer_capacity=2)
                with patch.object(prepared._store, "lookup", return_value=expanded):
                    with self.assertRaises(ValueError):
                        binding.resolve(context_size=512, bucket_identity=None)
            finally:
                prepared.close()

    def test_prepare_rejects_missing_draft_and_mismatched_identity_evidence(self):
        async def identity(): return "GPU-test"
        proof = GPUProof(identity, lambda: True, expected_supervisor=ProcessIdentity(1, 2))
        for failure in ("missing", "draft", "runtime", "manifest", "gpu", "model", "bucket"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                config, _, hashes, runtime = _provisioned_config(directory)
                with ProfileStore(config.profile_db) as registry:
                    for model in ModelId:
                        if failure == "missing" and model is ModelId.GECTOR:
                            continue
                        name = model.value
                        artifact = "f" * 64 if failure == "manifest" and model is ModelId.GECTOR else config.manifest_sha256
                        model_hash = "e" * 64 if failure == "model" and model is ModelId.GECTOR else hashes[name]
                        profile_runtime = "other-runtime" if failure == "runtime" and model is ModelId.GECTOR else runtime[name]
                        gpu = "GPU-other" if failure == "gpu" and model is ModelId.GECTOR else "GPU-test"
                        bucket = "other-bucket" if failure == "bucket" and model is ModelId.GECTOR else (None if model is ModelId.SMOLLM else _BUCKETS[model])
                        profile, metadata = _profile(model, artifact, model_hash, profile_runtime, f"adapter-{name}",
                                                     context=512 if model is ModelId.SMOLLM else None, bucket=bucket)
                        if gpu != "GPU-test":
                            profile = CapacityProfile(profile.model_id, gpu, profile.artifact_manifest_hash, profile.model_hash,
                                                      profile.runtime_identity, profile.adapter_identity, profile.profile_identity,
                                                      profile.optimal_parallelism, profile.memory_safe_n, profile.buffer_capacity,
                                                      profile.safety_reserve_percent, profile.raw_samples, profile.context_size,
                                                      profile.bucket_identity)
                            # The altered row is deliberately invalid identity evidence; it need
                            # not be persisted because the absent exact GPU row is the assertion.
                            continue
                        if failure == "draft" and model is ModelId.GECTOR:
                            registry.save_draft(profile, metadata)
                        else:
                            registry.save_measured(profile, metadata)
                with self.assertRaises(ValueError):
                    __import__("asyncio").run(prepare_bindings(config, proof, runtime))

    def test_missing_profile_database_is_not_created(self):
        async def identity(): return "GPU-test"
        proof = GPUProof(identity, lambda: True, expected_supervisor=ProcessIdentity(1, 2))
        with tempfile.TemporaryDirectory() as directory:
            config, _, _, runtime = _provisioned_config(directory)
            self.assertFalse(config.profile_db.exists())
            with self.assertRaises(Exception):
                __import__("asyncio").run(prepare_bindings(config, proof, runtime))
            self.assertFalse(config.profile_db.exists())

    def test_direct_configuration_construction_is_strict_and_frozen(self):
        with self.assertRaises(ValueError):
            BootstrapConfig("bad", Path("/artifacts"), "0" * 64, Path("/profiles"), Path("/ollama"), Path("/home"), True, {})
        model = ModelConfig("runtime", "adapter")
        with self.assertRaises(Exception):
            model.runtime_identity = "changed"


class MeasurementBindingsProofCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_supervisor_identity_and_ownership_for_raw_or_typed_proof(self):
        supervisor = ProcessIdentity(17, 23)
        ownership = lambda: object()

        async def identity():
            return "GPU-test"

        raw = SimpleNamespace(identity=identity, cleanup=lambda: True,
                              residency=lambda: None, residency_for_runner=None,
                              memory=lambda: None, supervisor_identity=supervisor,
                              ollama_ownership=ownership)
        typed = GPUProof(identity, raw.cleanup, raw.residency,
                         expected_supervisor=supervisor,
                         ollama_ownership=ownership)
        config = type("Config", (), {
            "gpu_uuid": "GPU-test",
            "manifest_sha256": "a" * 64,
            "artifact_root": Path("/artifacts"),
            "models": {model.value: type("Model", (), {
                "runtime_identity": "runtime",
                "adapter_identity": "adapter",
            })() for model in ModelId},
            "ollama_binary": "/usr/bin/ollama",
            "ollama_home": Path("/ollama-home"),
            "ollama_port": 11434,
        })()
        hashes = {model.value: "b" * 64 for model in ModelId}

        for proof in (raw, typed):
            with self.subTest(proof=type(proof).__name__), \
                    patch("services.llm.bootstrap.measurement_bindings._verify_and_hashes",
                          return_value=hashes):
                prepared = await prepare_measurement_bindings(
                    config, proof, {model.value: "runtime" for model in ModelId}, ceiling=1)
                for binding in prepared.bindings.values():
                    typed_proof = binding.provider.config.gpu_proof
                    self.assertIs(typed_proof.ollama_ownership, ownership)
                    self.assertEqual(typed_proof.expected_supervisor, supervisor)
