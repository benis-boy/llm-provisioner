import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services.llm.provisioning.artifacts import SPECS, inventory, manifest, spec
from tools.compatibility import prepare_measurement as subject


def fixture_context(root: Path, *, source_identity_differs=False) -> Path:
    context = root / "context"
    models = context / "models"
    entries = {}
    for model in subject.MODELS:
        model_root = models / model
        model_root.mkdir(parents=True)
        for name in SPECS[model]:
            (model_root / name).write_bytes((model + ":" + name).encode("ascii"))
        entry = inventory(spec(model, model_root))
        entries[model] = {**entry, "root": (f"source/{model}" if source_identity_differs else f"models/{model}")}
    context.mkdir(exist_ok=True)
    document = manifest(entries)
    # Artifact manifests use the provisioning module's canonical JSON form,
    # which has no trailing newline (bundle documents do include one).
    (context / "manifest.json").write_bytes(json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii"))
    return context


def fixture_retained_context(root: Path) -> Path:
    context = fixture_context(root)
    for name in ("requirements-candidate.txt", "Dockerfile", "spike.py", "rm_spike.py",
                 "model_runtime.py", "input_bounds.py", "artifacts.py",
                 "requirements.lock", "ollama-linux-amd64.tgz", "provenance.json",
                 ".compatibility-spike-owned"):
        (context / name).write_bytes(b"compatibility build input")
    wheelhouse = context / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "candidate.whl").write_bytes(b"wheel")
    for relative in subject.PREPARE_HARNESS_FILES:
        path = context / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# harness")
    return context


class PrepareMeasurementTests(unittest.TestCase):
    def prepare(self, root: Path, *, mount="/opt/measurement", replace=False):
        return subject._prepare(root / "bundle", root / "context", "GPU-test", "0.11.6",
                                mount, {m: "runtime:" + m for m in subject.MODELS},
                                {m: "adapter:" + m for m in subject.MODELS}, replace)

    def test_two_runs_are_byte_identical_and_identity_is_fixture_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_context(root)
            first = self.prepare(root)
            first_bytes = {path.name: path.read_bytes() for path in (root / "bundle").iterdir()
                           if path.is_file()}
            identity = subject.verify(root / "bundle")
            (root / "bundle").rename(root / "first")
            second = self.prepare(root)
            second_bytes = {path.name: path.read_bytes() for path in (root / "bundle").iterdir()
                            if path.is_file()}
            self.assertEqual(first, second)
            self.assertEqual(identity, second)
            self.assertEqual(first_bytes, second_bytes)
            provenance = json.loads((root / "bundle" / "provenance.json").read_bytes())
            self.assertEqual(tuple(provenance["matrix"]), subject.MATRIX_IDS)
            self.assertEqual(set(provenance["request_fingerprints"]), set(subject.MODELS))
            self.assertEqual(set(provenance["model_sha256"]), set(subject.MODELS))
            self.assertEqual(provenance["container_mount"], "/opt/measurement")
            selected_manifest = json.loads((root / "bundle" / "artifacts" / "current" /
                                            "manifest.json").read_bytes())
            self.assertEqual(provenance["model_sha256"], {
                model: next(item["sha256"] for item in selected_manifest["models"][model]["files"]
                            if item["path"] == SPECS[model][0])
                for model in subject.MODELS})

    def test_runtime_manifest_is_authoritative_when_context_identity_differs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_context(root, source_identity_differs=True)
            self.prepare(root)
            provenance = json.loads((root / "bundle" / "provenance.json").read_bytes())
            selected = json.loads((root / "bundle" / "artifacts" / "current" /
                                   "manifest.json").read_bytes())
            self.assertEqual(provenance["schema"], 2)
            self.assertNotEqual(provenance["source_context_manifest_sha256"],
                                selected["manifest_sha256"])
            self.assertEqual(provenance["manifest_sha256"], selected["manifest_sha256"])
            self.assertEqual(provenance["model_sha256"], provenance["source_model_sha256"])
            self.assertEqual(subject.verify(root / "bundle"),
                             (root / "bundle" / "provenance.sha256").read_text().strip())

            provenance["source_model_sha256"]["SmolLM"] = "0" * 64
            provenance_bytes = subject.canonical(provenance)
            (root / "bundle" / "provenance.json").write_bytes(provenance_bytes)
            (root / "bundle" / "provenance.sha256").write_text(
                subject.digest(provenance_bytes) + "\n")
            with self.assertRaisesRegex(ValueError, "source/runtime model provenance mismatch"):
                subject.verify(root / "bundle")

    def test_verifier_rejects_tampered_inputs_and_extra_entries(self):
        mutations = ("config.json", "requests.json", "provenance.json")
        for name in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); fixture_context(root); self.prepare(root)
                path = root / "bundle" / name
                value = path.read_bytes()
                path.write_bytes(value[:-1] + (b" " if value[-1:] != b" " else b"!"))
                with self.assertRaises(ValueError): subject.verify(root / "bundle")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture_context(root); self.prepare(root)
            (root / "bundle" / "extra").write_bytes(b"x")
            with self.assertRaises(ValueError): subject.verify(root / "bundle")

    def test_verifier_rejects_model_manifest_and_current_tampering(self):
        for kind in ("model", "manifest", "current"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); fixture_context(root); self.prepare(root)
                artifacts = root / "bundle" / "artifacts"
                if kind == "model":
                    selected = artifacts / artifacts.joinpath("current").readlink() / "models" / "SmolLM" / SPECS["SmolLM"][0]
                    selected.write_bytes(b"tampered")
                elif kind == "manifest":
                    selected = artifacts / artifacts.joinpath("current").readlink() / "manifest.json"
                    selected.write_bytes(b"{}")
                else:
                    (artifacts / "current").unlink()
                    (artifacts / "current").symlink_to("../outside")
                with self.assertRaises(ValueError): subject.verify(root / "bundle")

    def test_paths_are_absolute_and_replacement_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture_context(root); self.prepare(root)
            config = json.loads((root / "bundle" / "config.json").read_bytes())
            self.assertTrue(all(Path(config[name]).is_absolute() for name in
                                ("artifact_root", "profile_db", "ollama_binary", "ollama_home")))
            self.assertEqual(config["artifact_root"], "/opt/measurement/artifacts")
            self.assertEqual(config["profile_db"], str(subject.RUNTIME_PROFILE_DB))
            self.assertEqual(config["ollama_home"], str(subject.RUNTIME_OLLAMA_HOME))
            self.assertEqual(config["ollama_binary"], str(subject.CANONICAL_OLLAMA_BINARY))
            # Recreating the same bundle is an idempotent, verified no-op.
            self.assertEqual(self.prepare(root), subject.verify(root / "bundle"))
            with self.assertRaises(ValueError):
                self.prepare(root, mount="/opt/other", replace=False)
            self.prepare(root, mount="/opt/other", replace=True)

    def test_host_verification_accepts_arbitrary_output_but_runtime_requires_recorded_mount(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_context(root)
            self.prepare(root)
            # Preparation and transfer verification bind the host artifact tree,
            # not the container's eventual mount spelling.
            self.assertEqual(subject.verify(root / "bundle"),
                             subject.verify_bundle_inputs(root / "bundle")[3])
            with self.assertRaisesRegex(ValueError, "not mounted at its recorded"):
                subject.verify_bundle_inputs(root / "bundle", require_runtime_mount=True)

            # A bundle at the recorded path is accepted by the runtime check.
            runtime_root = root / "runtime-bundle"
            fixture_context(runtime_root)
            subject._prepare(runtime_root / "bundle", runtime_root / "context", "GPU-test", "0.11.6",
                             str((runtime_root / "bundle").absolute()),
                             {m: "runtime:" + m for m in subject.MODELS},
                             {m: "adapter:" + m for m in subject.MODELS}, False)
            subject.verify_bundle_inputs(runtime_root / "bundle", require_runtime_mount=True)

    def test_verifier_rejects_bundle_owned_runtime_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture_context(root); self.prepare(root)
            path = root / "bundle" / "config.json"
            config = json.loads(path.read_bytes())
            config["ollama_home"] = "/opt/measurement/ollama"
            path.write_bytes(subject.canonical(config))
            with self.assertRaisesRegex(ValueError, "runtime mount contract"):
                subject.verify(root / "bundle")

    def test_failed_replace_restores_the_previous_valid_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture_context(root); original = self.prepare(root)
            real_replace = subject.os.replace
            failed = False
            def fail_new(source, destination):
                # Fail only when selecting the staged replacement.  Rollback
                # moves the sibling backup back into place and must succeed.
                nonlocal failed
                if Path(destination) == root / "bundle" and not failed:
                    failed = True
                    raise OSError("injected replace failure")
                return real_replace(source, destination)
            with patch.object(subject.os, "replace", side_effect=fail_new):
                with self.assertRaises(OSError):
                    self.prepare(root, mount="/opt/changed", replace=True)
            self.assertEqual(subject.verify(root / "bundle"), original)

    def test_json_inputs_are_bounded_and_reject_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture_context(root); self.prepare(root)
            for value in (b'{"schema":1,"schema":1}', b"{" + b"x" * (subject.MAX_JSON_BYTES + 1)):
                (root / "bundle" / "provenance.json").write_bytes(value)
                with self.assertRaises(ValueError):
                    subject.verify(root / "bundle")

    def test_verifier_rejects_wrong_nested_provenance_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture_context(root); self.prepare(root)
            path = root / "bundle" / "provenance.json"
            value = json.loads(path.read_bytes())
            value["request_fingerprints"]["SmolLM"] = 1
            path.write_bytes(subject.canonical(value))
            with self.assertRaises(ValueError):
                subject.verify(root / "bundle")

    def test_adapter_image_contains_exact_transfer_entrypoint(self):
        dockerfile = (Path(__file__).parents[2] / "tools" / "compatibility" /
                      "Dockerfile.adapter").read_text(encoding="utf-8")
        self.assertIn(
            "ln -s tools/compatibility/prepare_measurement.py /opt/llm/prepare_measurement.py",
            dockerfile,
        )

    def test_context_symlinks_and_out_of_root_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture_context(root)
            (root / "context" / "models" / "SmolLM" / "unexpected").write_bytes(b"x")
            with self.assertRaises(ValueError): self.prepare(root)

    def test_retained_prepare_context_ignores_build_inputs_for_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_retained_context(root)
            first = self.prepare(root)
            (root / "bundle").rename(root / "first")
            (root / "context" / "spike.py").write_bytes(b"changed build input")
            (root / "context" / "wheelhouse" / "another.whl").write_bytes(b"another")
            second = self.prepare(root)
            self.assertEqual(first, second)
            self.assertEqual(subject.verify(root / "bundle"), second)

    def test_context_rejects_unknown_root_and_symlinked_ignored_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_retained_context(root)
            (root / "context" / "not-from-prepare.py").write_bytes(b"x")
            with self.assertRaises(ValueError): self.prepare(root)

            (root / "context" / "not-from-prepare.py").unlink()
            outside = root / "outside"
            outside.write_bytes(b"x")
            ignored = root / "context" / "spike.py"
            ignored.unlink()
            ignored.symlink_to(outside)
            with self.assertRaises(ValueError): self.prepare(root)
            outside = root / "outside"; outside.write_bytes(b"x")
            link = root / "context" / "models" / "CoEdIT" / SPECS["CoEdIT"][0]
            link.unlink(); link.symlink_to(outside)
            with self.assertRaises(ValueError): self.prepare(root)

    def test_python_caches_are_ignored_for_identity_but_checked_for_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_retained_context(root)
            cache = root / "context" / "services" / "llm" / "__pycache__"
            cache.mkdir(parents=True)
            (cache / "contracts.cpython-313.pyc").write_bytes(b"first")
            first = self.prepare(root)
            (root / "bundle").rename(root / "first")
            (cache / "contracts.cpython-313.pyc").write_bytes(b"changed")
            second = self.prepare(root)
            self.assertEqual(first, second)

            (cache / "unsafe.pyc").symlink_to(root / "outside")
            with self.assertRaises(ValueError):
                self.prepare(root)

            (cache / "unsafe.pyc").unlink()
            os.mkfifo(cache / "unsafe.pyc")
            with self.assertRaises(ValueError):
                self.prepare(root)


if __name__ == "__main__":
    unittest.main()
