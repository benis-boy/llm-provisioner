from pathlib import Path
import json
import tempfile
import threading
import unittest
from unittest import mock

from services.llm.provisioning.artifacts import SPECS
from services.llm.provisioning.volume import provision, verify_current
import services.llm.provisioning.volume as volume


def source(root: Path, model: str, suffix: str = "") -> Path:
    path = root / model
    path.mkdir(parents=True)
    for name in SPECS[model]:
        (path / name).write_bytes((model + name + suffix).encode())
    return path


class ArtifactVolumeTests(unittest.TestCase):
    def test_verify_current_is_read_only_and_returns_minimal_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "source", Path(tmp) / "volume"
            model = source(root, "SmolLM")
            document = provision({"SmolLM": model}, output)
            before = sorted(p.name for p in output.iterdir())
            result = verify_current(output)
            self.assertEqual(result["manifestSha256"], document["manifest_sha256"])
            self.assertEqual(result["volumeId"], None)
            self.assertEqual(result["models"], [{"modelId": "SmolLM", "fileCount": 2,
                                                 "totalBytes": sum((output / document["manifest_sha256"] / "models/SmolLM" / n).stat().st_size for n in SPECS["SmolLM"])}])
            self.assertEqual(before, sorted(p.name for p in output.iterdir()))

    def test_verify_current_rejects_unsafe_or_changed_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "source", Path(tmp) / "volume"
            document = provision({"SmolLM": source(root, "SmolLM")}, output)
            (output / "current").unlink()
            (output / "current").symlink_to("../outside")
            with self.assertRaises(ValueError):
                verify_current(output)
            (output / "current").unlink()
            (output / "current").symlink_to(document["manifest_sha256"])
            manifest_path = output / document["manifest_sha256"] / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_current(output)

    def test_verify_current_rejects_exact_set_violations_and_changed_selected_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "source", Path(tmp) / "volume"
            document = provision({"SmolLM": source(root, "SmolLM")}, output)
            artifact = output / document["manifest_sha256"]
            selected_file = artifact / "models" / "SmolLM" / SPECS["SmolLM"][0]
            selected_file.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_current(output)

            selected_file.write_bytes(("SmolLM" + SPECS["SmolLM"][0]).encode())
            (artifact / "models" / "SmolLM" / "unexpected").write_bytes(b"extra")
            with self.assertRaises(ValueError):
                verify_current(output)

    def test_verify_current_rejects_manifest_size_duplicates_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "source", Path(tmp) / "volume"
            document = provision({"SmolLM": source(root, "SmolLM")}, output)
            artifact = output / document["manifest_sha256"]
            manifest_path = artifact / "manifest.json"
            original = manifest_path.read_text(encoding="utf-8")
            manifest_path.write_text(original[:-1] + ",\"schema\":1}", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_current(output)

            manifest_path.write_text("x" * (64 * 1024 + 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_current(output)

            manifest_path.unlink()
            manifest_path.symlink_to(root / "not-a-manifest")
            with self.assertRaises(ValueError):
                verify_current(output)

    def test_verify_current_rejects_current_traversal_and_non_directory_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "source", Path(tmp) / "volume"
            document = provision({"SmolLM": source(root, "SmolLM")}, output)
            current = output / "current"
            current.unlink()
            current.symlink_to("../" + document["manifest_sha256"])
            with self.assertRaises(ValueError):
                verify_current(output)
            current.unlink()
            current.symlink_to("0" * 64)
            with self.assertRaises(ValueError):
                verify_current(output)
    def test_deterministic_exact_set_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "source", Path(tmp) / "volume"
            model = source(root, "SmolLM")
            first = provision({"SmolLM": model}, output)
            second = provision({"SmolLM": model}, output)
            self.assertEqual(first, second)
            self.assertEqual(set((output / first["manifest_sha256"] / "models/SmolLM").iterdir()),
                             {output / first["manifest_sha256"] / "models/SmolLM" / name for name in SPECS["SmolLM"]})
            self.assertEqual((output / "current").readlink(), Path(first["manifest_sha256"]))

    def test_missing_or_changed_source_fails_and_current_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "source", Path(tmp) / "volume"
            model = source(root, "SmolLM")
            provision({"SmolLM": model}, output)
            current = (output / "current").readlink()
            (model / "Modelfile").unlink()
            with self.assertRaises(ValueError):
                provision({"SmolLM": model}, output)
            self.assertEqual((output / "current").readlink(), current)

    def test_existing_manifest_and_models_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = source(root, "SmolLM")
            output = root / "out"
            document = provision({"SmolLM": model}, output)
            artifact = output / document["manifest_sha256"]
            manifest_path = artifact / "manifest.json"
            manifest_path.unlink()
            manifest_path.symlink_to(root / "not-a-manifest")
            with self.assertRaises(ValueError):
                provision({"SmolLM": model}, output)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = source(root, "SmolLM")
            output = root / "out"
            document = provision({"SmolLM": model}, output)
            models_path = output / document["manifest_sha256"] / "models"
            models_path.rename(models_path.with_name("models-real"))
            models_path.symlink_to(models_path.with_name("models-real"), target_is_directory=True)
            with self.assertRaises(ValueError):
                provision({"SmolLM": model}, output)

    def test_existing_digest_directory_must_match_requested_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = source(root, "SmolLM")
            output = root / "out"
            document = provision({"SmolLM": model}, output)
            manifest_path = output / document["manifest_sha256"] / "manifest.json"
            manifest_path.write_text(json.dumps({"schema": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                provision({"SmolLM": model}, output)

    def test_overlapping_source_roots_and_source_file_substitution_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smol = source(root, "SmolLM")
            coedit = source(smol / "nested", "CoEdIT")
            with self.assertRaises(ValueError):
                provision({"SmolLM": smol, "CoEdIT": coedit}, root / "out")
            replacement = root / "replacement"
            replacement.write_bytes(b"unsafe")
            selected = smol / "Modelfile"
            selected.unlink()
            selected.symlink_to(replacement)
            with self.assertRaises((ValueError, OSError)):
                provision({"SmolLM": smol}, root / "out2")

    def test_manifest_schema_and_counts_reject_boolean_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = source(root, "SmolLM")
            document = provision({"SmolLM": model}, root / "out")
            for entry in (document, document["models"]["SmolLM"]):
                broken = json.loads(json.dumps(document))
                if entry is document:
                    broken["schema"] = True
                else:
                    broken["models"]["SmolLM"]["required_count"] = True
                with self.assertRaises(ValueError):
                    volume._validate_selected(broken)

    def test_changed_valid_input_gets_new_digest_and_switches_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "source", Path(tmp) / "volume"
            model = source(root, "SmolLM", "one")
            first = provision({"SmolLM": model}, output)
            (model / "Modelfile").write_bytes(b"two")
            second = provision({"SmolLM": model}, output)
            self.assertNotEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual((output / "current").readlink(), Path(second["manifest_sha256"]))
            self.assertTrue((output / first["manifest_sha256"]).is_dir())

    def test_source_path_does_not_affect_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = source(root / "a", "SmolLM")
            second = source(root / "b", "SmolLM")
            one = provision({"SmolLM": first}, root / "out-a")
            two = provision({"SmolLM": second}, root / "out-b")
            self.assertEqual(one["manifest_sha256"], two["manifest_sha256"])

    def test_symlink_source_parent_and_unrelated_staging_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = source(root, "SmolLM")
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                provision({"SmolLM": linked}, root / "out")
            output = root / "out2"
            output.mkdir()
            unrelated = output / ".artifact-volume-staging-not-owned"
            unrelated.mkdir()
            provision({"SmolLM": real}, output)
            self.assertTrue(unrelated.exists())

    def test_corrupt_digest_and_unsafe_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "source", Path(tmp) / "volume"
            model = source(root, "SmolLM")
            document = provision({"SmolLM": model}, output)
            (output / document["manifest_sha256"] / "manifest.json").write_text("{}")
            with self.assertRaises(ValueError):
                provision({"SmolLM": model}, output)
            with self.assertRaises(ValueError):
                provision({"SmolLM": model}, output / "../outside")

    def test_owned_staging_cleanup_and_concurrent_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "source", Path(tmp) / "volume"
            model = source(root, "SmolLM")
            output.mkdir()
            old = output / ".artifact-volume-staging-old"
            old.mkdir()
            (old / ".artifact-volume-staging").write_text('{"owned":true}')
            unrelated = output / ".artifact-volume-staging-unrelated"
            unrelated.mkdir()
            results = []
            errors = []
            def run():
                try:
                    results.append(provision({"SmolLM": model}, output)["manifest_sha256"])
                except Exception as exc:  # pragma: no cover - diagnostic for concurrent failures
                    errors.append(exc)
            threads = [threading.Thread(target=run) for _ in range(2)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertFalse(errors)
            self.assertEqual(len(set(results)), 1)
            self.assertFalse((output / ".artifact-volume-staging-old").exists())
            self.assertTrue(unrelated.exists())

    def test_staging_cleanup_requires_the_owned_marker_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp), Path(tmp) / "volume"
            model = source(root, "SmolLM")
            output.mkdir()
            unrelated = output / ".artifact-volume-staging-untrusted"
            unrelated.mkdir()
            (unrelated / ".artifact-volume-staging").write_text('{"owned": false}')
            provision({"SmolLM": model}, output)
            self.assertTrue(unrelated.exists())

    def test_copy_and_current_replace_faults_preserve_current_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp), Path(tmp) / "volume"
            model = source(root, "SmolLM", "one")
            first = provision({"SmolLM": model}, output)
            current = (output / "current").readlink()
            (model / "Modelfile").write_bytes(b"two")
            with mock.patch.object(volume.shutil, "copyfileobj", side_effect=OSError("copy fault")):
                with self.assertRaises(OSError):
                    provision({"SmolLM": model}, output)
            self.assertEqual((output / "current").readlink(), current)
            self.assertFalse(any(p.name.startswith(".artifact-volume-staging-") and
                                 (p / ".artifact-volume-staging").exists() for p in output.iterdir()))
            with mock.patch.object(volume.os, "replace", side_effect=OSError("rename fault")):
                with self.assertRaises(OSError):
                    provision({"SmolLM": model}, output)
            self.assertEqual((output / "current").readlink(), current)


if __name__ == "__main__":
    unittest.main()
