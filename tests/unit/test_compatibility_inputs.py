import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from services.llm.provisioning.artifacts import ArtifactSpec, inventory, manifest, spec, validate, verify_manifest


COMPATIBILITY = Path(__file__).parents[2] / "tools" / "compatibility"


def _module(name: str):
    module_spec = importlib.util.spec_from_file_location(name, COMPATIBILITY / f"{name}.py")
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _docker_sources(context: Path) -> set[str]:
    sources = set()
    for line in (context / "Dockerfile").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[:1] == ["COPY"]:
            sources.add(fields[1].rstrip("/"))
    return sources


class CompatibilityInputTests(unittest.TestCase):
    def test_prepare_and_refresh_stage_all_flat_image_imports_without_torch(self):
        """The candidate image imports only staged modules before its CUDA child starts."""
        prepare = _module("prepare")
        refresh = _module("refresh_context")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_roots = {}
            for model in ("SmolLM", "CoEdIT", "GECToR"):
                model_root = root / model
                model_root.mkdir()
                for relative in spec(model, model_root).required:
                    (model_root / relative).write_bytes(relative.encode())
                model_roots[model] = model_root
            output = root / "context"

            def write_lock(_wheelhouse, lock):
                lock.write_text("# test lock\n", encoding="utf-8")

            with (patch.object(prepare, "_write_lock", write_lock),
                  patch.object(prepare, "_sha256", return_value=prepare.OLLAMA_SHA256),
                  patch.object(prepare.subprocess, "run"),
                  patch.object(sys, "argv", ["prepare.py", "--output", str(output),
                                              "--smollm-root", str(model_roots["SmolLM"]),
                                              "--coedit-root", str(model_roots["CoEdIT"]),
                                              "--gector-root", str(model_roots["GECToR"]), "--download"])):
                self.assertEqual(prepare.main(), 0)
            # The mocked downloader has no output; its presence is otherwise
            # part of the Dockerfile's flat build-context contract.
            (output / "ollama-linux-amd64.tgz").write_bytes(b"test archive")

            # Simulate the omitted flat helper that a refresh must repair.
            (output / "input_bounds.py").unlink()
            with patch.object(sys, "argv", ["refresh_context.py", "--output", str(output)]):
                self.assertEqual(refresh.main(), 0)

            for source in _docker_sources(output):
                self.assertTrue((output / source).exists(), source)
            # -S prevents site packages (and therefore local Torch) from making
            # an incomplete staged layout appear importable.
            completed = subprocess.run(
                [sys.executable, "-S", "-c", "import artifacts, input_bounds, model_runtime, spike, rm_spike"],
                cwd=output, check=False, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_missing_transitive_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            result = inventory(spec("CoEdIT", root))
            self.assertIn("config.json", result["missing"])
            with self.assertRaises(ValueError):
                validate(result)

    def test_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                inventory(ArtifactSpec("test", Path(directory), ("../outside",)))

    def test_manifest_is_deterministic_and_hashes_only_required_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in spec("SmolLM", root).required:
                (root / name).write_bytes(name.encode())
            (root / "unrelated.bin").write_bytes(b"not selected")
            entry = inventory(spec("SmolLM", root))
            result = manifest({"SmolLM": entry})
            self.assertEqual(result, manifest({"SmolLM": inventory(spec("SmolLM", root))}))
            self.assertEqual({f["path"] for f in result["models"]["SmolLM"]["files"]},
                             set(spec("SmolLM", root).required))

    def test_unselected_metadata_cannot_influence_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in spec("SmolLM", root).required:
                (root / name).write_bytes(name.encode())
            (root / "config.json").write_text('{"model_type":"llama","weights":["omitted"]}')
            result = inventory(spec("SmolLM", root))
            self.assertNotIn("metadata", result)

    def test_manifest_verification_detects_changed_selected_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in spec("SmolLM", root).required:
                (root / name).write_bytes(name.encode())
            document = manifest({"SmolLM": inventory(spec("SmolLM", root))})
            verify_manifest(document)
            (root / "Modelfile").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_manifest(document)


if __name__ == "__main__":
    unittest.main()
