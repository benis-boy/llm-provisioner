import hashlib
import os
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.deployment.prepare_inputs import assemble, union_locks


def wheel(path: Path, name: str = "demo", version: str = "1") -> None:
    normalized = name.replace("-", "_").replace(".", "_")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{normalized}-{version}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")


class DeploymentInputsTests(unittest.TestCase):
    def test_conflicting_identity_and_canonical_duplicates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one").write_text("demo-package==1 --hash=sha256:" + "a" * 64 + "\n")
            (root / "two").write_text("demo_package==2 --hash=sha256:" + "b" * 64 + "\n")
            with self.assertRaisesRegex(ValueError, "lock conflict"):
                union_locks((root / "one", root / "two"))

    def test_real_wheel_metadata_is_selected_and_normalized(self):
        with TemporaryDirectory() as directory:
            root, house = Path(directory), Path(directory) / "wheelhouse"
            house.mkdir()
            candidate = house / "demo_package-1-py3-none-any.whl"
            wheel(candidate, "demo-package")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            lock = root / "requirements.lock"
            lock.write_text(f"demo_package==1 --hash=sha256:{digest}\n")
            archive = root / "ollama.tgz"
            archive.write_bytes(b"archive")
            output = root / "inputs"
            assemble(locks=(lock,), wheelhouses=(house,), ollama=archive, output=output,
                     ollama_sha256=hashlib.sha256(b"archive").hexdigest())
            self.assertEqual([candidate.name], [p.name for p in (output / "wheelhouse").iterdir()])

    def test_dotted_local_version_and_build_tag_are_parsed_as_separate_wheel_fields(self):
        with TemporaryDirectory() as directory:
            root, house = Path(directory), Path(directory) / "wheelhouse"
            house.mkdir()
            candidate = house / "demo_package-1.2.3+local-2abc-py3-none-any.whl"
            wheel(candidate, "demo-package", "1.2.3+local")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            lock = root / "requirements.lock"
            lock.write_text(f"demo.package==1.2.3+local --hash=sha256:{digest}\n")
            archive = root / "ollama.tgz"
            archive.write_bytes(b"archive")
            output = root / "inputs"
            assemble(locks=(lock,), wheelhouses=(house,), ollama=archive, output=output,
                     ollama_sha256=hashlib.sha256(b"archive").hexdigest())
            self.assertTrue((output / "wheelhouse" / candidate.name).is_file())

    def test_top_level_metadata_is_used_when_wheel_vendors_dist_info(self):
        with TemporaryDirectory() as directory:
            root, house = Path(directory), Path(directory) / "wheelhouse"
            house.mkdir()
            candidate = house / "setuptools-84.0.0-py3-none-any.whl"
            wheel(candidate, "setuptools", "84.0.0")
            with zipfile.ZipFile(candidate, "a") as archive:
                archive.writestr(
                    "setuptools/_vendor/packaging-26.0.dist-info/METADATA",
                    "Name: packaging\nVersion: 26.0\n",
                )
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            lock = root / "requirements.lock"
            lock.write_text(f"setuptools==84.0.0 --hash=sha256:{digest}\n")
            archive = root / "ollama.tgz"
            archive.write_bytes(b"archive")
            output = root / "inputs"
            assemble(locks=(lock,), wheelhouses=(house,), ollama=archive, output=output,
                     ollama_sha256=hashlib.sha256(b"archive").hexdigest())
            self.assertTrue((output / "wheelhouse" / candidate.name).is_file())

    def test_canonical_duplicate_locks_with_same_values_are_accepted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            digest = "a" * 64
            (root / "one").write_text(f"demo-package==1.2.3 --hash=sha256:{digest}\n")
            (root / "two").write_text(f"demo_package==1.2.3 --hash=sha256:{digest}\n")
            merged = union_locks((root / "one", root / "two"))
            self.assertEqual("demo-package", merged["demo-package"].name)

    def test_corruption_does_not_replace_existing_output(self):
        with TemporaryDirectory() as directory:
            root, house = Path(directory), Path(directory) / "wheelhouse"
            house.mkdir()
            candidate = house / "demo-1-py3-none-any.whl"
            wheel(candidate)
            lock = root / "requirements.lock"
            lock.write_text("demo==1 --hash=sha256:" + "a" * 64 + "\n")
            archive = root / "ollama.tgz"
            archive.write_bytes(b"archive")
            output = root / "inputs"
            output.mkdir()
            (output / "marker").write_text("old")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                assemble(locks=(lock,), wheelhouses=(house,), ollama=archive, output=output,
                         ollama_sha256=hashlib.sha256(b"archive").hexdigest())
            self.assertEqual("old", (output / "marker").read_text())

    def test_staged_wheel_corruption_fails_before_replacing_existing_output(self):
        with TemporaryDirectory() as directory:
            root, house = Path(directory), Path(directory) / "wheelhouse"
            house.mkdir()
            candidate = house / "demo-1.2.3-py3-none-any.whl"
            wheel(candidate, version="1.2.3")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            lock = root / "requirements.lock"
            lock.write_text(f"demo==1.2.3 --hash=sha256:{digest}\n")
            archive = root / "ollama.tgz"
            archive.write_bytes(b"archive")
            output = root / "inputs"
            output.mkdir()
            (output / "marker").write_text("old")
            real_copy = __import__("shutil").copy2

            def corrupt_copy(source, destination, *args, **kwargs):
                result = real_copy(source, destination, *args, **kwargs)
                if Path(destination).suffix == ".whl":
                    Path(destination).write_bytes(b"corrupt")
                return result

            with patch("tools.deployment.prepare_inputs.shutil.copy2", side_effect=corrupt_copy):
                with self.assertRaisesRegex(ValueError, "staged wheel changed"):
                    assemble(locks=(lock,), wheelhouses=(house,), ollama=archive, output=output,
                             ollama_sha256=hashlib.sha256(b"archive").hexdigest())
            self.assertEqual("old", (output / "marker").read_text())

    def test_rejects_symlink_output_and_source_overlap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "link"
            output.symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                assemble(locks=(), wheelhouses=(), ollama=source / "archive", output=output,
                          ollama_sha256="0" * 64)

    def test_output_overlap_with_lock_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "inputs"
            output.mkdir()
            lock = output / "requirements.lock"
            lock.write_text("")
            archive = root / "ollama.tgz"
            archive.write_bytes(b"archive")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                assemble(locks=(lock,), wheelhouses=(), ollama=archive, output=output,
                         ollama_sha256=hashlib.sha256(b"archive").hexdigest())

    def test_failed_commit_restores_output_and_preserves_unowned_old_sibling(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "ollama.tgz"
            archive.write_bytes(b"archive")
            output = root / "inputs"
            output.mkdir()
            (output / "marker").write_text("old")
            unrelated_old = root / "inputs.old"
            unrelated_old.mkdir()
            (unrelated_old / "marker").write_text("keep")
            real_replace = os.replace
            calls = 0

            def fail_new_commit(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected rename failure")
                return real_replace(source, destination)

            with patch("tools.deployment.prepare_inputs.os.replace", side_effect=fail_new_commit):
                with self.assertRaisesRegex(OSError, "injected rename failure"):
                    assemble(locks=(), wheelhouses=(), ollama=archive, output=output,
                             ollama_sha256=hashlib.sha256(b"archive").hexdigest())
            self.assertEqual("old", (output / "marker").read_text())
            self.assertEqual("keep", (unrelated_old / "marker").read_text())

    def test_postcommit_owned_backup_cleanup_failure_is_explicit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "ollama.tgz"
            archive.write_bytes(b"archive")
            output = root / "inputs"
            output.mkdir()
            (output / "marker").write_text("old")
            with patch("tools.deployment.prepare_inputs.shutil.rmtree", side_effect=OSError("injected cleanup failure")):
                with self.assertRaisesRegex(RuntimeError, "output committed; owned backup cleanup failed"):
                    assemble(locks=(), wheelhouses=(), ollama=archive, output=output,
                             ollama_sha256=hashlib.sha256(b"archive").hexdigest())
            self.assertTrue((output / "provenance.txt").is_file())


if __name__ == "__main__":
    unittest.main()
