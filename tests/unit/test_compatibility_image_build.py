import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
import sys
import time
from unittest.mock import Mock, patch


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location("image_build", ROOT / "tools/compatibility/image_build.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class CompatibilityImageBuildTests(unittest.TestCase):
    def _wheel(self, path, name="demo", version="1"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{name}-{version}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")

    def _root(self, base_id="sha256:base"):
        root = Path(tempfile.mkdtemp())
        (root / "tools/compatibility").mkdir(parents=True)
        (root / "services").mkdir()
        (root / "services/a.py").write_text("a")
        (root / "tools/compatibility/Dockerfile.adapter").write_text("COPY services/ /opt/llm/services/\n")
        (root / "tools/compatibility/Dockerfile.adapter.dockerignore").write_text("**\n!services/\n!services/**\n")
        deps = root / ".compatibility/adapter-deps/wheelhouse"
        deps.mkdir(parents=True)
        wheel = deps / "demo-1-py3-none-any.whl"
        self._wheel(wheel)
        digest = module.sha256(wheel)
        (deps.parent / "requirements.lock").write_text(f"demo==1 --hash=sha256:{digest}\n")
        return root, {"Id": base_id, "Os": "linux", "Architecture": "amd64", "RepoDigests": ["repo@sha256:digest"]}

    def _actual_root(self):
        root = Path(tempfile.mkdtemp())
        shutil.copytree(ROOT / "services", root / "services")
        destination = root / "tools/compatibility"
        destination.mkdir(parents=True)
        for name in ("Dockerfile.adapter", "Dockerfile.adapter.dockerignore"):
            shutil.copy(ROOT / "tools/compatibility" / name, destination / name)
        deps = root / ".compatibility/adapter-deps/wheelhouse"
        deps.mkdir(parents=True)
        wheel = deps / "demo-1-py3-none-any.whl"
        self._wheel(wheel)
        (deps.parent / "requirements.lock").write_text(
            f"demo==1 --hash=sha256:{module.sha256(wheel)}\n"
        )
        for line in (ROOT / "tools/compatibility/Dockerfile.adapter").read_text().splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[0] == "COPY" and fields[1].startswith("tools/"):
                target = root / fields[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                source = ROOT / fields[1]
                if source.is_dir():
                    shutil.copytree(source, target, dirs_exist_ok=True)
                else:
                    shutil.copy(source, target)
        return root, {"Id": "sha256:base", "Os": "linux", "Architecture": "amd64", "RepoDigests": ["repo@sha256:digest"]}

    def test_identity_is_deterministic_and_changes_for_inputs(self):
        root, base = self._root()
        first = module.calculate_identity(root, base)
        self.assertEqual(first, module.calculate_identity(root, base))
        (root / "services/a.py").write_text("changed")
        self.assertNotEqual(first["identity"], module.calculate_identity(root, base)["identity"])

    def test_mutable_tag_is_accepted_only_after_immutable_inspect(self):
        seen = Mock(return_value=module.subprocess.CompletedProcess([], 0, '{"Id":"sha256:x"}', ""))
        self.assertEqual(module.inspect_image("local:tag", seen)["Id"], "sha256:x")
        seen.assert_called_once()
        bad = Mock(return_value=module.subprocess.CompletedProcess([], 0, '{"RepoTags":["local:tag"]}', ""))
        with self.assertRaises(ValueError):
            module.inspect_image("local:tag", bad)

    def test_build_command_contains_offline_reproducibility_flags(self):
        root, base = self._root()
        identity = module.calculate_identity(root, base)
        command = module.build_command(root, "adapter:current", "base:tag", identity)
        for flag in ("--platform=linux/amd64", "--network=none", "--provenance=false", "--sbom=false", "--load"):
            self.assertIn(flag, command)

    def test_identity_write_is_atomic_and_idempotent(self):
        root, base = self._root()
        document = module.calculate_identity(root, base)
        path = root / "identity.json"
        module.write_identity(path, document)
        before = path.stat().st_mtime_ns
        module.write_identity(path, document)
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertEqual(json.loads(path.read_text()), document)

    def test_actual_repository_copy_contract_and_identity(self):
        root, base = self._actual_root()
        identity = module.calculate_identity(root, base)
        copied = {item["path"] for item in identity["sources"]}
        for source in module._copy_sources(root):
            self.assertIn(source, copied)
        self.assertGreater(len(copied), 5)

    def test_measurement_entrypoint_preserves_package_import_contract(self):
        dockerfile = (ROOT / "tools/compatibility/Dockerfile.adapter").read_text()
        dockerignore = (ROOT / "tools/compatibility/Dockerfile.adapter.dockerignore").read_text()
        self.assertIn("COPY tools/compatibility/ /opt/llm/tools/compatibility/", dockerfile)
        self.assertIn("ln -s tools/compatibility/measure_profiles.py /opt/llm/measure_profiles.py", dockerfile)
        self.assertIn("!tools/compatibility/**", dockerignore)

        root, base = self._actual_root()
        identity = module.calculate_identity(root, base)
        paths = {item["path"] for item in identity["sources"]}
        self.assertIn("tools/compatibility/__init__.py", paths)
        self.assertIn("tools/compatibility/measure_profiles.py", paths)
        self.assertIn("tools/compatibility/prepare_measurement.py", paths)

        stage = module._stage_context(root, identity)
        try:
            staged = Path(stage.name)
            self.assertTrue((staged / "tools/compatibility/__init__.py").is_file())
            self.assertTrue((staged / "tools/compatibility/measure_profiles.py").is_file())
            self.assertTrue((staged / "tools/compatibility/prepare_measurement.py").is_file())
            self.assertIn("COPY tools/compatibility/ /opt/llm/tools/compatibility/", (staged / "tools/compatibility/Dockerfile.adapter").read_text())
        finally:
            stage.cleanup()

    def test_ignored_bytecode_does_not_change_identity(self):
        root, base = self._actual_root()
        before = module.calculate_identity(root, base)
        ignored = root / "services/__pycache__/junk.pyc"
        ignored.parent.mkdir(exist_ok=True)
        ignored.write_bytes(b"one")
        self.assertEqual(before, module.calculate_identity(root, base))
        ignored.write_bytes(b"two")
        self.assertEqual(before, module.calculate_identity(root, base))

    def test_wheelhouse_rejects_extra_wheel(self):
        root, _ = self._root()
        self._wheel(root / ".compatibility/adapter-deps/wheelhouse/other-2-py3-none-any.whl", "other", "2")
        with self.assertRaises(ValueError):
            module._lock_inputs(root)

    def test_dependency_inputs_reject_symlinks(self):
        for relative in ("requirements.lock", "wheelhouse", "wheelhouse/demo-1-py3-none-any.whl"):
            root, _ = self._root()
            path = root / ".compatibility/adapter-deps" / relative
            target = root / "replacement"
            if path.is_dir():
                shutil.rmtree(path)
                target.mkdir()
            else:
                path.unlink()
                target.write_text("replacement")
            path.symlink_to(target, target_is_directory=target.is_dir())
            with self.assertRaises(ValueError):
                module._lock_inputs(root)

    def test_wheel_filename_accepts_pep427_build_tag(self):
        root, _ = self._root()
        wheelhouse = root / ".compatibility/adapter-deps/wheelhouse"
        old = wheelhouse / "demo-1-py3-none-any.whl"
        wheel = wheelhouse / "demo-1-2build-py3-none-any.whl"
        old.rename(wheel)
        (wheelhouse.parent / "requirements.lock").write_text(
            f"demo==1 --hash=sha256:{module.sha256(wheel)}\n"
        )
        self.assertEqual(module._lock_inputs(root)[0][0]["name"], "demo")

    def test_base_platform_is_required_and_identity_binds_it(self):
        root, base = self._root()
        for field, value in (("Os", "windows"), ("Architecture", "arm64")):
            rejected = dict(base); rejected[field] = value
            with self.assertRaises(ValueError):
                module.calculate_identity(root, rejected)
        identity = module.calculate_identity(root, base)
        self.assertEqual(identity["base"]["platform"], "linux/amd64")

    def test_lock_rejects_duplicate_entry(self):
        root, _ = self._root()
        lock = root / ".compatibility/adapter-deps/requirements.lock"
        lock.write_text(lock.read_text() * 2)
        with self.assertRaises(ValueError):
            module._lock_inputs(root)

    def test_wheel_name_and_version_must_match_lock(self):
        root, _ = self._root()
        wheel = root / ".compatibility/adapter-deps/wheelhouse/demo-1-py3-none-any.whl"
        wheel.rename(wheel.with_name("other-1-py3-none-any.whl"))
        with self.assertRaises(ValueError):
            module._lock_inputs(root)

    def test_mutable_base_retarget_between_inspection_and_build_is_rejected(self):
        root, base = self._root()
        identity = module.calculate_identity(root, base)
        calls = []
        first = module.subprocess.CompletedProcess([], 0, json.dumps(base), "")
        changed = module.subprocess.CompletedProcess([], 0, json.dumps({"Id": "sha256:new", "Os": "linux", "Architecture": "amd64", "RepoDigests": base["RepoDigests"]}), "")

        def runner(command, **kwargs):
            calls.append(command)
            if command[2:4] == ["inspect", "candidate"]:
                return first if len([c for c in calls if c[2:4] == ["inspect", "candidate"]]) == 1 else changed
            return module.subprocess.CompletedProcess(command, 0, "", "")

        args = ["build-adapter", "--root", str(root), "--base-image", "candidate", "--tag", "out", "--identity", str(root / "identity.json")]
        module.write_identity(root / "identity.json", identity)
        with patch.object(module, "_run", side_effect=runner):
            self.assertEqual(module.main(args), 2)
        self.assertFalse(any(command[:3] == ["docker", "tag", base["Id"]] for command in calls))

    def test_owned_base_uses_digest_or_unique_tag_and_removes_only_owned_tag(self):
        inspected = {"Id": "sha256:id", "Os": "linux", "Architecture": "amd64", "RepoDigests": []}
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            if command[:3] == ["docker", "image", "inspect"]:
                return module.subprocess.CompletedProcess(command, 0, json.dumps(inspected), "")
            return module.subprocess.CompletedProcess(command, 0, "", "")
        ref, owned = module._owned_base_ref("mutable:tag", inspected, runner)
        self.assertEqual(ref, owned)
        self.assertRegex(owned, r"^compatibility-owned-base-[0-9a-f]+:build$")
        # The mutable user ref is never used for the owned tag or its cleanup.
        self.assertEqual(ref, owned)
        self.assertEqual(calls[0], ["docker", "tag", "sha256:id", owned])
        self.assertEqual(calls[1][:3], ["docker", "image", "inspect"])
        self.assertEqual(calls[1][3], owned)
        digest_runner = Mock(return_value=module.subprocess.CompletedProcess([], 0, json.dumps({"Id": "x"}), ""))
        self.assertEqual(module._owned_base_ref("tag", {"Id": "x", "RepoDigests": ["repo@sha256:x"]}, digest_runner), ("repo@sha256:x", None))
        digest_runner.assert_called_once()

    def test_owned_base_tag_lifecycle_is_exact_and_cleanup_is_owned_only(self):
        root, base = self._root()
        base["RepoDigests"] = []
        identity = module.calculate_identity(root, base)
        identity_path = root / "identity.json"
        module.write_identity(identity_path, identity)
        calls = []

        def runner(command, **kwargs):
            calls.append(command)
            if command[:3] == ["docker", "image", "inspect"]:
                return module.subprocess.CompletedProcess(command, 0, json.dumps(base), "")
            return module.subprocess.CompletedProcess(command, 0, "", "")

        args = ["build-adapter", "--root", str(root), "--base-image", "mutable:tag", "--tag", "out", "--identity", str(identity_path)]
        with patch.object(module, "_run", side_effect=runner):
            self.assertEqual(module.main(args), 0)
        tag = next(command for command in calls if command[:2] == ["docker", "tag"])
        owned = tag[3]
        build = next(command for command in calls if command[:3] == ["docker", "buildx", "build"])
        self.assertIn(f"BASE_IMAGE={owned}", build)
        cleanup = [command for command in calls if command[:3] == ["docker", "image", "rm"]]
        self.assertEqual(cleanup, [["docker", "image", "rm", owned]])
        self.assertNotIn("mutable:tag", " ".join(cleanup[0]))

    def test_build_command_is_independent_of_invocation_cwd(self):
        root, base = self._root()
        identity = module.calculate_identity(root, base)
        old = Path.cwd()
        try:
            os.chdir("/")
            command = module.build_command(root, "tag", "repo@sha256:x", identity)
        finally:
            os.chdir(old)
        self.assertEqual(command[-1], str(root))
        self.assertIn("BASE_IMAGE=repo@sha256:x", command)

    def test_staged_context_isolated_from_later_live_source_changes(self):
        root, base = self._root()
        identity = module.calculate_identity(root, base)
        stage = module._stage_context(root, identity)
        try:
            staged = Path(stage.name) / "services/a.py"
            staged_deps = Path(stage.name) / ".compatibility/adapter-deps"
            staged_lock = staged_deps / "requirements.lock"
            staged_wheels = staged_deps / "wheelhouse"
            self.assertEqual(staged.read_text(), "a")
            self.assertTrue(staged_lock.is_file())
            self.assertFalse(staged_lock.is_symlink())
            self.assertEqual(staged_lock.read_bytes(), (root / ".compatibility/adapter-deps/requirements.lock").read_bytes())
            self.assertEqual(staged_lock.stat().st_mode & 0o777, 0o644)
            self.assertEqual(staged_lock.stat().st_mtime, 0)
            self.assertEqual(module._lock_inputs(Path(stage.name))[0], identity["adapter_inputs"])
            self.assertEqual(
                {wheel.name for wheel in staged_wheels.iterdir()},
                {"demo-1-py3-none-any.whl"},
            )
            staged_wheel = staged_wheels / "demo-1-py3-none-any.whl"
            self.assertTrue(staged_wheel.is_file())
            self.assertFalse(staged_wheel.is_symlink())
            self.assertEqual(staged_wheel.stat().st_mode & 0o777, 0o644)
            self.assertEqual(staged_wheel.stat().st_mtime, 0)
            self.assertEqual(
                module.sha256(staged_wheel),
                identity["adapter_inputs"][0]["sha256"],
            )
            (root / "services/a.py").write_text("changed after staging")
            self.assertEqual(staged.read_text(), "a")
            command = module.build_command(Path(stage.name), "tag", "base", identity)
            self.assertEqual(command[-1], stage.name)
        finally:
            stage.cleanup()

    def test_nested_source_and_input_tampering_is_rejected(self):
        root, base = self._root()
        identity = module.calculate_identity(root, base)
        module.write_identity(root / "identity.json", identity)
        (root / "services/nested.py").write_text("tampered")
        with self.assertRaises(ValueError):
            module.verify("image", root / "identity.json", root, "base", None, Mock())

    def _verify_with_probe(self, stdout, stderr=""):
        root, base = self._root()
        identity = module.calculate_identity(root, base)
        runner = Mock(side_effect=[
            module.subprocess.CompletedProcess([], 0, json.dumps(base), ""),
            module.subprocess.CompletedProcess([], 0, json.dumps({"Id": "img", "Config": {"Labels": {
                module.LABELS["schema"]: module.SCHEMA, module.LABELS["role"]: module.ROLE,
                module.LABELS["identity"]: identity["identity"], module.LABELS["base"]: base["Id"],
                module.LABELS["source"]: identity["source_sha256"], module.LABELS["inputs"]: identity["inputs_sha256"]}, "Entrypoint": module.ENTRYPOINT}}), ""),
            module.subprocess.CompletedProcess([], 0, stdout, stderr),
        ])
        path = root / "identity.json"
        module.write_identity(path, identity)
        return root, path, runner, identity

    def test_runtime_versions_require_exact_output(self):
        for output, error in (("1\n", ""), ("", ""), ("1\n2\n", ""), ("1\n", "warning")):
            root, path, runner, _ = self._verify_with_probe(output, error)
            if output == "1\n" and not error:
                module.verify("image", path, root, "base", None, runner)
            else:
                with self.assertRaises(ValueError):
                    module.verify("image", path, root, "base", None, runner)

    def test_identity_stales_after_source_lock_or_base_change(self):
        root, base = self._root()
        identity = module.calculate_identity(root, base)
        mutations = [
            (lambda: (root / "services/a.py").write_text("changed"), base),
            (lambda: (root / ".compatibility/adapter-deps/requirements.lock").write_text("demo==2 --hash=sha256:" + "0" * 64 + "\n"), base),
            (lambda: None, {"Id": "sha256:other", "RepoDigests": base["RepoDigests"]}),
        ]
        for mutate, current_base in mutations:
            mutate()
            with self.assertRaises(ValueError):
                module._verify_current(root, "base", None, identity, Mock(return_value=module.subprocess.CompletedProcess([], 0, json.dumps(current_base), "")))

    def test_runner_drains_large_output_with_bounded_buffers(self):
        code = "import sys; sys.stdout.write('x'*50000); sys.stderr.write('y'*50000)"
        result = module._run([sys.executable, "-c", code], timeout=5)
        self.assertEqual(len(result.stdout), module.MAX_OUTPUT)
        self.assertEqual(len(result.stderr), module.MAX_OUTPUT)

    def test_runner_timeout_kills_descendants_that_hold_pipes(self):
        code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)"
        started = time.monotonic()
        with self.assertRaises(RuntimeError):
            module._run([sys.executable, "-c", code], timeout=.1)
        self.assertLess(time.monotonic() - started, 3)

    def test_copy_source_symlinks_fail_closed(self):
        root, _ = self._root()
        outside = root.parent / "outside.py"
        outside.write_text("outside")
        (root / "services/a.py").unlink()
        (root / "services/a.py").symlink_to(outside)
        with self.assertRaises(ValueError):
            module._copy_sources(root)
        root, _ = self._root()
        (root / "services").rename(root / "real-services")
        (root / "services").symlink_to(root / "real-services", target_is_directory=True)
        with self.assertRaises(ValueError):
            module._copy_sources(root)

    def test_wheel_metadata_must_match_lock_and_filename(self):
        root, _ = self._root()
        wheel = root / ".compatibility/adapter-deps/wheelhouse/demo-1-py3-none-any.whl"
        self._wheel(wheel, "demo", "2")
        (root / ".compatibility/adapter-deps/requirements.lock").write_text(
            f"demo==1 --hash=sha256:{module.sha256(wheel)}\n"
        )
        with self.assertRaises(ValueError):
            module._lock_inputs(root)

    def test_documented_verify_command_has_required_current_source_inputs(self):
        docs = (ROOT / "docs/compatibility-image-build.md").read_text()
        command = next(line for line in docs.splitlines() if "image_build.py verify --root ." in line)
        self.assertIn("--root .", command)
        self.assertIn("--base-image", docs[docs.index(command):docs.index(command) + 300])
        self.assertIn("--base-context", docs[docs.index(command):docs.index(command) + 300])

    def test_unsupported_copy_forms_fail_closed(self):
        root, _ = self._root()
        dockerfile = root / "tools/compatibility/Dockerfile.adapter"
        for line in ("COPY --from=stage services/ /x/\n", "COPY [\"services/\", \"/x/\"]\n", "COPY services/ other/ /x/\n", "COPY services/* /x/\n"):
            dockerfile.write_text(line)
            with self.assertRaises(ValueError):
                module._copy_sources(root)


if __name__ == "__main__":
    unittest.main()
