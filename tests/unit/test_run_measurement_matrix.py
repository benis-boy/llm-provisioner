import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tools.compatibility import run_measurement_matrix as subject
from tools.compatibility import measure_profiles
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.profiles import BenchmarkMetadata, CorruptProfileStore


@dataclass(frozen=True)
class _TransportProfile:
    model_id: measure_profiles.ModelId
    profile_identity: str
    internal_details: str = "/private/full-profile-must-not-be-serialized"


class MeasurementMatrixRunnerTests(unittest.TestCase):
    @staticmethod
    def _real_measured_registry(path):
        matrix = tuple(subject.measurement_matrix())
        summaries = []
        baseline = tuple(SampleMetadata(1, wave, 1, 100, 100, (100,)) for wave in range(1, 5))
        warmup = (SampleMetadata(1, 0, 1, 100, 100, (100,)),)
        measured = tuple(SampleMetadata(1, wave, 1, 100, 100, (100,)) for wave in range(1, 5))
        with subject.ProfileStore(path) as store:
            for model, selector in matrix:
                context = 512 if model is measure_profiles.ModelId.SMOLLM else None
                bucket = None if context is not None else selector
                fingerprint = hashlib.sha256(model.value.encode()).hexdigest()
                identity = dict(model_id=model.value, gpu_uuid="GPU-test",
                    artifact_manifest_hash="a" * 64, model_hash="b" * 64,
                    runtime_identity="test-runtime", adapter_identity="test-adapter",
                    context_size=context, bucket_identity=bucket, fingerprint=fingerprint)
                profile_identity = hashlib.sha256(json.dumps(identity, sort_keys=True,
                    separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
                profile = CapacityProfile(model, identity["gpu_uuid"], identity["artifact_manifest_hash"],
                    identity["model_hash"], identity["runtime_identity"], identity["adapter_identity"],
                    profile_identity, 1, 1, 1, 20, baseline + warmup + measured, context, bucket)
                metadata = BenchmarkMetadata(fingerprint, "2026-10-07T00:00:00Z", "unit-fixture-not-GPU-proof",
                    baseline, warmup, measured, bucket or "context:512")
                store.save_measured(profile, metadata)
                summaries.append(dict(model=model.value, profile_identity=profile_identity))
        connection = sqlite3.connect(path)
        try:
            if connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone() != (0, 0, 0):
                raise AssertionError("fixture checkpoint did not complete")
        finally:
            connection.close()
        return dict(status="complete", matrix=[[model.value, selector] for model, selector in matrix],
                    profiles=summaries)

    def _real_registry_copy_runner(self, source, stages, after_copy=None):
        def runner(command, **kwargs):
            if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                stage = Path(command[-1])
                stages.append(stage)
                shutil.copyfile(source, stage)
                if after_copy is not None:
                    after_copy(stage)
            return subprocess.CompletedProcess(command, 0, "", "")
        return self._owned_runner(runner)

    def test_real_checkpointed_sqlite_audit_commits_without_stage_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.sqlite", root / "profiles.sqlite"
            result = self._real_measured_registry(source)
            self.assertFalse(Path(str(source) + "-wal").exists())
            self.assertFalse(Path(str(source) + "-shm").exists())
            # Real mode=ro reads of a WAL-mode, fully checkpointed registry can
            # create persistent sidecars despite making no database writes.
            with subject.ProfileStore.open_readonly(source) as store:
                self.assertEqual(len(store.validate_all_measured({
                    item["profile_identity"] for item in result["profiles"]})), 3)
            observed = {suffix: Path(str(source) + suffix).stat().st_size for suffix in ("-wal", "-shm")}
            self.assertEqual(observed["-wal"], 0)
            self.assertGreater(observed["-shm"], 0)
            self.assertTrue(all(stat.S_ISREG(Path(str(source) + suffix).lstat().st_mode)
                                for suffix in ("-wal", "-shm")))
            stages, descriptors = [], []
            original_open = os.open
            def opened(*args, **kwargs):
                fd = original_open(*args, **kwargs)
                descriptors.append(fd)
                return fd
            with patch.object(subject, "run", side_effect=self._real_registry_copy_runner(source, stages)), \
                 patch.object(subject, "_read_runtime_result", return_value=result), \
                 patch.object(subject.os, "open", side_effect=opened):
                try:
                    response = subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                        gpu_uuid="GPU-test", ollama_version="v", ceiling=1, output=output))
                except subject.OperatorFailure as exc:
                    residue = {entry.name: entry.stat().st_size for entry in stages[0].parent.iterdir()}
                    self.fail(f"real readonly audit failed: {exc.code}; stage residue={residue}")
            self.assertEqual(response["status"], "complete")
            self.assertEqual(response["profiles"], result["profiles"])
            self.assertEqual(output.read_bytes(), source.read_bytes())
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)
            self.assertFalse(stages[0].parent.exists())
            self.assertFalse(Path(str(output) + "-wal").exists())
            self.assertFalse(Path(str(output) + "-shm").exists())
            self.assertFalse((root / ".profiles.sqlite.measurement.lock").exists())
            for fd in descriptors:
                with self.assertRaises(OSError):
                    os.fstat(fd)

    def test_real_sqlite_audit_failure_cleans_reader_generated_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.sqlite", root / "profiles.sqlite"
            result = self._real_measured_registry(source)
            result["profiles"][0]["profile_identity"] = "f" * 64
            stages, descriptors = [], []
            original_open = os.open
            def opened(*args, **kwargs):
                fd = original_open(*args, **kwargs)
                descriptors.append(fd)
                return fd
            with patch.object(subject, "run", side_effect=self._real_registry_copy_runner(source, stages)), \
                 patch.object(subject, "_read_runtime_result", return_value=result), \
                 patch.object(subject.os, "open", side_effect=opened), \
                 self.assertRaises(subject.OperatorFailure) as raised:
                subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                    gpu_uuid="GPU-test", ollama_version="v", ceiling=1, output=output))
            self.assertEqual(raised.exception.code, "export_audit_failed")
            self.assertIsInstance(raised.exception.__cause__, CorruptProfileStore)
            self.assertFalse(output.exists())
            self.assertFalse(stages[0].parent.exists())
            self.assertFalse((root / ".profiles.sqlite.measurement.lock").exists())
            for fd in descriptors:
                with self.assertRaises(OSError):
                    os.fstat(fd)

    def test_real_sqlite_audit_interrupt_closes_reader_before_sidecar_capture(self):
        for interrupted_at in ("construction", "validation"):
            with self.subTest(interrupted_at=interrupted_at), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output = root / "source.sqlite", root / "profiles.sqlite"
                result = self._real_measured_registry(source)
                original = KeyboardInterrupt("real audit interrupted")
                stages, readers, descriptors = [], [], []
                original_inspect = subject.ProfileStore._inspect_existing
                original_capture = subject._capture_stage_sidecars
                original_open = os.open
                def opened(*args, **kwargs):
                    fd = original_open(*args, **kwargs)
                    descriptors.append(fd)
                    return fd
                def inspected(reader):
                    original_inspect(reader)
                    readers.append(reader)
                    if interrupted_at == "construction":
                        raise original
                def validated(reader, expected):
                    raise original
                def captured(host):
                    self.assertEqual(len(readers), 1)
                    with self.assertRaises(sqlite3.ProgrammingError):
                        readers[0]._db.execute("SELECT 1")
                    original_capture(host)
                    self.assertEqual(len(host["stage_sidecars"]), 2)
                with patch.object(subject, "run", side_effect=self._real_registry_copy_runner(source, stages)), \
                     patch.object(subject, "_read_runtime_result", return_value=result), \
                     patch.object(subject.ProfileStore, "_inspect_existing", new=inspected), \
                     patch.object(subject.ProfileStore, "validate_all_measured", new=validated), \
                     patch.object(subject, "_capture_stage_sidecars", side_effect=captured), \
                     patch.object(subject.os, "open", side_effect=opened), \
                     self.assertRaises(KeyboardInterrupt) as raised:
                    subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                        gpu_uuid="GPU-test", ollama_version="v", ceiling=1, output=output))
                self.assertIs(raised.exception, original)
                self.assertFalse(output.exists())
                self.assertFalse(stages[0].parent.exists())
                self.assertFalse((root / ".profiles.sqlite.measurement.lock").exists())
                for fd in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(fd)

    def test_staged_real_sqlite_audit_rejects_preexisting_sidecars_without_deleting_them(self):
        for kind in ("wal", "shm", "dangling_wal"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output = root / "source.sqlite", root / "profiles.sqlite"
                result = self._real_measured_registry(source)
                stages, foreign = [], []
                def after_copy(stage):
                    sidecar = Path(str(stage) + ("-shm" if kind == "shm" else "-wal"))
                    foreign.append(sidecar)
                    if kind == "dangling_wal":
                        sidecar.symlink_to(root / "missing-target")
                    else:
                        sidecar.write_bytes(b"foreign preexisting sidecar")
                with patch.object(subject, "run", side_effect=self._real_registry_copy_runner(source, stages, after_copy)), \
                     patch.object(subject, "_read_runtime_result", return_value=result), \
                     self.assertRaises(subject.OperatorFailure):
                    subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                        gpu_uuid="GPU-test", ollama_version="v", ceiling=1, output=output))
                self.assertFalse(output.exists())
                self.assertTrue(os.path.lexists(foreign[0]))
                if kind == "dangling_wal":
                    self.assertTrue(foreign[0].is_symlink())
                    self.assertFalse((root / "missing-target").exists())
                else:
                    self.assertEqual(foreign[0].read_bytes(), b"foreign preexisting sidecar")
                self.assertFalse((root / ".profiles.sqlite.measurement.lock").exists())

    def test_real_sqlite_audit_preserves_replaced_sidecar_and_refuses_commit(self):
        for suffix in ("-wal", "-shm"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output = root / "source.sqlite", root / "profiles.sqlite"
                result = self._real_measured_registry(source)
                stages, foreign, descriptors = [], [], []
                original_capture, original_open = subject._capture_stage_sidecars, os.open
                def opened(*args, **kwargs):
                    fd = original_open(*args, **kwargs)
                    descriptors.append(fd)
                    return fd
                def captured(host):
                    original_capture(host)
                    sidecar = Path(str(host["temporary"]) + suffix)
                    self.assertIn(sidecar, host["stage_sidecars"])
                    sidecar.rename(root / "original-reader-sidecar")
                    sidecar.write_bytes(b"foreign replacement")
                    foreign.append(sidecar)
                with patch.object(subject, "run", side_effect=self._real_registry_copy_runner(source, stages)), \
                     patch.object(subject, "_read_runtime_result", return_value=result), \
                     patch.object(subject, "_capture_stage_sidecars", side_effect=captured), \
                     patch.object(subject.os, "open", side_effect=opened), \
                     self.assertRaises(subject.OperatorFailure) as raised:
                    subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                        gpu_uuid="GPU-test", ollama_version="v", ceiling=1, output=output))
                self.assertEqual(raised.exception.code, "cleanup_failed")
                self.assertFalse(raised.exception.db_retained)
                self.assertFalse(output.exists())
                self.assertEqual(foreign[0].read_bytes(), b"foreign replacement")
                other = Path(str(stages[0]) + ("-shm" if suffix == "-wal" else "-wal"))
                self.assertFalse(other.exists())
                self.assertFalse((root / ".profiles.sqlite.measurement.lock").exists())
                for fd in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(fd)

    def test_real_sqlite_export_unknown_stage_entry_refuses_commit_without_recursive_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.sqlite", root / "profiles.sqlite"
            result = self._real_measured_registry(source)
            stages, foreign = [], []
            def after_copy(stage):
                entry = stage.parent / "foreign-entry"
                entry.write_bytes(b"must survive")
                foreign.append(entry)
            with patch.object(subject, "run", side_effect=self._real_registry_copy_runner(source, stages, after_copy)), \
                 patch.object(subject, "_read_runtime_result", return_value=result), \
                 self.assertRaises(subject.OperatorFailure):
                subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                    gpu_uuid="GPU-test", ollama_version="v", ceiling=1, output=output))
            self.assertFalse(output.exists())
            self.assertEqual(foreign[0].read_bytes(), b"must survive")
            self.assertFalse((root / ".profiles.sqlite.measurement.lock").exists())

    def test_real_sqlite_sidecar_cleanup_failure_blocks_commit_and_preserves_interruptions(self):
        for failure in (PermissionError("sidecar removal failed"),
                        KeyboardInterrupt("sidecar cleanup interrupted"), SystemExit(33)):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output = root / "source.sqlite", root / "profiles.sqlite"
                result = self._real_measured_registry(source)
                stages, descriptors = [], []
                original_unlink, original_open = Path.unlink, os.open
                failed = False
                def unlinked(path, *args, **kwargs):
                    nonlocal failed
                    if path.name == "profiles.sqlite-wal" and not failed:
                        failed = True
                        raise failure
                    return original_unlink(path, *args, **kwargs)
                def opened(*args, **kwargs):
                    fd = original_open(*args, **kwargs)
                    descriptors.append(fd)
                    return fd
                with patch.object(subject, "run", side_effect=self._real_registry_copy_runner(source, stages)), \
                     patch.object(subject, "_read_runtime_result", return_value=result), \
                     patch.object(Path, "unlink", unlinked), \
                     patch.object(subject.os, "open", side_effect=opened), \
                     self.assertRaises(subject.OperatorFailure if isinstance(failure, Exception)
                                       else type(failure)) as raised:
                    subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                        gpu_uuid="GPU-test", ollama_version="v", ceiling=1, output=output))
                if not isinstance(failure, Exception):
                    self.assertIs(raised.exception, failure)
                else:
                    self.assertEqual(raised.exception.code, "cleanup_failed")
                self.assertTrue(failed)
                self.assertFalse(output.exists())
                self.assertFalse(stages[0].parent.exists())
                self.assertFalse((root / ".profiles.sqlite.measurement.lock").exists())
                for fd in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(fd)


    def test_partial_export_copy_failures_clean_owned_stage_and_allow_same_output_retry(self):
        for failure in (subprocess.CalledProcessError(1, ["docker", "cp"]),
                        KeyboardInterrupt("copy interrupted"), SystemExit(19)):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "profiles.sqlite"
                lock = root / ".profiles.sqlite.measurement.lock"
                original_open = os.open
                descriptors, stages, initial_states = [], [], []
                failed = False
                def opened(*args, **kwargs):
                    fd = original_open(*args, **kwargs)
                    descriptors.append(fd)
                    return fd
                def runner(command, **kwargs):
                    nonlocal failed
                    if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                        stage = Path(command[-1])
                        stages.append(stage)
                        initial_states.append((stage.is_file(), stage.parent.stat().st_mode & 0o777))
                        stage.write_bytes(b"partial" if not failed else b"audited sqlite")
                        if not failed:
                            failed = True
                            raise failure
                    return subprocess.CompletedProcess(command, 0, "", "")
                args = argparse.Namespace(image="image", bundle=root, gpu_uuid="GPU-1",
                    ollama_version="v", ceiling=1, output=output)
                with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                     patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result()), \
                     patch.object(subject, "_audit_export"), \
                     patch.object(subject.os, "open", side_effect=opened):
                    with self.assertRaises(subject.OperatorFailure if isinstance(failure, Exception)
                                           else type(failure)) as raised:
                        subject.orchestrate(args)
                    if not isinstance(failure, Exception):
                        self.assertIs(raised.exception, failure)
                    self.assertFalse(output.exists())
                    self.assertFalse(lock.exists())
                    self.assertFalse(stages[0].exists())
                    self.assertFalse(stages[0].parent.exists())
                    for fd in descriptors:
                        with self.assertRaises(OSError):
                            os.fstat(fd)
                    response = subject.orchestrate(args)
                self.assertEqual(response["status"], "complete")
                self.assertEqual(output.read_bytes(), b"audited sqlite")
                self.assertEqual(output.stat().st_mode & 0o777, 0o444)
                self.assertEqual(initial_states, [(True, 0o700), (True, 0o700)])
                self.assertNotEqual(stages[0].parent, stages[1].parent)
                self.assertFalse(stages[1].parent.exists())
                for fd in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(fd)

    def test_foreign_private_stage_file_or_directory_substitution_is_preserved(self):
        for replacement in ("file", "directory"):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "profiles.sqlite"
                original_open = os.open
                descriptors, stages = [], []
                original = KeyboardInterrupt("audit interrupted")
                def opened(*args, **kwargs):
                    fd = original_open(*args, **kwargs)
                    descriptors.append(fd)
                    return fd
                def runner(command, **kwargs):
                    if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                        stages.append(Path(command[-1]))
                        stages[-1].write_bytes(b"audited sqlite")
                    return subprocess.CompletedProcess(command, 0, "", "")
                def audited(stage, _result):
                    self.assertNotEqual(stage.parent, root)
                    if replacement == "file":
                        stage.rename(stage.parent / "original-file")
                    else:
                        stage.parent.rename(root / "original-directory")
                        stage.parent.mkdir(mode=0o700)
                    stage.write_bytes(b"foreign")
                    raise original
                with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                     patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result()), \
                     patch.object(subject, "_audit_export", side_effect=audited), \
                     patch.object(subject.os, "open", side_effect=opened), \
                     self.assertRaises(KeyboardInterrupt) as raised:
                    subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                        gpu_uuid="GPU-1", ollama_version="v", ceiling=1, output=output))
                self.assertIs(raised.exception, original)
                self.assertEqual(stages[0].read_bytes(), b"foreign")
                self.assertTrue(stages[0].parent.is_dir())
                self.assertFalse(output.exists())
                self.assertFalse((root / ".profiles.sqlite.measurement.lock").exists())
                for fd in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(fd)

    def test_successful_copy_may_replace_precreated_stage_file_without_losing_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profiles.sqlite"
            stages = []
            def runner(command, **kwargs):
                if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                    stage = Path(command[-1])
                    stages.append(stage)
                    self.assertTrue(stage.is_file())
                    initial = stage.stat().st_ino
                    stage.unlink()
                    stage.write_bytes(b"audited sqlite")
                    self.assertNotEqual(stage.stat().st_ino, initial)
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result()), \
                 patch.object(subject, "_audit_export"):
                response = subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                    gpu_uuid="GPU-1", ollama_version="v", ceiling=1, output=output))
            self.assertEqual(response["status"], "complete")
            self.assertEqual(output.read_bytes(), b"audited sqlite")
            self.assertFalse(stages[0].parent.exists())

    def test_failed_copy_replacement_is_not_adopted_or_deleted_and_retry_uses_fresh_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profiles.sqlite"
            stages, descriptors = [], []
            original_open = os.open
            failed = False
            def opened(*args, **kwargs):
                fd = original_open(*args, **kwargs)
                descriptors.append(fd)
                return fd
            def runner(command, **kwargs):
                nonlocal failed
                if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                    stage = Path(command[-1])
                    stages.append(stage)
                    if not failed:
                        failed = True
                        initial = stage.stat().st_ino
                        stage.unlink()
                        stage.write_bytes(b"unproved replacement")
                        self.assertNotEqual(stage.stat().st_ino, initial)
                        raise subprocess.CalledProcessError(1, command)
                    stage.write_bytes(b"audited sqlite")
                return subprocess.CompletedProcess(command, 0, "", "")
            args = argparse.Namespace(image="image", bundle=root, gpu_uuid="GPU-1",
                ollama_version="v", ceiling=1, output=output)
            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result()), \
                 patch.object(subject, "_audit_export"), \
                 patch.object(subject.os, "open", side_effect=opened):
                with self.assertRaises(subject.OperatorFailure) as raised:
                    subject.orchestrate(args)
                self.assertEqual(raised.exception.code, "cleanup_failed")
                self.assertFalse(output.exists())
                self.assertFalse((root / ".profiles.sqlite.measurement.lock").exists())
                self.assertEqual(stages[0].read_bytes(), b"unproved replacement")
                for fd in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(fd)
                response = subject.orchestrate(args)
            self.assertEqual(response["status"], "complete")
            self.assertNotEqual(stages[0].parent, stages[1].parent)
            self.assertEqual(stages[0].read_bytes(), b"unproved replacement")
            self.assertFalse(stages[1].parent.exists())

    def test_staging_initialization_interrupt_releases_proved_stage_lock_and_descriptors(self):
        for failure_stage in ("directory_open", "file_fstat"):
            with self.subTest(failure_stage=failure_stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                original_open, original_fstat = os.open, os.fstat
                original = KeyboardInterrupt("staging initialization interrupted")
                descriptors, commands = [], []
                interrupted = False
                def opened(path, flags, *args, **kwargs):
                    nonlocal interrupted
                    if failure_stage == "directory_open" and flags & os.O_DIRECTORY and not interrupted:
                        interrupted = True
                        raise original
                    fd = original_open(path, flags, *args, **kwargs)
                    descriptors.append(fd)
                    return fd
                def inspected(fd):
                    nonlocal interrupted
                    value = original_fstat(fd)
                    if failure_stage == "file_fstat" and value.st_size == 0 and not interrupted:
                        interrupted = True
                        raise original
                    return value
                def runner(command, **kwargs):
                    commands.append(command)
                    return subprocess.CompletedProcess(command, 0, "", "")
                with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                     patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result()), \
                     patch.object(subject.os, "open", side_effect=opened), \
                     patch.object(subject.os, "fstat", side_effect=inspected), \
                     self.assertRaises(KeyboardInterrupt) as raised:
                    subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                        gpu_uuid="GPU-1", ollama_version="v", ceiling=1, output=root / "profiles.sqlite"))
                self.assertIs(raised.exception, original)
                self.assertFalse((root / "profiles.sqlite").exists())
                self.assertFalse((root / ".profiles.sqlite.measurement.lock").exists())
                self.assertEqual(list(root.glob(".profiles.sqlite.stage-*")), [])
                self.assertFalse(any(command[:2] == ["docker", "cp"] and
                                     "/export/profiles.sqlite" in command[2] for command in commands))
                for fd in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(fd)

    def test_secondary_stage_directory_cleanup_interrupt_preserves_original_and_closes_fds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_open, original_rmdir = os.open, Path.rmdir
            original = KeyboardInterrupt("audit interrupted")
            secondary = SystemExit(29)
            descriptors, stages = [], []
            def opened(*args, **kwargs):
                fd = original_open(*args, **kwargs)
                descriptors.append(fd)
                return fd
            def removed(path):
                if ".stage-" in path.name:
                    raise secondary
                return original_rmdir(path)
            def runner(command, **kwargs):
                if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                    stages.append(Path(command[-1]))
                    stages[-1].write_bytes(b"audited sqlite")
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result()), \
                 patch.object(subject, "_audit_export", side_effect=original), \
                 patch.object(subject.os, "open", side_effect=opened), \
                 patch.object(Path, "rmdir", removed), \
                 self.assertRaises(KeyboardInterrupt) as raised:
                subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                    gpu_uuid="GPU-1", ollama_version="v", ceiling=1, output=root / "profiles.sqlite"))
            self.assertIs(raised.exception, original)
            self.assertFalse((root / "profiles.sqlite").exists())
            self.assertFalse((root / ".profiles.sqlite.measurement.lock").exists())
            self.assertFalse(stages[0].exists())
            self.assertTrue(stages[0].parent.exists())
            for fd in descriptors:
                with self.assertRaises(OSError):
                    os.fstat(fd)


    def test_measurement_result_file_and_cli_stdout_compose_with_exact_outer_validation(self):
        matrix = tuple(subject.measurement_matrix())
        profiles = measure_profiles._MeasuredRun(
            [_TransportProfile(model, model.value + "-identity") for model, _ in matrix], matrix)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            bundle = root / "bundle"
            result_file = runtime / "measurement-result.json"
            config = SimpleNamespace(profile_db=runtime / "profiles.sqlite")
            argv = ["measure_profiles.py", "--bundle", str(bundle), "--db", str(config.profile_db),
                    "--result-file", str(result_file), "--ollama-version", "0.11.6"]
            measured = AsyncMock(return_value=profiles)
            with patch.object(measure_profiles, "verify_bundle_inputs", return_value=(
                    bundle / "config.json", bundle / "requests.json", bundle / "provenance.json", "a" * 64)), \
                 patch.object(measure_profiles, "load_config", return_value=config), \
                 patch.object(measure_profiles, "_run", new=measured), \
                 patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO) as output:
                self.assertEqual(measure_profiles.main(), 0)
            measured.assert_awaited_once()
            stdout, persisted = output.getvalue(), result_file.read_text()
            self.assertEqual(stdout, persisted)
            self.assertEqual(len(stdout.splitlines()), 1)
            for transport, text in (("stdout", stdout), ("file", persisted)):
                with self.subTest(transport=transport):
                    self.assertLessEqual(len(text.encode()), subject.MAX_RESULT)
                    accepted = subject._result(text)
                    self.assertEqual(accepted["matrix"], [[model.value, selector] for model, selector in matrix])
                    self.assertEqual([item["model"] for item in accepted["profiles"]],
                                     ["SmolLM", "CoEdIT", "GECToR"])
                    self.assertEqual([item["profile_identity"] for item in accepted["profiles"]],
                                     [profile.profile_identity for profile in profiles])
                    self.assertEqual(len({item["profile_identity"] for item in accepted["profiles"]}), 3)
                    self.assertTrue(all(set(item) == {"model", "profile_identity"}
                                        for item in accepted["profiles"]))
                    self.assertNotIn("/private", text)
                    self.assertNotIn("internal_details", text)
            unknown = json.loads(persisted)
            unknown["profiles"][0]["model"] = "unknown"
            with self.assertRaises(ValueError):
                subject._result(json.dumps(unknown))

    def test_measurement_transport_rejects_identity_only_profiles_as_unknown(self):
        matrix = tuple(subject.measurement_matrix())
        profiles = measure_profiles._MeasuredRun(
            [model.value + "-identity" for model, _ in matrix], matrix)
        with tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory) / "measurement-result.json"
            measure_profiles._write_measurement_result(result_file, profiles)
            self.assertEqual([item["model"] for item in json.loads(result_file.read_text())["profiles"]],
                             ["unknown", "unknown", "unknown"])
            with self.assertRaises(ValueError):
                subject._result(result_file.read_text())

    def test_measurement_result_writer_rejects_oversized_identity_summary_before_publication(self):
        matrix = tuple(subject.measurement_matrix())
        profiles = measure_profiles._MeasuredRun([
            _TransportProfile(model, "x" * (subject.MAX_RESULT + 1) if index == 0 else model.value + "-identity")
            for index, (model, _) in enumerate(matrix)], matrix)
        with tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory) / "measurement-result.json"
            with self.assertRaises(ValueError):
                measure_profiles._write_measurement_result(result_file, profiles)
            self.assertFalse(result_file.exists())
            self.assertFalse(result_file.with_name(".measurement-result.json.tmp").exists())

    def test_interruptions_release_host_resources_and_attempt_remaining_owned_teardown(self):
        for stage in ("measurement", "audit", "container", "volume", "trace", "trace_cleanup", "commit"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "profiles.sqlite"
                lock = root / ".profiles.sqlite.measurement.lock"
                temporary = None
                commands, descriptors = [], []
                original = KeyboardInterrupt("original interruption")
                secondary = SystemExit(17)
                interrupted = False
                secondary_raised = False
                original_open, original_link = os.open, os.link
                def opened(path, flags, *args, **kwargs):
                    fd = original_open(path, flags, *args, **kwargs)
                    if Path(path) == lock:
                        descriptors.append(fd)
                    return fd
                def interrupt():
                    nonlocal interrupted
                    interrupted = True
                    raise original
                def runner(command, **kwargs):
                    nonlocal secondary_raised, temporary
                    commands.append(command)
                    if not interrupted and (
                            stage == "measurement" and command[:3] == ["docker", "start", "-a"]
                            and command[-1].endswith("-measure") or
                            stage == "container" and command[:3] == ["docker", "rm", "-f"] or
                            stage == "volume" and command[:3] == ["docker", "volume", "rm"] or
                            stage in {"trace", "trace_cleanup"} and command[:3] == ["docker", "start", "-a"]
                            and command[-1].endswith("-trace")):
                        interrupt()
                    if interrupted and not secondary_raised and "rm" in command:
                        secondary_raised = True
                        raise secondary
                    if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                        temporary = Path(command[-1])
                        temporary.write_bytes(b"audited sqlite")
                    return subprocess.CompletedProcess(command, 0, "", "")
                def audited(*args):
                    if stage == "audit":
                        interrupt()
                def linked(*args, **kwargs):
                    if stage == "commit":
                        interrupt()
                    return original_link(*args, **kwargs)
                with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                     patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result(),
                                  side_effect=RuntimeError("reader failed") if stage == "trace_cleanup" else None), \
                     patch.object(subject, "_audit_export", side_effect=audited), \
                     patch.object(subject.os, "open", side_effect=opened), \
                     patch.object(subject.os, "link", side_effect=linked), \
                     patch("sys.stderr", new_callable=StringIO), \
                     self.assertRaises(KeyboardInterrupt) as raised:
                    subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                        gpu_uuid="GPU-1", ollama_version="v", ceiling=1, output=output,
                        debug=stage in {"trace", "trace_cleanup"}))
                self.assertIs(raised.exception, original)
                self.assertFalse(output.exists())
                if temporary is not None:
                    self.assertFalse(temporary.exists())
                    self.assertFalse(temporary.parent.exists())
                self.assertEqual(list(root.glob(".profiles.sqlite.stage-*")), [])
                self.assertFalse(lock.exists())
                self.assertEqual(len(descriptors), 1)
                with self.assertRaises(OSError):
                    os.fstat(descriptors[0])
                created = {command[3] for command in commands
                           if command[:3] == ["docker", "create", "--name"]}
                removed = {command[-1] for command in commands
                           if command[:3] == ["docker", "rm", "-f"]}
                self.assertEqual(removed, created)
                self.assertEqual({command[-1] for command in commands
                                  if command[:3] == ["docker", "volume", "rm"]},
                                 {command[-1] for command in commands
                                  if command[:3] == ["docker", "volume", "create"]})

    def test_interrupted_teardown_preserves_foreign_substituted_lock_and_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profiles.sqlite"
            lock = root / ".profiles.sqlite.measurement.lock"
            temporary = None
            original_open = os.open
            descriptors, commands = [], []
            original = KeyboardInterrupt("original interruption")
            interrupted = False
            def opened(path, flags, *args, **kwargs):
                fd = original_open(path, flags, *args, **kwargs)
                if Path(path) == lock:
                    descriptors.append(fd)
                return fd
            def runner(command, **kwargs):
                nonlocal interrupted, temporary
                commands.append(command)
                if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                    temporary = Path(command[-1])
                    temporary.write_bytes(b"audited sqlite")
                if not interrupted and command[:3] == ["docker", "rm", "-f"]:
                    interrupted = True
                    lock.rename(root / "original-lock")
                    temporary.rename(root / "original-staging")
                    lock.write_bytes(b"foreign lock")
                    temporary.write_bytes(b"foreign staging")
                    raise original
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result()), \
                 patch.object(subject, "_audit_export"), \
                 patch.object(subject.os, "open", side_effect=opened), \
                 self.assertRaises(KeyboardInterrupt) as raised:
                subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                    gpu_uuid="GPU-1", ollama_version="v", ceiling=1, output=output))
            self.assertIs(raised.exception, original)
            self.assertFalse(output.exists())
            self.assertEqual(lock.read_bytes(), b"foreign lock")
            self.assertEqual(temporary.read_bytes(), b"foreign staging")
            self.assertEqual(len(descriptors), 1)
            with self.assertRaises(OSError):
                os.fstat(descriptors[0])
            self.assertEqual(len([command for command in commands
                                 if command[:3] == ["docker", "volume", "rm"]]), 4)

    def test_initialization_interrupt_releases_owned_partial_token_and_fd_without_docker(self):
        for stage in ("generation", "partial_write", "fsync"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                lock = root / ".profiles.sqlite.measurement.lock"
                original_open, original_random = os.open, os.urandom
                original_write, original_fsync = os.write, os.fsync
                descriptors = []
                original = KeyboardInterrupt("original interruption")
                def opened(path, flags, *args, **kwargs):
                    fd = original_open(path, flags, *args, **kwargs)
                    if Path(path) == lock:
                        descriptors.append(fd)
                    return fd
                def generated(count):
                    if stage == "generation" and count == 32:
                        raise original
                    return original_random(count)
                def written(fd, data):
                    if stage == "partial_write" and fd in descriptors:
                        original_write(fd, data[:3])
                        raise original
                    return original_write(fd, data)
                def synced(fd):
                    if stage == "fsync" and fd in descriptors:
                        raise original
                    return original_fsync(fd)
                with patch.object(subject.os, "open", side_effect=opened), \
                     patch.object(subject.os, "urandom", side_effect=generated), \
                     patch.object(subject.os, "write", side_effect=written), \
                     patch.object(subject.os, "fsync", side_effect=synced), \
                     patch.object(subject, "run") as docker, \
                     self.assertRaises(KeyboardInterrupt) as raised:
                    subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                        gpu_uuid="GPU-1", ollama_version="v", ceiling=1,
                        output=root / "profiles.sqlite", debug=True))
                self.assertIs(raised.exception, original)
                docker.assert_not_called()
                self.assertFalse(lock.exists())
                self.assertEqual(len(descriptors), 1)
                with self.assertRaises(OSError):
                    os.fstat(descriptors[0])

    def test_original_interrupt_survives_secondary_host_cleanup_failures(self):
        for failed_path in ("temporary", "lock"):
            with self.subTest(failed_path=failed_path), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                lock = root / ".profiles.sqlite.measurement.lock"
                temporary = None
                original_open, original_unlink = os.open, Path.unlink
                descriptors, attempted = [], []
                original = KeyboardInterrupt("original interruption")
                secondary = SystemExit(23)
                def opened(path, flags, *args, **kwargs):
                    fd = original_open(path, flags, *args, **kwargs)
                    if Path(path) == lock:
                        descriptors.append(fd)
                    return fd
                def unlinked(path, *args, **kwargs):
                    attempted.append(path)
                    if path == (temporary if failed_path == "temporary" else lock):
                        raise secondary
                    return original_unlink(path, *args, **kwargs)
                def runner(command, **kwargs):
                    nonlocal temporary
                    if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                        temporary = Path(command[-1])
                        temporary.write_bytes(b"audited sqlite")
                    return subprocess.CompletedProcess(command, 0, "", "")
                with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                     patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result()), \
                     patch.object(subject, "_audit_export", side_effect=original), \
                     patch.object(subject.os, "open", side_effect=opened), \
                     patch.object(Path, "unlink", unlinked), \
                     self.assertRaises(KeyboardInterrupt) as raised:
                    subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                        gpu_uuid="GPU-1", ollama_version="v", ceiling=1,
                        output=root / "profiles.sqlite"))
                self.assertIs(raised.exception, original)
                self.assertIn(temporary, attempted)
                self.assertIn(lock, attempted)
                self.assertFalse((root / "profiles.sqlite").exists())
                self.assertEqual(temporary.exists(), failed_path == "temporary")
                self.assertEqual(lock.exists(), failed_path == "lock")
                with self.assertRaises(OSError):
                    os.fstat(descriptors[0])

    def test_system_exit_is_reraised_after_owned_cleanup_instead_of_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / ".profiles.sqlite.measurement.lock"
            original_open = os.open
            original = SystemExit(31)
            descriptors, commands = [], []
            def opened(path, flags, *args, **kwargs):
                fd = original_open(path, flags, *args, **kwargs)
                if Path(path) == lock:
                    descriptors.append(fd)
                return fd
            def runner(command, **kwargs):
                commands.append(command)
                if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                    raise original
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject.os, "open", side_effect=opened), \
                 self.assertRaises(SystemExit) as raised:
                subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                    gpu_uuid="GPU-1", ollama_version="v", ceiling=1,
                    output=root / "profiles.sqlite"))
            self.assertIs(raised.exception, original)
            self.assertFalse(lock.exists())
            self.assertFalse((root / "profiles.sqlite").exists())
            with self.assertRaises(OSError):
                os.fstat(descriptors[0])
            self.assertEqual(len([command for command in commands
                                 if command[:3] == ["docker", "volume", "rm"]]), 4)

    def test_postcommit_interrupt_keeps_committed_database_and_releases_host_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profiles.sqlite"
            lock = root / ".profiles.sqlite.measurement.lock"
            original_open, original_link = os.open, os.link
            original = KeyboardInterrupt("original interruption")
            descriptors = []
            def opened(path, flags, *args, **kwargs):
                fd = original_open(path, flags, *args, **kwargs)
                if Path(path) == lock:
                    descriptors.append(fd)
                return fd
            def linked(*args, **kwargs):
                original_link(*args, **kwargs)
                raise original
            def runner(command, **kwargs):
                if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                    Path(command[-1]).write_bytes(b"audited sqlite")
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result()), \
                 patch.object(subject, "_audit_export"), \
                 patch.object(subject.os, "open", side_effect=opened), \
                 patch.object(subject.os, "link", side_effect=linked), \
                 self.assertRaises(KeyboardInterrupt) as raised:
                subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                    gpu_uuid="GPU-1", ollama_version="v", ceiling=1, output=output))
            self.assertIs(raised.exception, original)
            self.assertEqual(output.read_bytes(), b"audited sqlite")
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)
            self.assertEqual(list(root.glob(".profiles.sqlite.stage-*")), [])
            self.assertFalse(lock.exists())
            with self.assertRaises(OSError):
                os.fstat(descriptors[0])

    @staticmethod
    def _owned_runner(runner):
        """Supply daemon identities/labels to the existing scenario command fakes.

        Scenario callbacks retain descriptive names for their behavior/assertions;
        the focused identity tests below separately assert the actual ID commands.
        """
        containers, volumes, names = {}, {}, {}
        def wrapped(command, **kwargs):
            scenario_command = [names.get(part.split(":", 1)[0], part.split(":", 1)[0])
                                + (":" + part.split(":", 1)[1] if ":" in part else "")
                                for part in command]
            if command[:3] == ["docker", "create", "--name"]:
                name = command[3]
                identity = hashlib.sha256(name.encode()).hexdigest()
                owner = command[command.index("--label") + 1].split("=", 1)[1]
                names[identity] = name
                containers[name] = {"Id": identity, "Config": {
                    "Labels": {subject.OWNER_LABEL: owner}}}
            if command[:3] == ["docker", "volume", "create"]:
                owner = command[command.index("--label") + 1].split("=", 1)[1]
                volumes[command[-1]] = {"Name": command[-1], "CreatedAt": "initial",
                                       "Labels": {subject.OWNER_LABEL: owner}}
            result = runner(scenario_command, **kwargs)
            if result.returncode == 0 and not result.stdout:
                if command[:3] == ["docker", "create", "--name"]:
                    return subprocess.CompletedProcess(command, 0, containers[command[3]]["Id"] + "\n", "")
                if "inspect" in command:
                    value = (containers if command[1] == "inspect" else volumes).get(scenario_command[-1])
                    return subprocess.CompletedProcess(command, 0 if value else 1,
                        json.dumps(value) if value else "", "" if value else
                        f"Error: No such container: {command[-1]}" if command[1] == "inspect" else
                        f"Error response from daemon: get {command[-1]}: no such volume")
            return result
        return wrapped

    def test_successful_volume_create_requires_positive_owner_proof(self):
        for inspected in ({"Name": "volume", "CreatedAt": "created", "Labels": {
                subject.OWNER_LABEL: "foreign"}}, None):
            with self.subTest(inspected=inspected):
                commands = []
                def runner(command, **kwargs):
                    commands.append(command)
                    text = json.dumps(inspected) if "inspect" in command else "volume\n"
                    return subprocess.CompletedProcess(command, 0, text, "")
                resources = subject._OwnedDockerResources("owner")
                with patch.object(subject, "run", side_effect=runner):
                    with self.assertRaises(RuntimeError):
                        resources.create_volume("volume")
                    resources.cleanup()
                self.assertFalse(any("rm" in command or "--mount" in command for command in commands))

    def test_generic_inspect_not_found_error_does_not_prove_resource_absence(self):
        resources = subject._OwnedDockerResources("owner")
        resources.containers["container"] = "a" * 64
        resources.volumes["volume"] = None
        with patch.object(subject, "run", return_value=subprocess.CompletedProcess(
                [], 1, "", "authorization plugin not found")) as runner:
            self.assertEqual(len(resources.cleanup()), 2)
        self.assertFalse(any("rm" in call.args[0] for call in runner.call_args_list))

    def test_volume_ownership_is_rechecked_before_mount_and_cleanup(self):
        for changed in ("label", "replacement"):
            with self.subTest(changed=changed):
                commands = []
                state = {"Name": "volume", "CreatedAt": "initial",
                         "Labels": {subject.OWNER_LABEL: "owner"}}
                def runner(command, **kwargs):
                    commands.append(command)
                    return subprocess.CompletedProcess(command, 0,
                        json.dumps(state) if "inspect" in command else "volume\n", "")
                resources = subject._OwnedDockerResources("owner")
                with patch.object(subject, "run", side_effect=runner):
                    resources.create_volume("volume")
                    if changed == "label":
                        state["Labels"] = {subject.OWNER_LABEL: "foreign"}
                    else:
                        state["CreatedAt"] = "replacement"
                    with self.assertRaises(RuntimeError):
                        subject._create("reader", "image", [("volume", "/runtime", True)], resources=resources)
                    self.assertTrue(resources.cleanup())
                self.assertFalse(any("rm" in command or "--mount" in command for command in commands))

    def test_container_cleanup_and_use_target_verified_immutable_id(self):
        identity = "a" * 64
        commands = []
        def runner(command, **kwargs):
            commands.append(command)
            if "inspect" in command:
                value = {"Id": identity if command[-1] == identity else "b" * 64,
                         "Config": {"Labels": {subject.OWNER_LABEL: "owner"}}}
                return subprocess.CompletedProcess(command, 0, json.dumps(value), "")
            return subprocess.CompletedProcess(command, 0, identity + "\n", "")
        resources = subject._OwnedDockerResources("owner")
        with patch.object(subject, "run", side_effect=runner):
            subject._create("container", "image", resources=resources)
            reference = resources.container_reference("container")
            self.assertEqual(reference, identity)
            self.assertFalse(resources.cleanup())
        self.assertIn(["docker", "rm", "-f", identity], commands)
        self.assertNotIn(["docker", "rm", "-f", "container"], commands)

    def test_changed_container_labels_block_use_and_removal_even_after_successful_create(self):
        identity = "a" * 64
        commands = []
        owner = "owner"
        def runner(command, **kwargs):
            commands.append(command)
            value = {"Id": identity, "Config": {"Labels": {subject.OWNER_LABEL: owner}}}
            return subprocess.CompletedProcess(command, 0,
                json.dumps(value) if "inspect" in command else identity, "")
        resources = subject._OwnedDockerResources("owner")
        with patch.object(subject, "run", side_effect=runner):
            subject._create("container", "image", resources=resources)
            owner = "foreign"
            with self.assertRaises(RuntimeError):
                resources.container_reference("container")
            self.assertTrue(resources.cleanup())
        self.assertFalse(any("rm" in command for command in commands))

    def test_uncertain_container_creation_cleans_only_inspected_owned_immutable_id(self):
        for inspected_owner in ("owner", "foreign"):
            with self.subTest(owner=inspected_owner):
                identity = "a" * 64
                commands = []
                def runner(command, **kwargs):
                    commands.append(command)
                    if command[:2] == ["docker", "create"]:
                        raise RuntimeError("ambiguous create")
                    value = {"Id": identity, "Config": {"Labels": {
                        subject.OWNER_LABEL: inspected_owner}}}
                    return subprocess.CompletedProcess(command, 0, json.dumps(value), "")
                resources = subject._OwnedDockerResources("owner")
                with patch.object(subject, "run", side_effect=runner):
                    with self.assertRaises(RuntimeError):
                        subject._create("container", "image", resources=resources)
                    self.assertFalse(resources.cleanup())
                removals = [command for command in commands if "rm" in command]
                self.assertEqual(removals, [["docker", "rm", "-f", identity]]
                                 if inspected_owner == "owner" else [])

    def test_successful_foreign_volume_collision_never_mounts_deletes_or_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profiles.sqlite"
            commands = []
            def runner(command, **kwargs):
                commands.append(command)
                value = {"Name": command[-1], "CreatedAt": "initial",
                         "Labels": {subject.OWNER_LABEL: "foreign"}}
                return subprocess.CompletedProcess(command, 0,
                    json.dumps(value) if "inspect" in command else command[-1], "")
            with patch.object(subject, "run", side_effect=runner), \
                 self.assertRaises(subject.OperatorFailure) as raised:
                subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                    gpu_uuid="GPU-1", ollama_version="v", ceiling=1, output=output, debug=True))
            self.assertEqual(raised.exception.code, "volume_setup_failed")
            self.assertFalse(output.exists())
            self.assertFalse(any("rm" in command or "--mount" in command or "cp" in command
                                 or "start" in command for command in commands))

    def test_volume_label_change_during_cleanup_blocks_commit_without_foreign_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profiles.sqlite"
            commands = []
            changed_volume = None
            def runner(command, **kwargs):
                nonlocal changed_volume
                commands.append(command)
                if command[:3] == ["docker", "rm", "-f"] and changed_volume is None:
                    changed_volume = next(call[-1] for call in commands
                                          if call[:3] == ["docker", "volume", "create"])
                if command[:3] == ["docker", "volume", "inspect"] and command[-1] == changed_volume:
                    return subprocess.CompletedProcess(command, 0, json.dumps({
                        "Name": changed_volume, "CreatedAt": "initial",
                        "Labels": {subject.OWNER_LABEL: "foreign"}}), "")
                if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                    Path(command[-1]).write_bytes(b"audited sqlite")
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result()), \
                 patch.object(subject, "_audit_export"), \
                 self.assertRaises(subject.OperatorFailure) as raised:
                subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                    gpu_uuid="GPU-1", ollama_version="v", ceiling=1, output=output))
            self.assertEqual(raised.exception.code, "cleanup_failed")
            self.assertEqual(raised.exception.docker_cleanup, "unproved")
            self.assertFalse(output.exists())
            self.assertNotIn(["docker", "volume", "rm", changed_volume], commands)

    def test_timed_out_preflight_is_tracked_and_removed_only_by_owned_immutable_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profiles.sqlite"
            commands = []
            probe_name = None
            probe_owner = None
            identity = "a" * 64
            def runner(command, **kwargs):
                nonlocal probe_name, probe_owner
                commands.append(command)
                if command[:3] == ["docker", "run", "--rm"]:
                    probe_name = command[command.index("--name") + 1]
                    probe_owner = command[command.index("--label") + 1].split("=", 1)[1]
                    raise subprocess.TimeoutExpired(command, 60)
                if "inspect" in command and command[-1] == probe_name:
                    return subprocess.CompletedProcess(command, 0, json.dumps({
                        "Id": identity, "Config": {"Labels": {
                            subject.OWNER_LABEL: probe_owner}}}), "")
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 self.assertRaises(subject.OperatorFailure) as raised:
                subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                    gpu_uuid="GPU-1", ollama_version="v", ceiling=1, output=output))
            self.assertEqual(raised.exception.code, "timeout")
            self.assertEqual(raised.exception.docker_cleanup, "proved")
            self.assertIn(["docker", "rm", "-f", identity], commands)
            self.assertNotIn(["docker", "rm", "-f", probe_name], commands)
            self.assertFalse(output.exists())

    def test_precommit_staged_stat_failure_still_releases_owned_lock_and_fd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profiles.sqlite"
            lock = root / ".profiles.sqlite.measurement.lock"
            temporary = None
            original_open, original_stat = os.open, Path.stat
            descriptors = []
            cleaned = False
            def opened(path, flags, *args, **kwargs):
                fd = original_open(path, flags, *args, **kwargs)
                if Path(path) == lock:
                    descriptors.append(fd)
                return fd
            def inspected(path, *args, **kwargs):
                if cleaned and path == temporary and kwargs.get("follow_symlinks", True):
                    raise PermissionError("staged stat failed")
                return original_stat(path, *args, **kwargs)
            def runner(command, **kwargs):
                nonlocal cleaned, temporary
                if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                    temporary = Path(command[-1])
                    temporary.write_bytes(b"audited sqlite")
                if "rm" in command:
                    cleaned = True
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject, "_read_runtime_result", return_value=self._complete_matrix_result()), \
                 patch.object(subject, "_audit_export"), \
                 patch.object(subject.os, "open", side_effect=opened), \
                 patch.object(Path, "stat", inspected), \
                 self.assertRaises(subject.OperatorFailure) as raised:
                subject.orchestrate(argparse.Namespace(image="image", bundle=root,
                    gpu_uuid="GPU-1", ollama_version="v", ceiling=1, output=output))
            self.assertEqual(raised.exception.code, "export_audit_failed")
            self.assertEqual(raised.exception.stage, "commit")
            self.assertFalse(output.exists())
            self.assertFalse(lock.exists())
            self.assertFalse(temporary.exists())
            self.assertFalse(temporary.parent.exists())
            self.assertEqual(len(descriptors), 1)
            with self.assertRaises(OSError):
                os.fstat(descriptors[0])

    def test_reservation_initialization_faults_release_owned_inode_and_fd_without_docker(self):
        for fault in ("random", "write", "short_write", "fsync", "foreign_replacement"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "profiles.sqlite"
                lock = root / ".profiles.sqlite.measurement.lock"
                args = argparse.Namespace(image="image", bundle=root, gpu_uuid="GPU-1",
                    ollama_version="v", ceiling=1, output=output, debug=True)
                original_open, original_write, original_fsync = os.open, os.write, os.fsync
                descriptors = []
                def opened(path, flags, *args, **kwargs):
                    fd = original_open(path, flags, *args, **kwargs)
                    if Path(path) == lock:
                        descriptors.append(fd)
                    return fd
                def written(fd, data):
                    if fd in descriptors:
                        if fault == "write":
                            raise OSError("write failed")
                        if fault == "short_write":
                            return original_write(fd, data[:3])
                        if fault == "foreign_replacement":
                            lock.unlink()
                            lock.write_bytes(b"foreign")
                            raise OSError("write failed")
                    return original_write(fd, data)
                def synced(fd):
                    if fault == "fsync" and fd in descriptors:
                        raise OSError("sync failed")
                    return original_fsync(fd)
                with patch.object(subject.os, "open", side_effect=opened), \
                     patch.object(subject.os, "write", side_effect=written), \
                     patch.object(subject.os, "fsync", side_effect=synced), \
                     patch.object(subject.os, "urandom", side_effect=lambda count:
                                  (_ for _ in ()).throw(OSError("random failed"))
                                  if fault == "random" and count == 32 else b"x" * count), \
                     patch.object(subject, "run") as docker, \
                     self.assertRaises(subject.OperatorFailure):
                    subject.orchestrate(args)
                docker.assert_not_called()
                self.assertFalse(output.exists())
                self.assertEqual(len(descriptors), 1)
                with self.assertRaises(OSError):
                    os.fstat(descriptors[0])
                if fault == "foreign_replacement":
                    self.assertEqual(lock.read_bytes(), b"foreign")
                else:
                    self.assertFalse(lock.exists())

    @staticmethod
    def _complete_matrix_result():
        matrix = [[model.value, selector] for model, selector in subject.measurement_matrix()]
        return {"status": "complete", "matrix": matrix,
                "profiles": [{"model": model, "profile_identity": model + "-identity"}
                             for model, _ in matrix]}

    def _run_successful_export(self, root, raw_commands=None):
        bundle = root / "bundle"
        bundle.mkdir()
        output = root / "profiles.sqlite"
        args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                                  ollama_version="0.11.6", ceiling=2, output=output)
        result = self._complete_matrix_result()
        observed = []
        def runner(command, *, timeout=60, check=True):
            observed.append((command, output.exists()))
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-result"):
                return subprocess.CompletedProcess(command, 0, json.dumps(result), "")
            if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                Path(command[-1]).write_bytes(b"audited sqlite")
            return subprocess.CompletedProcess(command, 0, "", "")

        owned_runner = self._owned_runner(runner)
        def observed_runner(command, **kwargs):
            if raw_commands is not None:
                raw_commands.append(command)
            return owned_runner(command, **kwargs)

        with patch.object(subject, "run", side_effect=observed_runner), \
             patch.object(subject, "_audit_export"):
            response = subject.orchestrate(args)
        return output, response, observed

    def test_successful_export_uses_immutable_ids_for_all_container_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            commands = []
            _, response, _ = self._run_successful_export(Path(directory), commands)
        self.assertEqual(response["status"], "complete")
        identities = {hashlib.sha256(command[3].encode()).hexdigest() for command in commands
                      if command[:3] == ["docker", "create", "--name"]}
        starts = [command[-1] for command in commands if command[:3] == ["docker", "start", "-a"]]
        removals = [command[-1] for command in commands if command[:3] == ["docker", "rm", "-f"]]
        copies = [argument.split(":", 1)[0] for command in commands
                  if command[:2] == ["docker", "cp"] for argument in command[2:]
                  if ":" in argument]
        self.assertTrue(starts)
        self.assertEqual(len(copies), 2)
        self.assertTrue(all(reference in identities for reference in starts + copies))
        self.assertEqual(set(removals), identities)

    def test_direct_script_help_resolves_local_debug_trace_import(self):
        completed = subprocess.run(
            [str(Path(".venv/bin/python")), str(Path(subject.__file__)), "--help"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_main_argument_failure_is_one_sanitized_json_line(self):
        with patch("sys.stdout", new_callable=StringIO) as output:
            status = subject.main(["--image", "image", "--bundle", "/missing",
                                   "--gpu-uuid", "not-gpu", "--ollama-version", "v",
                                   "--output", "/tmp/out"])
        self.assertEqual(status, 2)
        value = json.loads(output.getvalue())
        self.assertEqual(value["status"], "incomplete")
        self.assertNotIn("/missing", output.getvalue())
    def test_create_constructs_network_gpu_pid_and_readonly_mounts(self):
        with patch.object(subject, "run") as runner:
            subject._create("c", "image", [("v", "/opt/measurement", True)],
                            ["--bundle", "/opt/measurement"], gpu="GPU-1", pid=True)
        command = runner.call_args.args[0]
        self.assertEqual(command[:5], ["docker", "create", "--name", "c", "--network"])
        self.assertIn("none", command)
        self.assertIn("--pid=host", command)
        self.assertIn("device=GPU-1", command)
        self.assertIn("type=volume,src=v,dst=/opt/measurement,readonly", command)
        self.assertEqual(command[-3:], ["image", "--bundle", "/opt/measurement"])

    def test_result_rejects_missing_malformed_and_partial_matrix(self):
        for value in ("", "not-json", json.dumps({"status": "incomplete"}),
                      json.dumps({"status": "complete", "matrix": [], "profiles": []})):
            with self.subTest(value=value), self.assertRaises((ValueError, json.JSONDecodeError)):
                subject._result(value)

    def test_result_rejects_duplicate_missing_and_wrong_identity_summaries(self):
        matrix = [[model.value, selector] for model, selector in subject.measurement_matrix()]
        valid = {"status": "complete", "matrix": matrix,
                 "profiles": [{"model": model, "profile_identity": model + "-identity"}
                              for model, _ in matrix]}
        for profiles in (valid["profiles"][:2],
                         [*valid["profiles"][:2], valid["profiles"][0]],
                         [{**item, "model": "wrong"} for item in valid["profiles"]]):
            with self.subTest(profiles=profiles), self.assertRaises(ValueError):
                subject._result(json.dumps({**valid, "profiles": profiles}))

    def test_two_preflights_are_required_before_measurement(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        output = root / "profiles.sqlite"
        args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                                  ollama_version="0.11.6", ceiling=2, output=output)
        calls = []
        probes = 0

        def runner(command, *, timeout=60, check=True):
            nonlocal probes
            calls.append(command)
            if command[:3] == ["docker", "run", "--rm"]:
                probes += 1
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-verify"):
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        # The runner seam keeps Docker entirely out of the test.
        with patch.object(subject, "run", side_effect=self._owned_runner(runner)):
            with self.assertRaises(RuntimeError):
                subject.orchestrate(args)
        self.assertEqual(probes, 2)
        measure_index = next(i for i, call in enumerate(calls)
                             if call[:3] == ["docker", "start", "-a"] and call[-1].endswith("-measure"))
        self.assertTrue(all(i < measure_index for i, call in enumerate(calls)
                            if call[:3] == ["docker", "run", "--rm"]))

    def test_failure_never_exports_and_owned_state_is_cleaned(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                                  ollama_version="0.11.6", ceiling=2,
                                  output=root / "profiles.sqlite")
        calls = []

        def runner(command, *, timeout=60, check=True):
            calls.append(command)
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-verify"):
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(subject, "run", side_effect=self._owned_runner(runner)), self.assertRaises(RuntimeError):
            subject.orchestrate(args)
        self.assertFalse(args.output.exists())
        self.assertFalse(any(call[-1].endswith("-export") for call in calls
                             if call[:3] == ["docker", "start", "-a"]))
        self.assertTrue(any(call[:3] == ["docker", "rm", "-f"] for call in calls))
        self.assertTrue(any(call[:3] == ["docker", "volume", "rm"] for call in calls))

    def test_primary_failure_proves_owned_docker_cleanup(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        output = root / "profiles.sqlite"
        args = ["--image", "image", "--bundle", str(bundle), "--gpu-uuid", "GPU-1",
                "--ollama-version", "0.11.6", "--ceiling", "2", "--output", str(output)]

        def runner(command, *, timeout=60, check=True):
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(subject, "run", side_effect=self._owned_runner(runner)), patch("sys.stdout", new_callable=StringIO) as output_stream:
            self.assertEqual(subject.main(args), 2)
        self.assertEqual(json.loads(output_stream.getvalue())["docker_cleanup"], "proved")

    def test_failed_owned_cleanup_remains_unproved(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                                  ollama_version="0.11.6", ceiling=1,
                                  output=root / "profiles.sqlite")

        def runner(command, *, timeout=60, check=True):
            if command[:3] == ["docker", "volume", "rm"]:
                return subprocess.CompletedProcess(command, 1, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(subject, "run", side_effect=self._owned_runner(runner)), self.assertRaises(subject.OperatorFailure) as raised:
            subject.orchestrate(args)
        self.assertEqual(raised.exception.code, "cleanup_failed")

    def test_successful_export_is_not_committed_when_container_or_volume_cleanup_fails(self):
        for failure in ("container", "volume"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "bundle"
                bundle.mkdir()
                output = root / "profiles.sqlite"
                args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                    ollama_version="0.11.6", ceiling=2, output=output)
                result = self._complete_matrix_result()
                commands = []

                def runner(command, *, timeout=60, check=True):
                    commands.append(command)
                    if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                        Path(command[-1]).write_bytes(b"audited sqlite")
                    if failure == "container" and command[:3] == ["docker", "rm", "-f"]:
                        return subprocess.CompletedProcess(command, 1, "", "")
                    if failure == "volume" and command[:3] == ["docker", "volume", "rm"]:
                        return subprocess.CompletedProcess(command, 1, "", "")
                    return subprocess.CompletedProcess(command, 0, "", "")

                with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                     patch.object(subject, "_read_runtime_result", return_value=result), \
                     patch.object(subject, "_audit_export"), \
                     self.assertRaises(subject.OperatorFailure) as raised:
                    subject.orchestrate(args)
                self.assertEqual(raised.exception.code, "cleanup_failed")
                self.assertFalse(output.exists())
                created = {command[3] for command in commands
                           if command[:3] == ["docker", "create", "--name"]}
                removed = {command[-1] for command in commands
                           if command[:3] == ["docker", "rm", "-f"]}
                self.assertEqual(removed, created)

    def test_changed_owner_lock_prevents_export_commit_and_preserves_foreign_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            output = root / "profiles.sqlite"
            args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                ollama_version="0.11.6", ceiling=2, output=output)
            result = self._complete_matrix_result()
            lock = output.with_name("." + output.name + ".measurement.lock")
            replaced = False

            def runner(command, *, timeout=60, check=True):
                nonlocal replaced
                if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                    Path(command[-1]).write_bytes(b"audited sqlite")
                if not replaced and command[:3] == ["docker", "rm", "-f"]:
                    replaced = True
                    lock.unlink()
                    lock.write_bytes(b"foreign owner")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject, "_read_runtime_result", return_value=result), \
                 patch.object(subject, "_audit_export"), \
                 self.assertRaises(subject.OperatorFailure) as raised:
                subject.orchestrate(args)
            self.assertEqual(raised.exception.code, "lock_ownership_changed")
            self.assertFalse(output.exists())
            self.assertEqual(lock.read_bytes(), b"foreign owner")

    def test_successful_export_commits_only_after_owned_docker_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, response, commands = self._run_successful_export(root)
            self.assertEqual(response["status"], "complete")
            self.assertTrue(output.is_file())
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)
            self.assertEqual(list(root.glob(".profiles.sqlite.stage-*")), [])
            self.assertFalse(root.joinpath(".profiles.sqlite.measurement.lock").exists())
            cleanups = [(command, existed) for command, existed in commands
                        if command[:3] in (["docker", "rm", "-f"],
                                          ["docker", "volume", "rm"])]
            self.assertTrue(cleanups)
            self.assertTrue(all(not existed for _, existed in cleanups))
            removed_names = [command[-1] for command, _ in cleanups
                             if command[:3] == ["docker", "rm", "-f"]]
            self.assertFalse(any(name.endswith("-trace") for name in removed_names))

    def test_debug_trace_container_is_owned_and_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            output = root / "profiles.sqlite"
            args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                ollama_version="0.11.6", ceiling=2, output=output, debug=True)
            result = self._complete_matrix_result()
            commands = []

            def runner(command, *, timeout=60, check=True):
                commands.append(command)
                if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                    Path(command[-1]).write_bytes(b"audited sqlite")
                if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-trace"):
                    return subprocess.CompletedProcess(command, 0,
                        '{"schema":"llm.debug-trace.v1","sequence":1,"component":"unknown","event":"wave","state":"failure"}\n', "")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject, "_read_runtime_result", return_value=result), \
                 patch("sys.stderr", new_callable=StringIO), \
                 patch.object(subject, "_audit_export"):
                response = subject.orchestrate(args)
            self.assertEqual(response["status"], "complete")
            trace_create = next(command for command in commands
                                if command[:3] == ["docker", "create", "--name"]
                                and command[3].endswith("-trace"))
            self.assertIn("--label", trace_create)
            self.assertTrue(any(command[:3] == ["docker", "rm", "-f"]
                                and command[-1] == trace_create[3] for command in commands))

    def test_early_transfer_and_preflight_failures_clean_only_created_resources(self):
        for failure_stage in ("transfer", "preflight"):
            with self.subTest(stage=failure_stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "bundle"
                bundle.mkdir()
                output = root / "profiles.sqlite"
                args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                    ollama_version="0.11.6", ceiling=2, output=output)
                commands = []

                def runner(command, *, timeout=60, check=True):
                    commands.append(command)
                    if failure_stage == "transfer" and command[:2] == ["docker", "cp"] \
                            and "/transfer" in command[3]:
                        if check:
                            raise RuntimeError("transfer failed")
                        return subprocess.CompletedProcess(command, 1, "", "")
                    if failure_stage == "preflight" and command[:3] == ["docker", "run", "--rm"]:
                        return subprocess.CompletedProcess(command, 1, "", "")
                    return subprocess.CompletedProcess(command, 0, "", "")

                with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                     self.assertRaises(subject.OperatorFailure) as raised:
                    subject.orchestrate(args)
                expected = "bundle_transfer_failed" if failure_stage == "transfer" else "preflight_failed"
                self.assertEqual(raised.exception.code, expected)
                removed_containers = [command[-1] for command in commands
                                      if command[:3] == ["docker", "rm", "-f"]]
                created_containers = [command[3] for command in commands
                                      if command[:3] == ["docker", "create", "--name"]]
                self.assertEqual(set(removed_containers), set(created_containers))
                self.assertTrue(all(not name.endswith(("-result", "-trace", "-export"))
                                    for name in removed_containers))
                self.assertFalse(output.exists())

    def test_partial_volume_create_failure_checks_uncertain_owner_before_cleanup(self):
        for owns_ambiguous_volume in (True, False):
            with self.subTest(owned=owns_ambiguous_volume), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "bundle"
                bundle.mkdir()
                output = root / "profiles.sqlite"
                args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                    ollama_version="0.11.6", ceiling=2, output=output)
                calls = []
                attempted_owner = None
                attempted_volume = None
                creates = 0

                def runner(command, *, timeout=60, check=True):
                    nonlocal attempted_owner, attempted_volume, creates
                    calls.append(command)
                    if command[:3] == ["docker", "volume", "create"]:
                        creates += 1
                        if creates == 2:
                            attempted_volume = command[-1]
                            attempted_owner = command[command.index("--label") + 1].split("=", 1)[1]
                            if check:
                                raise RuntimeError("ambiguous volume creation")
                            return subprocess.CompletedProcess(command, 1, "", "")
                    if command[:3] == ["docker", "volume", "inspect"] and command[-1] == attempted_volume:
                        owner = attempted_owner if owns_ambiguous_volume else "foreign-owner"
                        return subprocess.CompletedProcess(command, 0,
                            json.dumps({"Name": attempted_volume, "CreatedAt": "initial",
                                        "Labels": {subject.OWNER_LABEL: owner}}), "")
                    return subprocess.CompletedProcess(command, 0, "", "")

                with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                     self.assertRaises(subject.OperatorFailure) as raised:
                    subject.orchestrate(args)
                self.assertEqual(raised.exception.code, "volume_setup_failed")
                removed_volumes = [command[-1] for command in calls
                                   if command[:3] == ["docker", "volume", "rm"]]
                expected_removed = [command[-1] for command in calls
                                    if command[:3] == ["docker", "volume", "create"]][:1]
                if owns_ambiguous_volume:
                    expected_removed.append(attempted_volume)
                self.assertEqual(set(removed_volumes), set(expected_removed))
                self.assertNotIn(attempted_volume, removed_volumes if not owns_ambiguous_volume else [])
                self.assertEqual(creates, 2)
                self.assertFalse(output.exists())

    def test_export_audit_failure_leaves_destination_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            output = root / "profiles.sqlite"
            args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                ollama_version="0.11.6", ceiling=2, output=output)
            result = self._complete_matrix_result()

            def runner(command, *, timeout=60, check=True):
                if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                    Path(command[-1]).write_bytes(b"unaudited sqlite")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject, "_read_runtime_result", return_value=result), \
                 patch.object(subject, "_audit_export", side_effect=ValueError("bad audit")), \
                 self.assertRaises(subject.OperatorFailure) as raised:
                subject.orchestrate(args)
            self.assertEqual(raised.exception.code, "export_audit_failed")
            self.assertFalse(output.exists())

    def test_postcommit_lock_release_failure_reports_retained_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            output = root / "profiles.sqlite"
            args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                ollama_version="0.11.6", ceiling=2, output=output)
            result = self._complete_matrix_result()
            original_unlink = Path.unlink

            def unlink(path, *args, **kwargs):
                if path.name == ".profiles.sqlite.measurement.lock":
                    raise PermissionError("simulated lock release failure")
                return original_unlink(path, *args, **kwargs)

            def runner(command, *, timeout=60, check=True):
                if command[:2] == ["docker", "cp"] and "/export/profiles.sqlite" in command[2]:
                    Path(command[-1]).write_bytes(b"audited sqlite")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
                 patch.object(subject, "_read_runtime_result", return_value=result), \
                 patch.object(subject, "_audit_export"), \
                 patch.object(Path, "unlink", unlink), \
                 self.assertRaises(subject.OperatorFailure) as raised:
                subject.orchestrate(args)
            self.assertEqual(raised.exception.code, "cleanup_failed")
            self.assertTrue(raised.exception.db_retained)
            self.assertEqual(raised.exception.docker_cleanup, "proved")
            self.assertTrue(output.is_file())
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)

    def test_nonzero_measurement_reads_result_and_emits_only_safe_inner_fields(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        args = ["--image", "image", "--bundle", str(bundle), "--gpu-uuid", "GPU-1",
                "--ollama-version", "0.11.6", "--ceiling", "2", "--output",
                str(root / "profiles.sqlite")]
        calls = []
        inner = {"stage": "measure", "failure_kind": "provider_failed",
                 "exception_type": "RuntimeError", "reason": "secret reason",
                 "message": "secret message", "path": "/secret/path",
                  "model": "SmolLM", "measurement_failure_code": "oom",
                  "measurement_failure_detail": "failed_output"}

        def runner(command, *, timeout=60, check=True):
            calls.append(command)
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-result"):
                return subprocess.CompletedProcess(command, 0, json.dumps(inner), "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(subject, "run", side_effect=self._owned_runner(runner)), patch("sys.stdout", new_callable=StringIO) as output:
            status = subject.main(args)

        self.assertEqual(status, 2)
        emitted = json.loads(output.getvalue())
        self.assertEqual(emitted["inner_stage"], "measure")
        self.assertEqual(emitted["inner_failure_kind"], "provider_failed")
        self.assertEqual(emitted["inner_exception_type"], "RuntimeError")
        self.assertEqual(emitted["inner_model"], "SmolLM")
        self.assertEqual(emitted["inner_measurement_failure_code"], "oom")
        self.assertEqual(emitted["inner_measurement_failure_detail"], "failed_output")
        self.assertNotIn("reason", emitted)
        self.assertNotIn("message", emitted)
        self.assertNotIn("path", emitted)

    def test_debug_flag_propagates_and_failure_trace_is_retained_before_cleanup(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        output = root / "profiles.sqlite"
        calls = []
        inner = {"status": "incomplete", "stage": "measurement", "failure_kind": "runner_error"}
        trace_text = '{"schema":"llm.debug-trace.v1","sequence":1,"component":"unknown","event":"wave","state":"failure"}\n'

        def runner(command, *, timeout=60, check=True):
            calls.append(command)
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-result"):
                return subprocess.CompletedProcess(command, 0, json.dumps(inner), "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-trace"):
                return subprocess.CompletedProcess(command, 0, trace_text, "")
            return subprocess.CompletedProcess(command, 0, "", "")

        args = ["--image", "image", "--bundle", str(bundle), "--gpu-uuid", "GPU-1",
                "--ollama-version", "0.11.6", "--ceiling", "2", "--output", str(output), "--debug"]
        with patch.object(subject, "run", side_effect=self._owned_runner(runner)), patch("sys.stderr", new_callable=StringIO) as stderr:
            self.assertEqual(subject.main(args), 2)
        measure_create = next(call for call in calls if call[:3] == ["docker", "create", "--name"] and "--debug" in call)
        self.assertIn("--debug", measure_create)
        self.assertLess(next(i for i, call in enumerate(calls) if call[-1].endswith("-trace")),
                        next(i for i, call in enumerate(calls) if call[:3] == ["docker", "volume", "rm"]))
        sidecar = Path(str(output) + ".debug.jsonl")
        self.assertEqual(sidecar.read_text(), trace_text)
        self.assertIn(trace_text.strip(), stderr.getvalue())

    def test_debug_sidecar_never_clobbers_existing_file(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        output = root / "profiles.sqlite"
        sidecar = Path(str(output) + ".debug.jsonl")
        sidecar.write_text("foreign\n")
        args = argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                                  ollama_version="0.11.6", ceiling=1, output=output, debug=True)
        with patch.object(subject, "run", side_effect=RuntimeError("stop")), self.assertRaises(subject.OperatorFailure):
            subject.orchestrate(args)
        self.assertEqual(sidecar.read_text(), "foreign\n")

    def test_sidecar_short_write_is_atomic_and_retryable(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        sidecar = root / "profiles.sqlite.debug.jsonl"
        text = '{"schema":"llm.debug-trace.v1","sequence":1,"component":"unknown","event":"wave","state":"failure"}\n'
        original = subject.os.write
        calls = [0]
        def short(fd, data):
            calls[0] += 1
            if calls[0] == 1:
                return 0
            return original(fd, data)
        with patch.object(subject.os, "write", side_effect=short):
            subject._publish_sidecar(sidecar, text)
        self.assertFalse(sidecar.exists())
        subject._publish_sidecar(sidecar, text)
        self.assertEqual(sidecar.read_text(), text)

    def test_debug_does_not_change_stdout_result_contract(self):
        with patch("sys.stdout", new_callable=StringIO) as stdout, patch("sys.stderr", new_callable=StringIO):
            status = subject.main(["--image", "image", "--bundle", "/missing", "--gpu-uuid", "bad",
                                  "--ollama-version", "v", "--output", "/tmp/out", "--debug"])
        self.assertEqual(status, 2)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "incomplete")
        self.assertNotIn("schema", result)

    def test_trace_is_retrieved_on_result_reader_exception_before_cleanup(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir(); output = root / "profiles.sqlite"
        calls = []
        def runner(command, *, timeout=60, check=True):
            calls.append(command)
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-trace"):
                return subprocess.CompletedProcess(command, 0, "trace\n", "")
            return subprocess.CompletedProcess(command, 0, "", "")
        with patch.object(subject, "run", side_effect=self._owned_runner(runner)), patch.object(subject, "_read_runtime_result", side_effect=RuntimeError("reader")):
            with self.assertRaises(subject.OperatorFailure):
                subject.orchestrate(argparse.Namespace(image="image", bundle=bundle, gpu_uuid="GPU-1",
                    ollama_version="0.11.6", ceiling=1, output=output, debug=True))
        trace_index = next(i for i, call in enumerate(calls) if call[-1].endswith("-trace"))
        cleanup_index = next(i for i, call in enumerate(calls) if call[:3] == ["docker", "volume", "rm"])
        self.assertLess(trace_index, cleanup_index)

    def test_trace_file_rejects_wrong_path_and_symlink(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); db = root / "profiles.sqlite"
            config = SimpleNamespace(profile_db=db)
            with self.assertRaises(ValueError):
                measure_profiles._trace_file(root / "other.jsonl", config)
            link = root / "debug-trace.jsonl"; target = root / "target"; target.write_text("")
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                measure_profiles._trace_file(link, config)

    def test_unknown_inner_measurement_failure_detail_is_omitted(self):
        result = subject._failure("measurement_failed", "measurement", inner={
            "measurement_failure_detail": "provider-secret"})
        self.assertNotIn("inner_measurement_failure_detail", result)

    def test_smollm_stage_category_is_relayed_from_measurement_failure_detail(self):
        result = subject._failure("measurement_failed", "measurement", inner={
            "measurement_failure_detail": "smollm_input_validation"})
        self.assertEqual(result["inner_measurement_failure_detail"], "smollm_input_validation")
        self.assertEqual(result["inner_failure_code"], "smollm_input_validation")

    def test_persistence_failure_code_is_transportable_without_detail(self):
        result = subject._failure("measurement_failed", "measurement", inner={
            "model": "GECToR",
            "measurement_failure_code": "persistence_validation_failed",
            "measurement_failure_detail": "/private/prompt=secret",
        })
        self.assertEqual(result["inner_model"], "GECToR")
        self.assertEqual(result["inner_measurement_failure_code"],
                         "persistence_validation_failed")
        self.assertNotIn("inner_measurement_failure_detail", result)
        self.assertNotIn("secret", json.dumps(result))

    def test_maximum_witness_detail_is_relayed_from_closed_vocabulary(self):
        result = subject._failure("measurement_failed", "measurement", inner={
            "model": "CoEdIT",
            "measurement_failure_code": "maximum_witness_failed",
            "measurement_failure_detail": "coedit_token_count",
        })
        self.assertEqual(result["inner_model"], "CoEdIT")
        self.assertEqual(result["inner_measurement_failure_code"],
                         "maximum_witness_failed")
        self.assertEqual(result["inner_measurement_failure_detail"],
                         "coedit_token_count")

    def test_outer_relays_every_closed_measurement_failure_detail_and_runtime_code(self):
        for detail in subject.MEASUREMENT_FAILURE_DETAILS:
            with self.subTest(detail=detail):
                result = subject._failure("measurement_failed", "measurement", inner={
                    "measurement_failure_detail": detail})
                self.assertEqual(result["inner_measurement_failure_detail"], detail)
        forwarded = subject._failure("measurement_failed", "measurement", inner={
            "measurement_failure_code": "runtime_generated_request_failed",
            "measurement_failure_detail": "coedit_generated_schema"})
        self.assertEqual(forwarded["inner_measurement_failure_code"],
                         "runtime_generated_request_failed")
        self.assertEqual(forwarded["inner_measurement_failure_detail"],
                         "coedit_generated_schema")
        rejected = subject._failure("measurement_failed", "measurement", inner={
            "measurement_failure_detail": "prompt=private"})
        self.assertNotIn("inner_measurement_failure_detail", rejected)

    def test_outer_failure_envelope_surfaces_only_safe_fields(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        bundle = root / "bundle"; bundle.mkdir()
        args = ["--image", "image", "--bundle", str(bundle), "--gpu-uuid", "GPU-1",
                "--ollama-version", "0.11.6", "--output", str(root / "profiles.sqlite")]
        calls = []
        inner = {"stage": "measure", "failure_kind": "provider_failed",
                 "exception_type": "RuntimeError", "reason": "/private/secret",
                 "message": "token=secret", "path": "/private/path"}

        def runner(command, *, timeout=60, check=True):
            calls.append(command)
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-measure"):
                return subprocess.CompletedProcess(command, 1, "", "")
            if command[:3] == ["docker", "start", "-a"] and command[-1].endswith("-result"):
                return subprocess.CompletedProcess(command, 0, json.dumps(inner), "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(subject, "run", side_effect=self._owned_runner(runner)), \
             patch("sys.stdout", new_callable=StringIO) as output:
            status = subject.main(args)

        self.assertEqual(status, 2)
        emitted = json.loads(output.getvalue())
        self.assertEqual(set(emitted), {"status", "failure_code", "stage", "db_retained",
                                        "docker_cleanup", "inner_stage", "inner_failure_kind",
                                        "inner_exception_type"})
        self.assertEqual(emitted["inner_stage"], "measure")
        self.assertEqual(emitted["inner_failure_kind"], "provider_failed")
        self.assertEqual(emitted["inner_exception_type"], "RuntimeError")
        self.assertNotIn("private", output.getvalue())
        self.assertNotIn("secret", output.getvalue())

    def test_output_reservation_is_no_clobber(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        destination = root / "profiles.sqlite"
        destination.write_bytes(b"foreign")
        args = argparse.Namespace(image="image", bundle=root, gpu_uuid="GPU-1",
                                  ollama_version="0.11.6", ceiling=1, output=destination)
        with patch.object(subject, "run", side_effect=RuntimeError("stop")), self.assertRaises(subject.OperatorFailure) as failure:
            subject.orchestrate(args)
        self.assertEqual(failure.exception.code, "output_reserved")
        self.assertEqual(destination.read_bytes(), b"foreign")

    def test_export_audit_rejects_wal_sidecar_and_tampered_profile_store(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        exported = root / "profiles.sqlite"
        exported.write_bytes(b"sqlite")
        exported.with_name(exported.name + "-wal").write_bytes(b"tampered")
        result = {"profiles": [{"model": "SmolLM", "profile_identity": "a"},
                                {"model": "CoEdIT", "profile_identity": "b"},
                                {"model": "GECToR", "profile_identity": "c"}]}
        with self.assertRaises(ValueError):
            subject._audit_export(exported, result)


if __name__ == "__main__":
    unittest.main()
