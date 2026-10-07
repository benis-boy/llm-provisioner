import copy
import errno
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from services.llm.provisioning.artifacts import SPECS
from tools import download_models as downloader


def pinned(content: bytes) -> dict[str, object]:
    return {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}


class BrokenResponse:
    def __init__(self):
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size=-1):
        self.reads += 1
        if self.reads == 1:
            return b"partial"
        raise OSError("connection interrupted: signed-token-secret")


class HeadResponse:
    def __init__(self, size):
        self.headers = {"Content-Length": str(size)}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class DownloadModelsTests(unittest.TestCase):
    def test_catalog_selection_matches_artifact_specs(self):
        catalog = downloader.load_catalog()
        self.assertEqual(set(catalog), set(SPECS))
        self.assertEqual(
            {model: {item["path"] for item in entry["files"]} for model, entry in catalog.items()},
            {model: set(files) for model, files in SPECS.items()},
        )
        revisions = {
            "SmolLM": ("unsloth/SmolLM2-1.7B-Instruct-GGUF", "e933f1cdf73cc87cb67915bf5dd6ea81d36080ca"),
            "CoEdIT": ("grammarly/coedit-large", "5637bcdf9d8d4419f97c8cfea36f7d35c79232b6"),
            "GECToR": ("gotutiyan/gector-deberta-large-5k", "5fa80d75504eaf7c867a0d4c5a26752df6585aa1"),
        }
        for model, expected in revisions.items():
            self.assertEqual((catalog[model]["repo"], catalog[model]["revision"]), expected)
        root = Path(__file__).resolve().parents[2]
        selected_manifest = json.loads((root / ".compatibility/context/manifest.json").read_text())
        for model, entry in catalog.items():
            catalog_files = {
                item["path"]: {key: item[key] for key in ("path", "size", "sha256")}
                for item in entry["files"]
            }
            source_files = {
                item["path"]: {key: item[key] for key in ("path", "size", "sha256")}
                for item in selected_manifest["models"][model]["files"]
            }
            self.assertEqual(catalog_files, source_files)
        self.assertEqual(downloader.MODELFILE_CONTENT,
                         b"FROM ./SmolLM2-1.7B-Instruct-Q8_0.gguf\n")
        modelfile = next(item for item in catalog["SmolLM"]["files"]
                         if item["path"] == "Modelfile")
        self.assertEqual(modelfile["size"], len(downloader.MODELFILE_CONTENT))
        self.assertEqual(modelfile["sha256"], hashlib.sha256(downloader.MODELFILE_CONTENT).hexdigest())
        vocabulary = next(item for item in catalog["GECToR"]["files"]
                          if item["path"] == "verb-form-vocab.txt")
        self.assertEqual(vocabulary["source"], {
            "url": "https://raw.githubusercontent.com/grammarly/gector/"
                   "3d41d2841512d2690cffce1b5ac6795fe9a0a5dd/data/verb-form-vocab.txt"
        })

    def test_streamed_download_verifies_digest_and_repeat_skips(self):
        content = b"small model fixture"
        spec = pinned(content)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.bin"
            with patch.object(downloader, "urlopen", return_value=io.BytesIO(content)) as open_url:
                self.assertTrue(downloader.download_file(target, spec, url="https://example.invalid/model"))
            self.assertEqual(target.read_bytes(), content)
            with patch.object(downloader, "urlopen") as open_url:
                self.assertFalse(downloader.download_file(target, spec, url="https://example.invalid/model"))
                open_url.assert_not_called()
            self.assertEqual(list(Path(directory).glob(".*.part")), [])

    def test_bad_hash_and_size_leave_no_partial_destination_or_temp(self):
        content = b"incorrect fixture"
        expected = {"size": len(content), "sha256": hashlib.sha256(b"different").hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.bin"
            with patch.object(downloader, "urlopen", return_value=io.BytesIO(content)):
                with self.assertRaisesRegex(downloader.DownloadError, "pinned artifact mismatch"):
                    downloader.download_file(target, expected, url="https://example.invalid/model")
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).glob(".*.part")), [])

            too_small = {**pinned(content), "size": len(content) - 1}
            with patch.object(downloader, "urlopen", return_value=io.BytesIO(content)):
                with self.assertRaisesRegex(downloader.DownloadError, "exceeded pinned size"):
                    downloader.download_file(target, too_small, url="https://example.invalid/model")
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).glob(".*.part")), [])

    def test_interrupted_stream_cleans_private_temporary_file(self):
        content = b"partial"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.bin"
            with patch.object(downloader, "urlopen", return_value=BrokenResponse()):
                with self.assertRaisesRegex(downloader.DownloadError, "download failed") as caught:
                    downloader.download_file(target, pinned(b"complete"), url="https://example.invalid/model")
                self.assertNotIn("signed-token-secret", str(caught.exception))
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).glob(".*.part")), [])

    def test_existing_conflict_needs_replace_opt_in(self):
        content = b"pinned content"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.bin"
            target.write_bytes(b"operator data")
            with patch.object(downloader, "urlopen") as open_url:
                with self.assertRaisesRegex(downloader.DownloadError, "use --replace"):
                    downloader.download_file(target, pinned(content), url="https://example.invalid/model")
                open_url.assert_not_called()
            self.assertEqual(target.read_bytes(), b"operator data")

            with patch.object(downloader, "urlopen", return_value=io.BytesIO(content)):
                self.assertTrue(downloader.download_file(
                    target, pinned(content), url="https://example.invalid/model", replace=True,
                ))
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(list(Path(directory).glob(".*.part")), [])

    def test_failed_replace_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.bin"
            original = b"old operator file"
            target.write_bytes(original)
            with patch.object(downloader, "urlopen", return_value=BrokenResponse()):
                with self.assertRaises(downloader.DownloadError):
                    downloader.download_file(
                        target, pinned(b"new pinned bytes"), url="https://example.invalid/model", replace=True,
                    )
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob(".*.part")), [])

    def test_unsupported_hardlink_uses_atomic_rename_fallback(self):
        content = b"portable commit"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.bin"
            unsupported = OSError(errno.EOPNOTSUPP, "hard links unavailable")
            with patch.object(downloader, "urlopen", return_value=io.BytesIO(content)):
                with patch.object(downloader.os, "link", side_effect=unsupported):
                    self.assertTrue(downloader.download_file(
                        target, pinned(content), url="https://example.invalid/model",
                    ))
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(list(Path(directory).glob(".*.part")), [])

    def test_no_replace_fallback_preserves_destination_that_appears_at_commit(self):
        content = b"verified candidate"
        for appeared in (b"unrelated operator file", content):
            with self.subTest(appeared=appeared), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "model.bin"

                def appear_then_fail_noreplace(_source, destination):
                    Path(destination).write_bytes(appeared)
                    raise FileExistsError(errno.EEXIST, "target appeared", str(destination))

                with patch.object(downloader, "urlopen", return_value=io.BytesIO(content)):
                    with patch.object(downloader.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "no links")):
                        with patch.object(downloader, "_rename_noreplace",
                                          side_effect=appear_then_fail_noreplace) as rename:
                            if appeared == content:
                                self.assertFalse(downloader.download_file(
                                    target, pinned(content), url="https://example.invalid/model",
                                ))
                            else:
                                with self.assertRaisesRegex(downloader.DownloadError, "refusing overwrite"):
                                    downloader.download_file(
                                        target, pinned(content), url="https://example.invalid/model",
                                    )
                            rename.assert_called_once()
                self.assertEqual(target.read_bytes(), appeared)
                self.assertEqual(list(Path(directory).glob(".*.part")), [])

    def test_no_replace_fallback_fails_closed_when_atomic_rename_is_unavailable(self):
        content = b"verified candidate"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.bin"
            with patch.object(downloader, "urlopen", return_value=io.BytesIO(content)):
                with patch.object(downloader.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "no links")):
                    with patch.object(downloader, "_rename_noreplace",
                                      side_effect=downloader.AtomicNoReplaceUnsupported("unsupported")):
                        with patch.object(downloader.os, "replace") as replace:
                            with self.assertRaisesRegex(downloader.DownloadError, "choose an output filesystem"):
                                downloader.download_file(
                                    target, pinned(content), url="https://example.invalid/model",
                                )
                            replace.assert_not_called()
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).glob(".*.part")), [])

    def test_symlink_destination_root_and_file_are_rejected(self):
        content = b"safe bytes"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "linked"
            link.symlink_to(real, target_is_directory=True)
            with patch.object(downloader, "urlopen") as open_url:
                with self.assertRaisesRegex(downloader.DownloadError, "symlink"):
                    downloader.download_file(link / "model.bin", pinned(content), url="https://example.invalid")
                open_url.assert_not_called()

            target = root / "model.bin"
            target.symlink_to(real / "other")
            with patch.object(downloader, "urlopen") as open_url:
                with self.assertRaisesRegex(downloader.DownloadError, "symlink"):
                    downloader.download_file(target, pinned(content), url="https://example.invalid")
                open_url.assert_not_called()

    def test_default_cli_path_is_project_models_without_downloading_in_tests(self):
        expected = Path(downloader.__file__).resolve().parents[1] / "LLMs" / "project-models"
        self.assertEqual(downloader.DEFAULT_OUTPUT, expected)
        with patch.object(downloader, "download_models") as download:
            self.assertEqual(downloader.main([]), 0)
        download.assert_called_once_with(expected, replace=False)

    def test_full_model_flow_routes_vocabulary_and_repeat_is_offline(self):
        catalog = copy.deepcopy(downloader.load_catalog())
        responses = {}
        for model, entry in catalog.items():
            for spec in entry["files"]:
                if model == "SmolLM" and spec["path"] == downloader.MODELFILE_NAME:
                    content = downloader.MODELFILE_CONTENT
                else:
                    content = f"fixture:{model}:{spec['path']}".encode()
                spec["size"] = len(content)
                spec["sha256"] = hashlib.sha256(content).hexdigest()
                responses[downloader._artifact_url(entry, spec)] = content

        def open_fixture(request, timeout):
            self.assertIn(request.full_url, responses)
            if request.get_method() == "HEAD":
                self.assertEqual(timeout, 30)
                return HeadResponse(len(responses[request.full_url]))
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(timeout, 60)
            return io.BytesIO(responses[request.full_url])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "models"
            with patch.object(downloader, "load_catalog", return_value=catalog):
                with patch.object(downloader, "urlopen", side_effect=open_fixture) as open_url:
                    downloader.download_models(output)
                self.assertTrue(any("raw.githubusercontent.com/grammarly/gector/" in call.args[0].full_url
                                    for call in open_url.call_args_list))

                for model, entry in catalog.items():
                    for spec in entry["files"]:
                        self.assertEqual((output / model / spec["path"]).read_bytes(),
                                         responses[downloader._artifact_url(entry, spec)]
                                         if not (model == "SmolLM" and spec["path"] == downloader.MODELFILE_NAME)
                                         else downloader.MODELFILE_CONTENT)

                with patch.object(downloader, "urlopen") as network:
                    downloader.download_models(output)
                    network.assert_not_called()

    def test_head_rejection_stops_before_get_or_file_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "models"
            error = HTTPError("https://example.invalid/model", 404, "missing", None, io.BytesIO())
            with patch.object(downloader, "urlopen", side_effect=error) as network:
                with patch.object(downloader, "download_file") as transfer:
                    with self.assertRaisesRegex(downloader.DownloadError, "returned HTTP 404"):
                        downloader.download_models(output)
                    transfer.assert_not_called()
            self.assertEqual(network.call_count, 1)
            self.assertEqual(network.call_args.args[0].get_method(), "HEAD")

    def test_head_content_length_mismatch_stops_before_get_or_file_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "models"
            with patch.object(downloader, "urlopen", return_value=HeadResponse(1)) as network:
                with patch.object(downloader, "download_file") as transfer:
                    with self.assertRaisesRegex(downloader.DownloadError, "size mismatch"):
                        downloader.download_models(output)
                    transfer.assert_not_called()
            self.assertEqual(network.call_count, 1)
            self.assertEqual(network.call_args.args[0].get_method(), "HEAD")


if __name__ == "__main__":
    unittest.main()
