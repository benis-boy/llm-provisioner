"""Regression checks for isolated devcontainer OpenCode state."""

import json
from pathlib import Path
import unittest


CONFIG = Path(__file__).parents[2] / ".devcontainer" / "devcontainer.json"


class DevcontainerConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.mounts = cls.config["mounts"]
        cls.post_create = cls.config["postCreateCommand"]

    def test_active_opencode_data_directory_uses_a_read_write_volume(self):
        mount = next(
            mount
            for mount in self.mounts
            if "target=/home/vscode/.local/share/opencode" in mount
        )
        self.assertIn(
            "source=llm-provider-opencode-data,"
            "target=/home/vscode/.local/share/opencode,type=volume",
            mount,
        )
        self.assertNotIn("readonly", mount)

    def test_windows_profile_is_read_only_transfer_source_not_active_state(self):
        data_mount = next(
            mount
            for mount in self.mounts
            if "target=/mnt/host-opencode" in mount
        )
        self.assertIn(
            "source=/mnt/c/Users/benja/.local/share/opencode", data_mount
        )
        self.assertIn("readonly", data_mount)
        self.assertNotIn("target=/home/vscode/.local/share/opencode", data_mount)
        self.assertFalse(
            any("target=/home/vscode/.config/opencode" in mount for mount in self.mounts)
        )

    def test_does_not_mount_legacy_credential_files(self):
        self.assertFalse(any("auth.json" in mount for mount in self.mounts))
        self.assertFalse(any("account.json" in mount for mount in self.mounts))

    def test_preflight_checks_directory_not_auth_file(self):
        self.assertIn("test -d /home/vscode/.local/share/opencode", self.post_create)
        self.assertIn("test -r /home/vscode/.local/share/opencode", self.post_create)
        self.assertIn("test -w /home/vscode/.local/share/opencode", self.post_create)
        self.assertNotIn("test -s", self.post_create)
        self.assertNotIn("auth.json", self.post_create)

    def test_identity_and_required_mounts_remain(self):
        self.assertEqual(self.config["remoteUser"], "vscode")
        self.assertIn("--gpus=all", self.config["runArgs"])
        self.assertIn("--pid=host", self.config["runArgs"])
        self.assertTrue(any("target=/host/proc" in mount for mount in self.mounts))
        self.assertIn("opencode --version", self.post_create)
        self.assertIn("python3 -m venv .venv", self.post_create)

    def test_pinned_cli_version_remains(self):
        dockerfile = CONFIG.parent / "Dockerfile"
        dockerfile_text = dockerfile.read_text()
        self.assertIn("ARG OPENCODE_VERSION=2.0.24", dockerfile_text)
        self.assertIn("/home/vscode/.config/opencode", dockerfile_text)
        self.assertIn("chown -R vscode:vscode", dockerfile_text)


if __name__ == "__main__":
    unittest.main()
