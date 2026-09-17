import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
DOCKERFILE = (ROOT / "deploy/docker/Dockerfile").read_text()
ENTRYPOINT = (ROOT / "deploy/docker/entrypoint.sh").read_text()


class DeploymentImageTests(unittest.TestCase):
    def test_image_is_offline_nonroot_and_keeps_code_root_owned(self):
        self.assertIn("USER llm", DOCKERFILE)
        self.assertIn("--no-index", DOCKERFILE)
        self.assertIn("--require-hashes", DOCKERFILE)
        self.assertIn("chown -R root:root /opt/venv /opt/llm", DOCKERFILE)
        self.assertIn("chown -R llm:llm /var/lib/llm", DOCKERFILE)
        self.assertIn("/var/lib/llm/results", DOCKERFILE)
        self.assertIn("ollama-linux-amd64.tgz", DOCKERFILE)
        self.assertIn("test -x /out/bin/ollama", DOCKERFILE)
        self.assertIn("test -d /out/lib/ollama", DOCKERFILE)

    def test_dockerignore_admits_only_staged_contract_inputs_and_traversable_parents(self):
        dockerignore = (ROOT / "deploy/docker/Dockerfile.dockerignore").read_text()
        self.assertIn("!deploy/", dockerignore)
        self.assertIn("!deploy/docker/", dockerignore)
        self.assertIn("!deploy/docker/.inputs/requirements.lock", dockerignore)
        self.assertIn("!deploy/docker/.inputs/ollama-linux-amd64.tgz", dockerignore)
        self.assertIn("!deploy/docker/.inputs/wheelhouse/*.whl", dockerignore)
        self.assertNotIn("!deploy/docker/.inputs/**", dockerignore)
        self.assertIn("**/__pycache__/", dockerignore)

    def test_entrypoint_dispatches_composed_runtime(self):
        result = subprocess.run([sys.executable, "-m", "services.llm.bootstrap", "--help"],
                                cwd=ROOT,
                                text=True, capture_output=True, check=False)
        self.assertEqual(0, result.returncode)
        self.assertIn("serve", result.stdout)

    def test_image_has_fixed_cross_user_launcher(self):
        self.assertIn("ollama-launcher.c", DOCKERFILE)
        self.assertIn("chmod 4555 /usr/local/bin/llm-ollama-launch", DOCKERFILE)
        self.assertIn("getpwnam(\"llm\")", (ROOT / "deploy/docker/ollama-launcher.c").read_text())
        self.assertIn("getpwnam(\"ollama\")", (ROOT / "deploy/docker/ollama-launcher.c").read_text())
        self.assertIn('"-I"', (ROOT / "deploy/docker/ollama-launcher.c").read_text())
        self.assertIn("clearenv()", (ROOT / "deploy/docker/ollama-launcher.c").read_text())

    def test_compose_has_truthful_isolated_runtime_contract(self):
        compose = (ROOT / "deploy/docker/compose.phase1.yml").read_text()
        self.assertIn("pid: host", compose)
        self.assertIn("network_mode: none", compose)
        self.assertIn("llm-provider-state:/var/lib/llm", compose)
        self.assertIn("llm-provider-ollama:/var/lib/ollama", compose)
        self.assertIn("device_ids:", compose)


if __name__ == "__main__":
    unittest.main()
