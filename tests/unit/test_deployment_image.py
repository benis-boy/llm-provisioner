import subprocess
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
        self.assertIn("chown -R llm:llm /srv/llm/state /srv/llm/results", DOCKERFILE)
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

    def test_entrypoint_refuses_unimplemented_runtime(self):
        result = subprocess.run(["sh", str(ROOT / "deploy/docker/entrypoint.sh"), "serve"],
                                text=True, capture_output=True, check=False)
        self.assertEqual(78, result.returncode)
        self.assertIn("distinct llm/ollama supervision is not implemented", result.stderr)


if __name__ == "__main__":
    unittest.main()
