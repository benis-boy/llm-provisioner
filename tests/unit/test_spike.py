import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


SPIKE = Path(__file__).parents[2] / "tools" / "compatibility" / "spike.py"
# The container image places the copied helper beside spike.py as ``artifacts``;
# make the repository-side harness provide the equivalent import.
sys.path.insert(0, str(SPIKE.parents[2]))
sys.path.insert(0, str(SPIKE.parent))
from services.llm.provisioning import artifacts
sys.modules.setdefault("artifacts", artifacts)
SPEC = importlib.util.spec_from_file_location("compatibility_spike", SPIKE)
spike = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spike)


class SpikeHelpersTests(unittest.TestCase):
    def test_text_accepts_pynvml_bytes_and_strings(self):
        self.assertEqual(spike._text(b"GPU-uuid"), "GPU-uuid")
        self.assertEqual(spike._text("GPU-uuid"), "GPU-uuid")

    def test_cuda_bare_uuid_and_nvml_gpu_prefix_identify_same_device(self):
        bare = "12345678-1234-5678-9abc-def012345678"
        self.assertTrue(spike._same_physical_uuid(bare, f"GPU-{bare}", uuid_bytes := bytes.fromhex(bare.replace("-", ""))))
        self.assertEqual(spike._physical_uuid(uuid_bytes), bare)

    def test_uuid_rejects_true_mismatch_malformed_and_mig_identifiers(self):
        self.assertFalse(spike._same_physical_uuid("GPU-12345678-1234-5678-9abc-def012345678", "GPU-aaaaaaaa-1234-5678-9abc-def012345678"))
        self.assertFalse(spike._same_physical_uuid("not-a-uuid", "GPU-12345678-1234-5678-9abc-def012345678"))
        self.assertFalse(spike._same_physical_uuid("MIG-GPU-12345678-1234-5678-9abc-def012345678/1/0", "GPU-12345678-1234-5678-9abc-def012345678"))

    def test_request_uses_get_for_health_and_post_for_generation(self):
        response = Mock()
        response.read.return_value = b'{"version":"x"}'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(spike, "urlopen", return_value=response) as urlopen:
            spike._request("/api/version")
            self.assertEqual(urlopen.call_args.args[0].get_method(), "GET")
            spike._request("/api/generate", {"model": "m"})
            self.assertEqual(urlopen.call_args.args[0].get_method(), "POST")

    def test_terminate_group_escalates_after_timeout(self):
        process = Mock(pid=42)
        process.wait.side_effect = [__import__("subprocess").TimeoutExpired("x", 1), None]
        with patch.object(spike.os, "killpg") as killpg:
            spike._terminate_group(process)
        self.assertEqual(killpg.call_args_list[0].args, (42, spike.signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[1].args, (42, spike.signal.SIGKILL))
