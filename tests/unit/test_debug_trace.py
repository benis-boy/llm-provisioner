import json
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from tools.compatibility import debug_trace


class DebugTraceTests(unittest.TestCase):
    def test_disabled_is_silent(self):
        with patch("sys.stderr", new_callable=BytesIO) as stream:
            debug_trace.configure(False).record("x", "event", "success", secret="/private/path")
        self.assertEqual(stream.getvalue(), b"")

    def test_enabled_is_jsonl_and_sanitized(self):
        with patch("sys.stderr", new_callable=BytesIO) as stream:
            debug_trace.configure(True).record("x", "event", "success", hostile="/private/token", safe=3)
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["safe"], 3)
        self.assertNotIn("hostile", rows[0])
        self.assertEqual(rows[0]["schema"], debug_trace.SCHEMA)

    def test_allowed_fields_reject_unsafe_strings_and_preserve_callsite_fields(self):
        with patch("sys.stderr", new_callable=BytesIO) as stream:
            debug_trace.configure(True).record(
                "matrix", "wave", "failure", model="SmolLM", selector="token/prompt",
                wave_type="steady", elapsed_ms=2.5, event_count=3,
                failure_detail="provider_failed")
        row = json.loads(stream.getvalue())
        self.assertNotIn("selector", row)
        self.assertEqual(row["wave_type"], "steady")
        self.assertEqual(row["failure_detail"], "provider_failed")
        self.assertEqual(row["event_count"], 3)

    def test_retrieved_valid_minimal_trace_is_retained(self):
        text = '{"schema":"llm.debug-trace.v1","sequence":1,"component":"unknown","event":"wave","state":"failure"}\n'
        self.assertEqual(debug_trace.sanitize_jsonl(text), text)

    def test_retrieved_malformed_and_oversized_lines_are_bounded(self):
        text = "not-json\n" + json.dumps({"schema": debug_trace.SCHEMA,
            "sequence": 1, "component": "inner", "event": "wave", "state": "failure",
            "model": "x" * 1000}) + "\n"
        result = debug_trace.sanitize_jsonl(text)
        self.assertLessEqual(len(result.encode()), debug_trace.MAX_BYTES)
        for line in result.splitlines():
            self.assertLessEqual(len(line.encode()) + 1, debug_trace.MAX_RECORD_BYTES)
            json.loads(line)

    def test_record_and_total_bounds_emit_truncation(self):
        with patch("sys.stderr", new_callable=BytesIO) as stream:
            tracer = debug_trace.configure(True)
            for _ in range(debug_trace.MAX_RECORDS + 100):
                tracer.record("component", "event", "success", value=1)
        lines = stream.getvalue().splitlines()
        self.assertLessEqual(len(lines), debug_trace.MAX_RECORDS)
        self.assertLessEqual(len(stream.getvalue()), debug_trace.MAX_BYTES)
        self.assertTrue(any(json.loads(line).get("event") == "truncated" for line in lines))

    def test_trace_file_is_optional_runtime_transport(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stderr", new_callable=BytesIO):
            path = Path(directory) / "debug-trace.jsonl"
            debug_trace.configure(True, path).record("x", "event", "success", count=2)
            self.assertEqual(json.loads(path.read_text())["count"], 2)

    def test_trace_file_is_fresh_and_bounded_for_current_run(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stderr", new_callable=BytesIO):
            path = Path(directory) / "trace.jsonl"
            path.write_text("stale\n")
            tracer = debug_trace.configure(True, path)
            tracer.record("x", "event", "success")
            self.assertEqual(path.read_text(), "stale\n")
            self.assertIsNone(tracer._fd)

    def test_text_only_stderr_handles_truncation_marker(self):
        from io import StringIO
        with patch("sys.stderr", new_callable=StringIO) as stream:
            tracer = debug_trace.configure(True)
            for _ in range(debug_trace.MAX_RECORDS + 1):
                tracer.record("x", "event", "success")
        self.assertIn('"event":"truncated"', stream.getvalue())

    def test_digest_ids_is_canonical_and_safe(self):
        self.assertEqual(debug_trace.digest_ids(["b", "a"]), debug_trace.digest_ids(["a", "b"]))
        self.assertIsNone(debug_trace.digest_ids("payload"))

    def test_payload_byte_count_is_retained_but_payload_content_is_not(self):
        with patch("sys.stderr", new_callable=BytesIO) as stream:
            debug_trace.configure(True).record(
                "resource_manager", "admission", "enter",
                payload_bytes=123, payload="secret", selector="prompt-secret")
        row = json.loads(stream.getvalue())
        self.assertEqual(row["payload_bytes"], 123)
        self.assertNotIn("payload", row)
        self.assertNotIn("selector", row)

    def test_labels_use_closed_vocabularies(self):
        with patch("sys.stderr", new_callable=BytesIO) as stream:
            debug_trace.configure(True).record("request-abc123", "opaque-secret", "token-value")
        row = json.loads(stream.getvalue())
        self.assertEqual((row["component"], row["event"], row["state"]),
                         ("unknown", "unknown", "unknown"))

    def test_concurrent_records_are_emitted_in_sequence_order(self):
        with patch("sys.stderr", new_callable=BytesIO) as stream:
            tracer = debug_trace.configure(True)
            threads = [threading.Thread(target=tracer.record,
                                        args=("measurement", "wave", "success"),
                                        kwargs={"wave": index})
                       for index in range(40)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([row["sequence"] for row in rows], list(range(1, 41)))

    def test_lifecycle_decorator_emits_success_and_failure(self):
        events = []
        with patch.object(debug_trace, "record", side_effect=lambda *args, **kwargs: events.append((args, kwargs))):
            @debug_trace.lifecycle("component")
            async def successful():
                return 1

            @debug_trace.lifecycle("component")
            async def failing():
                raise ValueError("not emitted")

            import asyncio
            self.assertEqual(asyncio.run(successful()), 1)
            with self.assertRaises(ValueError):
                asyncio.run(failing())
        self.assertEqual([item[0][2] for item in events], ["enter", "success", "enter", "failure"])

    def test_failures_only_lifecycle_omits_hot_success_records(self):
        events = []
        with patch.object(debug_trace, "record", side_effect=lambda *args, **kwargs: events.append(args)):
            @debug_trace.lifecycle("component", failures_only=True)
            async def successful():
                return 1

            @debug_trace.lifecycle("component", failures_only=True)
            async def failing():
                raise ValueError("not emitted")

            import asyncio
            self.assertEqual(asyncio.run(successful()), 1)
            with self.assertRaises(ValueError): asyncio.run(failing())
        self.assertEqual([item[2] for item in events], ["failure"])

    def test_event_family_contract_covers_matrix_lifecycle_boundaries(self):
        events = []
        with patch.object(debug_trace, "record", side_effect=lambda component, event, state, **fields:
                          events.append((component, event, state))):
            for component, event in (
                ("provider.smollm", "validate"), ("provider.coedit", "load"),
                ("provider.gector", "ready"), ("resource_manager", "admission"),
                ("resource_manager", "execute"), ("resource_manager", "cleanup"),
                ("matrix", "selector"), ("persistence", "validation"),
                ("profile_store", "audit"), ("supervisor", "output")):
                debug_trace.record(component, event, "success", count=1)
        self.assertEqual({(component, event) for component, event, _ in events},
                         {("provider.smollm", "validate"), ("provider.coedit", "load"),
                          ("provider.gector", "ready"), ("resource_manager", "admission"),
                          ("resource_manager", "execute"), ("resource_manager", "cleanup"),
                          ("matrix", "selector"), ("persistence", "validation"),
                          ("profile_store", "audit"), ("supervisor", "output")})


if __name__ == "__main__":
    unittest.main()
