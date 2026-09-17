import asyncio
import tempfile
import unittest
from pathlib import Path

from services.llm.queue.contracts import FunctionDescriptor, FunctionRegistry, ModelId
from services.llm.queue.eligibility import EligibilityEvaluator
from services.llm.queue.store import EvaluatedClaim, QueueStore, StaleCallback


class EligibilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.queue = QueueStore(Path(self.tmp.name) / "q.sqlite", "s", ModelId.SMOLLM)
        self.session = self.queue.start_session()
        self.registry = FunctionRegistry()

    def tearDown(self):
        self.queue.close(); self.tmp.cleanup()

    def add(self, name, *, ready=None, template=None, deps=()):
        self.queue.enqueue(name, name, deps, idempotency_key=name, ready=ready, template=template)

    async def test_fifo_skips_blocked_and_bounded_slices_yield(self):
        self.add("blocked", ready=FunctionDescriptor("false"))
        self.add("ready", ready=FunctionDescriptor("true"))
        self.registry.register("false", lambda **_: False)
        self.registry.register("true", lambda **_: True)
        evaluator = EligibilityEvaluator(self.queue, self.registry, slice_size=1)
        yielded = False
        original = evaluator.evaluate
        async def wrapped(request_id):
            nonlocal yielded
            yielded = True
            return await original(request_id)
        evaluator.evaluate = wrapped
        result = await evaluator.next_eligible()
        self.assertEqual(result.request_id, "ready")
        self.assertTrue(yielded)

    async def test_sync_async_functions_receive_detached_selected_dependencies(self):
        self.add("dep")
        token = self.queue.claim("dep", self.session.token, self.session.generation)
        self.queue.stage_handoff("dep", token, self.session.token, self.session.generation, "a" * 64, "h")
        self.queue.acknowledge_handoff("dep", "h")
        seen = []
        def sync(args, dependency_results):
            seen.append((args, dependency_results)); return True
        async def template(args, dependency_results): return "b" * 64
        self.registry.register("sync", sync); self.registry.register("template", template)
        self.add("work", ready=FunctionDescriptor("sync", {"nested": {"x": 1}}, ("dep",)), template=FunctionDescriptor("template", {}, ("dep",)), deps=("dep",))
        result = await EligibilityEvaluator(self.queue, self.registry, lambda _: True).next_eligible()
        self.assertTrue(result.eligible); self.assertEqual(seen[0][1], {"dep": "a" * 64})
        self.assertEqual(seen[0][0], {"nested": {"x": 1}})

    async def test_invalidation_and_session_fence_prevent_stale_success_or_failure(self):
        gate = asyncio.Event()
        entered = asyncio.Event()
        async def slow(**_):
            entered.set(); await gate.wait(); return True
        self.registry.register("slow", slow)
        self.add("work", ready=FunctionDescriptor("slow"))
        task = asyncio.create_task(EligibilityEvaluator(self.queue, self.registry).evaluate("work"))
        await entered.wait(); self.queue.recover_session(); gate.set()
        result = await task
        self.assertFalse(result.eligible); self.assertEqual(self.queue.get("work")["status"], "scheduled")

    async def test_session_fence_discards_stale_callback_error(self):
        gate, entered = asyncio.Event(), asyncio.Event()
        async def slow_error(**_):
            entered.set(); await gate.wait(); raise ValueError("late")
        self.registry.register("slow_error", slow_error)
        self.add("work", ready=FunctionDescriptor("slow_error"))
        task = asyncio.create_task(EligibilityEvaluator(self.queue, self.registry).evaluate("work"))
        await entered.wait(); self.queue.recover_session(); gate.set()
        result = await task
        self.assertEqual(result.error_code, "stale_evaluation")
        self.assertEqual(self.queue.get("work")["status"], "scheduled")

    async def test_cancel_and_stop_discard_stale_callbacks(self):
        cancel_gate, cancel_entered = asyncio.Event(), asyncio.Event()
        async def slow_success(**_):
            cancel_entered.set(); await cancel_gate.wait(); return True
        self.registry.register("slow_success", slow_success)
        self.add("cancelled", ready=FunctionDescriptor("slow_success"))
        cancelled = asyncio.create_task(EligibilityEvaluator(self.queue, self.registry).evaluate("cancelled"))
        await cancel_entered.wait(); self.queue.cancel("cancelled"); cancel_gate.set()
        self.assertEqual((await cancelled).error_code, "stale_evaluation")

        stop_gate, stop_entered = asyncio.Event(), asyncio.Event()
        async def slow_failure(**_):
            stop_entered.set(); await stop_gate.wait(); raise ValueError("late")
        self.registry.register("slow_failure", slow_failure)
        self.add("stopped", ready=FunctionDescriptor("slow_failure"))
        stopped = asyncio.create_task(EligibilityEvaluator(self.queue, self.registry).evaluate("stopped"))
        await stop_entered.wait(); self.queue.stop(); stop_gate.set()
        self.assertEqual((await stopped).error_code, "stale_evaluation")

    async def test_capability_is_single_use_and_claim_replays_exact_payload(self):
        self.add("work", template=FunctionDescriptor("render"))
        self.registry.register("render", lambda **_: "payload-output")
        evaluator = EligibilityEvaluator(self.queue, self.registry)
        result = await evaluator.evaluate("work")
        self.assertTrue(result.eligible)
        token = self.queue.claim_evaluated(result.capability)
        self.assertIsNotNone(token)
        self.assertEqual(self.queue.outbox()[0]["payload_reference"], "payload-output")
        with self.assertRaises(StaleCallback): self.queue.claim_evaluated(result.capability)

    async def test_validator_and_unknown_function_are_request_local_errors(self):
        self.add("unknown", ready=FunctionDescriptor("missing"))
        self.add("valid", ready=FunctionDescriptor("yes"))
        self.registry.register("yes", lambda **_: True)
        evaluator = EligibilityEvaluator(self.queue, self.registry)
        self.assertEqual((await evaluator.evaluate("unknown")).error_code, "function_unavailable")
        self.assertTrue((await evaluator.evaluate("valid")).eligible)

    async def test_readiness_requires_exact_boolean(self):
        self.registry.register("truthy", lambda **_: 1)
        self.add("work", ready=FunctionDescriptor("truthy"))
        result = await EligibilityEvaluator(self.queue, self.registry).evaluate("work")
        self.assertEqual(result.error_code, "function_failed")
        self.assertEqual(self.queue.get("work")["status"], "error")

    async def test_false_readiness_leaves_request_scheduled(self):
        self.registry.register("no", lambda **_: False)
        self.add("work", ready=FunctionDescriptor("no"))
        result = await EligibilityEvaluator(self.queue, self.registry).evaluate("work")
        self.assertFalse(result.eligible)
        self.assertEqual(self.queue.get("work")["status"], "scheduled")

    async def test_invalid_template_is_request_local_error(self):
        self.registry.register("empty", lambda **_: "")
        self.add("bad", template=FunctionDescriptor("empty"))
        self.assertEqual((await EligibilityEvaluator(self.queue, self.registry).evaluate("bad")).error_code, "template_invalid")
        self.assertEqual(self.queue.get("bad")["status"], "error")

    async def test_validator_rejection_and_error_are_template_errors(self):
        self.registry.register("render", lambda **_: "payload")
        self.add("rejected", template=FunctionDescriptor("render"))
        self.add("broken", template=FunctionDescriptor("render"))
        rejected = EligibilityEvaluator(self.queue, self.registry, lambda _: False)
        broken = EligibilityEvaluator(self.queue, self.registry, lambda _: (_ for _ in ()).throw(ValueError()))
        self.assertEqual((await rejected.evaluate("rejected")).error_code, "template_invalid")
        self.assertEqual((await broken.evaluate("broken")).error_code, "template_invalid")

    async def test_retry_delayed_work_is_not_evaluated(self):
        called = False
        def ready(**_):
            nonlocal called; called = True; return True
        self.registry.register("ready", ready)
        self.add("work", ready=FunctionDescriptor("ready"))
        self.queue.db.execute("UPDATE requests SET next_attempt_at=? WHERE request_id='work'", (10**12,))
        self.assertFalse((await EligibilityEvaluator(self.queue, self.registry).evaluate("work")).eligible)
        self.assertFalse(called)

    async def test_failed_dependency_terminates_waiting_request(self):
        self.add("dep")
        self.queue.cancel("dep")
        self.add("work", deps=("dep",))
        result = await EligibilityEvaluator(self.queue, self.registry).evaluate("work")
        self.assertEqual(result.error_code, "dependency_failed")
        self.assertEqual(self.queue.get("work")["status"], "error")

    async def test_unacknowledged_done_dependency_remains_blocked(self):
        self.add("dep")
        token = self.queue.claim("dep", self.session.token, self.session.generation)
        self.queue.stage_handoff("dep", token, self.session.token, self.session.generation, "a" * 64, "h")
        self.add("work", deps=("dep",))
        self.assertFalse((await EligibilityEvaluator(self.queue, self.registry).evaluate("work")).eligible)

    async def test_yield_allows_a_peer_task_to_run(self):
        self.registry.register("no", lambda **_: False)
        self.add("first", ready=FunctionDescriptor("no")); self.add("second")
        peer_ran = False
        async def peer():
            nonlocal peer_ran; await asyncio.sleep(0); peer_ran = True
        peer_task = asyncio.create_task(peer())
        result = await EligibilityEvaluator(self.queue, self.registry, slice_size=1).next_eligible()
        await peer_task
        self.assertTrue(peer_ran); self.assertEqual(result.request_id, "second")

    async def test_early_node_change_restarts_scan_during_later_await(self):
        gate, entered = asyncio.Event(), asyncio.Event()
        readiness = {"value": False}
        self.registry.register("no", lambda **_: readiness["value"])
        async def slow(**_):
            entered.set(); await gate.wait(); return True
        self.registry.register("slow", slow)
        self.add("first", ready=FunctionDescriptor("no")); self.add("later", ready=FunctionDescriptor("slow"))
        evaluator = EligibilityEvaluator(self.queue, self.registry, slice_size=2)
        task = asyncio.create_task(evaluator.next_eligible())
        await entered.wait()
        readiness["value"] = True; evaluator.invalidate(); gate.set()
        result = await task
        self.assertEqual(result.request_id, "first")

    async def test_capability_tampering_fails_closed(self):
        self.add("work", template=FunctionDescriptor("render")); self.registry.register("render", lambda **_: "payload")
        capability = (await EligibilityEvaluator(self.queue, self.registry).evaluate("work")).capability
        forged = EvaluatedClaim(capability.request_id, capability.version, capability.session_token, capability.generation, "other", capability.intent_fingerprint, capability.nonce)
        self.assertIsNone(self.queue.claim_evaluated(forged))
        self.assertEqual(self.queue.get("work")["status"], "scheduled")

    async def test_version_change_discards_capability_and_reopen_preserves_claim_payload(self):
        self.add("work", template=FunctionDescriptor("render")); self.registry.register("render", lambda **_: "payload")
        evaluator = EligibilityEvaluator(self.queue, self.registry)
        stale = (await evaluator.evaluate("work")).capability
        self.add("other")
        with self.assertRaises(StaleCallback): self.queue.claim_evaluated(stale)
        capability = (await evaluator.evaluate("work")).capability
        token = self.queue.claim_evaluated(capability)
        database = self.queue.path
        self.queue.close()
        self.queue = QueueStore(database, "s", ModelId.SMOLLM)
        self.queue.recover_session()
        attempt = self.queue.db.execute("SELECT payload_reference FROM attempts WHERE token=?", (token,)).fetchone()
        self.assertEqual(attempt[0], "payload")
        submit = self.queue.db.execute("SELECT payload_reference FROM outbox WHERE token=? AND kind='submit'", (token,)).fetchone()
        self.assertEqual(submit[0], "payload")

    async def test_poll_argument_validation(self):
        with self.assertRaises(ValueError): EligibilityEvaluator(self.queue, slice_size=True)
        with self.assertRaises(ValueError): EligibilityEvaluator(self.queue, poll_interval=0)


if __name__ == "__main__": unittest.main()
