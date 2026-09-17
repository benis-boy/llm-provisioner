"""Small Phase 0 wire-contract acceptance checks.

These checks deliberately consume the production serializers rather than
recreating their dictionaries.  They complement the endpoint lifecycle suites
by making the OpenAPI authority fail when a nullable field, error envelope, or
future-operation marker drifts from the implemented boundary.
"""

import copy
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from aiohttp import ClientSession
from aiohttp.test_utils import TestServer
import jsonschema
import yaml
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from services.llm.queue.contracts import ModelId
from services.llm.queue.store import QueueStore
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.http import Capacity, HttpError, _capacity_wire, _error, _event_wire
from services.llm.resource_manager.protocol import EventKind, Failure, ProgressEvent
import tests.integration.test_profile_validation_http as profile_validation_http
import tests.integration.test_resource_manager_http as resource_manager_http
import tests.integration.test_scheduler_http as scheduler_http
from services.llm.provisioning.http import create_app
from services.llm.provisioning.artifacts import SPECS
from services.llm.provisioning.volume import provision
from services.llm.health import HealthBoundary, HealthSnapshot, create_app as health_app


class Phase0ContractAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = yaml.safe_load(Path("docs/openapi.yaml").read_text())

    def validate(self, value, ref):
        document = copy.deepcopy(self.document)
        def nullable(node):
            if isinstance(node, dict):
                is_nullable = node.pop("nullable", False)
                children = list(node.items())
                if is_nullable:
                    original = {key: value for key, value in node.items()}
                    node.clear()
                    node.update({"anyOf": [original, {"type": "null"}]})
                    children = list(node.items())
                for _, child in children:
                    nullable(child)
            elif isinstance(node, list):
                for child in node:
                    nullable(child)
        nullable(document)
        registry = Registry().with_resource(
            "urn:phase0", Resource.from_contents(document, default_specification=DRAFT202012))
        resolved = registry.resolver(base_uri="urn:phase0").lookup(ref)
        validator = jsonschema.Draft202012Validator(document, registry=registry)
        errors = list(validator.descend(value, resolved.contents))
        if errors:
            raise errors[0]

    def assert_valid(self, payload, ref):
        self.validate(payload, ref)

    def assert_invalid(self, payload, ref):
        with self.assertRaises(jsonschema.ValidationError):
            self.validate(payload, ref)

    def test_production_rm_serializers_validate_nullable_progress_and_capacity(self):
        sample = SampleMetadata(1, 0, 1, 10, 100, (4,))
        profile = CapacityProfile(ModelId.SMOLLM, "gpu", "manifest", "model",
                                   "runtime", "adapter", "identity", 1, 1, 1, 20,
                                   (sample,), context_size=128)
        capacity = _capacity_wire(Capacity(profile, 1, 1, 1))
        self.validate(capacity, "#/components/schemas/RMCapacity")
        for event in (
            _event_wire(ProgressEvent(
            3, 2, EventKind.RESPONSE_FINISHED, None, None, "session", 1,
            None, Failure("cancelled", "request cancelled", False), None, False))
            ,
            _event_wire(ProgressEvent(
                4, 3, EventKind.RESPONSE_FINISHED, "request", "attempt", "session", 1,
                b"result", None, 12, True)),
        ):
            self.validate(event, "#/components/schemas/RMProgressEvent")

    def test_actual_scheduler_projection_and_rm_error_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            store = QueueStore(Path(directory) / "queue.sqlite", "scheduler", ModelId.SMOLLM)
            store.start_session("start")
            store.enqueue("request", "sha256:payload", idempotency_key="enqueue")
            self.validate(store.projection("request"), "#/components/schemas/Request")
            store.close()
        error = _error(None, HttpError(409, "fenced", "stale session"))
        self.validate(json.loads(error.body), "#/components/schemas/RMError")

    def test_complete_operation_inventory_future_set_and_post_contracts(self):
        paths = self.document["paths"]
        expected = {
            ("POST", "/schedulers/{schedulerId}/start", "startScheduler"),
            ("POST", "/schedulers/{schedulerId}/requests", "enqueue"),
            ("GET", "/schedulers/{schedulerId}/requests/{requestId}", "getRequest"),
            ("GET", "/schedulers/{schedulerId}/requests/{requestId}/watch", "watchRequest"),
            ("POST", "/schedulers/{schedulerId}/requests/{requestId}/cancel", "cancelSchedulerRequest"),
            ("POST", "/schedulers/{schedulerId}/stop", "stopScheduler"),
            ("POST", "/resource-manager/sessions", "startSession"),
            ("POST", "/resource-manager/sessions/{sessionToken}/submissions", "submit"),
            ("POST", "/resource-manager/sessions/{sessionToken}/requests/{requestId}/cancel", "cancelResourceRequest"),
            ("POST", "/resource-manager/sessions/{sessionToken}/stop", "stopSession"),
            ("GET", "/resource-manager/sessions/{sessionToken}/capacity", "getSessionCapacity"),
            ("GET", "/resource-manager/sessions/{sessionToken}/watch", "watchProgress"),
            ("POST", "/provisioning/verify-artifacts", "verifyArtifacts"),
            ("POST", "/provisioning/validate-profile", "validateProfile"),
            ("GET", "/health/live", "healthLive"),
            ("GET", "/health/ready", "healthReady"),
            ("GET", "/health/dependencies", "healthDependencies"),
        }
        actual = {(method.upper(), path, operation["operationId"])
                  for path, item in paths.items()
                  for method, operation in item.items()
                  if method in {"get", "post"} and operation.get("x-implementation") != "future"}
        self.assertEqual(actual, expected)
        self.assertEqual({path for _, path, _ in expected} | {
            "/resource-manager/capacity", "/provisioning/measure-capacity", "/metrics"}, set(paths))
        for method, path, _ in expected:
            operation = paths[path][method.lower()]
            self.assertNotEqual(operation.get("x-implementation"), "future")
            self.assertIn("responses", operation)
            if method == "POST":
                self.assertTrue(any(parameter.get("$ref", "").endswith("/IdempotencyKey")
                                    for parameter in operation.get("parameters", ())), path)
                if path not in {"/schedulers/{schedulerId}/requests/{requestId}/cancel",
                                "/resource-manager/sessions/{sessionToken}/requests/{requestId}/cancel",
                                "/resource-manager/sessions/{sessionToken}/stop"}:
                    self.assertIn("requestBody", operation, path)
        body_refs = {
            "/schedulers/{schedulerId}/start": "StartRequest",
            "/schedulers/{schedulerId}/requests": "EnqueueRequest",
            "/schedulers/{schedulerId}/stop": "StopRequest",
            "/resource-manager/sessions": "StartSessionRequest",
            "/resource-manager/sessions/{sessionToken}/submissions": "SubmitRequest",
            "/resource-manager/sessions/{sessionToken}/stop": "RMStopRequest",
            "/provisioning/verify-artifacts": "ArtifactVerificationRequest",
            "/provisioning/validate-profile": "ProfileRequest",
        }
        for path, name in body_refs.items():
            self.assertEqual(paths[path]["post"]["requestBody"]["$ref"],
                             f"#/components/requestBodies/{name}")
        future = {
            ("GET", "/resource-manager/capacity", "getCapacity"),
            ("POST", "/provisioning/measure-capacity", "measureCapacity"),
            ("GET", "/metrics", "metrics"),
        }
        self.assertEqual({(method.upper(), path, operation["operationId"])
                          for path, item in paths.items()
                          for method, operation in item.items()
                          if method in {"get", "post"} and operation.get("x-implementation") == "future"}, future)
        for path in ("/resource-manager/capacity", "/provisioning/measure-capacity", "/metrics"):
            operation = next(iter(paths[path].values()))
            self.assertEqual(operation.get("x-implementation"), "future", path)
            self.assertIn("responses", operation)
        for path, item in paths.items():
            for method, operation in item.items():
                if method == "post":
                    self.assertTrue(any(parameter.get("$ref", "").endswith("/IdempotencyKey")
                                        for parameter in operation.get("parameters", ())), path)
        for path in ("/schedulers/{schedulerId}/requests/{requestId}/watch",
                     "/resource-manager/sessions/{sessionToken}/watch"):
            self.assertTrue(any(parameter.get("$ref", "").endswith("/LastEventId")
                                for parameter in paths[path]["get"]["parameters"]))

    def test_error_and_sse_response_media_types_are_declared(self):
        for name in ("FutureError", "RMError"):
            response = self.document["components"]["responses"][name]
            self.assertIn("application/json", response["content"])
        for name in ("FutureSse", "RMSse"):
            response = self.document["components"]["responses"][name]
            self.assertIn("text/event-stream", response["content"])
        self.validate({"sequence": 1, "completionSequence": 1, "kind": "failure",
                       "requestId": None, "attemptToken": None, "sessionToken": "s",
                       "residencyGeneration": 1, "resultBase64": None,
                       "failure": None, "timeOnGpuMs": None,
                       "gpuTimingComplete": False}, "#/components/schemas/RMProgressEvent")

    async def test_actual_scheduler_sse_frame_validates_with_durable_cursor(self):
        fixture = scheduler_http.SchedulerHttpIntegrationTests()
        await fixture.asyncSetUp()
        try:
            await fixture.client.start("sched", "SmolLM", idempotency_key="contract-start")
            await fixture.client.enqueue(
                "sched", {"schedulerId": "sched", "requestId": "contract-r",
                           "modelId": "SmolLM", "payloadReference": "p"},
                idempotency_key="contract-enqueue")
            await fixture.client.cancel("sched", "contract-r", idempotency_key="contract-cancel")
            async with ClientSession() as client:
                response = await client.get(
                    fixture.client.base_url + "/schedulers/sched/requests/contract-r/watch")
                try:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers["Content-Type"].split(";", 1)[0],
                                     "text/event-stream")
                    frames = [frame for frame in
                              (await asyncio.wait_for(response.read(), 1)).split(b"\n\n") if frame]
                finally:
                    response.close()
            self.assertTrue(frames)
            for frame in frames:
                fields = dict(line.split(": ", 1) for line in frame.decode().splitlines())
                self.assertRegex(fields["id"], r"^[0-9]+$")
                self.assertIn("event", fields)
                self.assert_valid(json.loads(fields["data"]), "#/components/schemas/Request")
        finally:
            await fixture.asyncTearDown()

    async def test_actual_rm_backpressure_response_validates_as_submission(self):
        fixture = resource_manager_http.ResourceManagerHttpIntegrationTests()
        await fixture.asyncSetUp()
        try:
            for number in range(4):
                await fixture.client.submit(
                    fixture.session.session_token, f"contract-{number}", "attempt", b"x",
                    context_size=128, idempotency_key=f"contract-submit-{number}")
            async with ClientSession() as client:
                url = (fixture.client.base_url + "/resource-manager/sessions/" +
                       fixture.session.session_token + "/submissions")
                reference = "sha256:" + fixture.results.write(b"x")
                async with client.post(
                        url, headers={"Idempotency-Key": "contract-backpressure"},
                        json={"requestId": "contract-blocked", "attemptToken": "attempt",
                               "inputReference": reference,
                               "contextSizeEstimate": 128}) as response:
                    self.assertEqual(response.status, 429)
                    payload = await response.json()
                    self.assert_valid(payload, "#/components/schemas/RMSubmission")
                    self.assertFalse(payload["accepted"])
        finally:
            await fixture.asyncTearDown()

    async def test_actual_provisioning_success_and_failure_responses_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            model = source / "SmolLM"
            model.mkdir()
            for name in SPECS["SmolLM"]:
                (model / name).write_bytes(name.encode())
            volume = root / "volume"
            expected = provision({"SmolLM": model}, volume)
            server = TestServer(create_app({"offline": volume}))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    url = server.make_url("/provisioning/verify-artifacts")
                    async with client.post(url, headers={"Idempotency-Key": "contract-ok"},
                                           json={"volumeId": "offline"}) as response:
                        self.assertEqual(response.status, 200)
                        payload = await response.json()
                        self.assertEqual(payload["manifestSha256"], expected["manifest_sha256"])
                        self.assert_valid(payload, "#/components/schemas/ArtifactVerification")
                    async with client.post(url, headers={"Idempotency-Key": "contract-bad"},
                                           json={"volumeId": "offline",
                                                 "expectedManifestSha256": "0" * 64}) as response:
                        self.assertEqual(response.status, 422)
                        self.assert_valid(await response.json(), "#/components/schemas/Error")
            finally:
                await server.close()

    async def test_actual_profile_validation_success_and_failure_responses_validate(self):
        fixture = profile_validation_http.ProfileValidationHTTPTests()
        with tempfile.TemporaryDirectory() as tmp:
            profile, _, _, binding = fixture.stored_profile(tmp)
            server = TestServer(create_app({}, profile_registries={"SmolLM": binding}))
            await server.start_server()
            try:
                async with ClientSession() as client:
                    url = server.make_url("/provisioning/validate-profile")
                    async with client.post(url, headers={"Idempotency-Key": "contract-ok"},
                                            json=profile_validation_http.wire(profile)) as response:
                        self.assertEqual(response.status, 200)
                        self.assert_valid(await response.json(), "#/components/schemas/Ack")
                    async with client.post(url, headers={"Idempotency-Key": "contract-bad"},
                                           json={"modelId": "SmolLM"}) as response:
                        self.assertEqual(response.status, 400)
                        self.assert_valid(await response.json(), "#/components/schemas/Error")
            finally:
                await server.close()

    async def test_actual_health_success_failure_and_dependency_responses_validate(self):
        states = {name: True for name in ("sqlite", "gpu", "artifacts", "adapter",
                                          "ollama", "profile", "cleanup")}
        boundary = HealthBoundary(HealthSnapshot("stable", states))
        server = TestServer(health_app(boundary))
        await server.start_server()
        try:
            async with ClientSession() as client:
                async with client.get(server.make_url("/health/live")) as response:
                    self.assertEqual(response.status, 200)
                    self.assert_valid(await response.json(), "#/components/schemas/HealthLive")
                async with client.get(server.make_url("/health/ready")) as response:
                    self.assertEqual(response.status, 200)
                    self.assert_valid(await response.json(), "#/components/schemas/HealthReady")
                async with client.get(server.make_url("/health/dependencies")) as response:
                    self.assertEqual(response.status, 200)
                    self.assert_valid(await response.json(), "#/components/schemas/HealthDependencies")
        finally:
            await server.close()

        failing = HealthBoundary(HealthSnapshot("startup", states))
        server = TestServer(health_app(failing))
        await server.start_server()
        try:
            async with ClientSession() as client:
                async with client.get(server.make_url("/health/ready")) as response:
                    self.assertEqual(response.status, 503)
                    self.assert_valid(await response.json(), "#/components/schemas/HealthReady")
        finally:
            await server.close()

    def test_schema_validator_rejects_invalid_payload(self):
        self.assert_invalid({"accepted": True}, "#/components/schemas/RMSubmission")


if __name__ == "__main__":
    unittest.main()
