import unittest
from copy import deepcopy
from pathlib import Path

import jsonschema
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
import yaml
from openapi_spec_validator import validate


class ResourceManagerOpenApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = yaml.safe_load(Path("docs/openapi.yaml").read_text())

    @staticmethod
    def _registry(document):
        """Resolve OpenAPI component refs without deprecated RefResolver."""
        # OpenAPI 3.0 documents do not declare a JSON-Schema dialect.  The
        # component schemas use the draft vocabulary understood by the local
        # jsonschema validator, so select it explicitly for ref resolution.
        return Registry().with_resource(
            "urn:openapi", Resource.from_contents(document, default_specification=DRAFT202012)
        )

    @classmethod
    def _validate_ref(cls, instance, ref, document):
        # Keep the root document as the resolution scope; validating an
        # extracted component directly would make its local ``#`` refs point
        # at that component instead of the OpenAPI document.
        registry = cls._registry(document)
        resolved = registry.resolver(base_uri="urn:openapi").lookup(ref)
        # Retain the document as the validator's root schema so nested local
        # references continue to resolve against ``#/components``.
        validator = jsonschema.Draft202012Validator(document, registry=registry)
        errors = list(validator.descend(instance, resolved.contents))
        if errors:
            raise errors[0]

    def test_resource_manager_openapi_is_fully_parsed_and_valid(self):
        validate(self.document)
        self.assertTrue({
            "/resource-manager/sessions",
            "/resource-manager/sessions/{sessionToken}/submissions",
            "/resource-manager/sessions/{sessionToken}/requests/{requestId}/cancel",
            "/resource-manager/sessions/{sessionToken}/stop",
            "/resource-manager/sessions/{sessionToken}/capacity",
            "/resource-manager/sessions/{sessionToken}/watch",
        }.issubset(self.document["paths"]))

    def test_submission_wire_schema_requires_session_token(self):
        response = self.document["paths"][
            "/resource-manager/sessions/{sessionToken}/submissions"
        ]["post"]["responses"]["202"]
        response_name = response["$ref"].rsplit("/", 1)[1]
        response_document = self.document["components"]["responses"][response_name]
        schema_ref = response_document["content"]["application/json"]["schema"]["$ref"]
        valid = {"accepted": True, "requestId": "r", "attemptToken": "a",
                 "sessionToken": "s", "residencyGeneration": 1, "backpressure": False}
        self._validate_ref(valid, schema_ref, self.document)
        with self.assertRaises(jsonschema.ValidationError):
            self._validate_ref({key: value for key, value in valid.items() if key != "sessionToken"},
                               schema_ref, self.document)

    def test_scheduler_operations_are_implemented_and_future_mutations_retain_contracts(self):
        paths = self.document["paths"]
        scheduler_paths = {
            "/schedulers/{schedulerId}/start", "/schedulers/{schedulerId}/requests",
            "/schedulers/{schedulerId}/requests/{requestId}/cancel", "/schedulers/{schedulerId}/stop",
        }
        for path in scheduler_paths:
            operation = next(iter(paths[path].values()))
            self.assertNotEqual(operation.get("x-implementation"), "future")
            self.assertIn("responses", operation, path)
        for path in {"/provisioning/measure-capacity"}:
            operation = next(iter(paths[path].values()))
            self.assertEqual(operation["x-implementation"], "future")
            self.assertIn("responses", operation, path)
            self.assertTrue(operation["responses"], path)
            self.assertTrue(any(parameter.get("$ref", "").endswith("/IdempotencyKey")
                                for parameter in operation.get("parameters", ())), path)
        for path in ("/schedulers/{schedulerId}/start", "/schedulers/{schedulerId}/requests",
                     "/schedulers/{schedulerId}/stop",
                     "/provisioning/measure-capacity"):
            operation = next(iter(paths[path].values()))
            self.assertIn("requestBody", operation, path)

        validation = paths["/provisioning/validate-profile"]["post"]
        self.assertNotEqual(validation.get("x-implementation"), "future")
        self.assertIn("200", validation["responses"])
        self.assertIn("400", validation["responses"])
        self.assertIn("422", validation["responses"])
        self.assertIn("429", validation["responses"])
        self.assertIn("504", validation["responses"])
        self.assertTrue(any(p.get("$ref", "").endswith("/IdempotencyKey")
                            for p in validation["parameters"]))
        self.assertEqual(validation["requestBody"]["$ref"], "#/components/requestBodies/ProfileRequest")

    def test_profile_request_shape_preserves_full_profile_and_model_selector_parity(self):
        schema = self.document["components"]["schemas"]["Profile"]
        self.assertTrue(set(schema["required"]).issuperset({"modelId", "rawSamples", "profileIdentity"}))
        self.assertFalse(schema["additionalProperties"])
        for selector in ("contextSize", "bucketIdentity"):
            self.assertIn(selector, schema["properties"])
        validation_schema = self.document["components"]["schemas"]["ProfileValidationRequest"]
        self.assertEqual(validation_schema["allOf"][0]["$ref"], "#/components/schemas/Profile")
        alternatives = validation_schema["allOf"][1]["oneOf"]
        self.assertEqual(alternatives[0]["properties"]["modelId"]["enum"], ["SmolLM"])
        self.assertEqual(alternatives[1]["properties"]["modelId"]["enum"], ["CoEdIT", "GECToR"])

    def test_profile_validation_responses_use_actual_ack_and_error_schemas(self):
        responses = self.document["paths"]["/provisioning/validate-profile"]["post"]["responses"]
        ack_ref = self.document["components"]["responses"]["FutureAck"]["content"]["application/json"]["schema"]["$ref"]
        error_ref = self.document["components"]["responses"]["FutureError"]["content"]["application/json"]["schema"]["$ref"]
        self.assertEqual(responses["200"]["$ref"], "#/components/responses/FutureAck")
        self._validate_ref({"accepted": True, "operationId": "a"}, ack_ref, self.document)
        for status in ("400", "413", "422", "429", "504"):
            self._validate_ref({"code": "profile_validation_failed", "message": "failed", "retryable": False},
                               error_ref, self.document)
            self.assertEqual(responses[status]["$ref"], "#/components/responses/FutureError")

    def test_artifact_verification_is_the_read_only_slice(self):
        operation = self.document["paths"]["/provisioning/verify-artifacts"]["post"]
        self.assertNotEqual(operation.get("x-implementation"), "future")
        self.assertEqual(operation["requestBody"]["$ref"], "#/components/requestBodies/ArtifactVerificationRequest")
        self.assertEqual(operation["responses"]["200"]["$ref"], "#/components/responses/ArtifactVerification")

    def test_artifact_http_success_and_errors_validate_against_declared_schemas(self):
        def response_schema(name):
            response = self.document["components"]["responses"][name]
            # Validate against the complete OpenAPI document so nested refs in
            # the resolved response schema retain their ``#/components`` base.
            return response["content"]["application/json"]["schema"]["$ref"]

        self._validate_ref({"volumeId": "offline", "manifestSha256": "0" * 64,
                            "models": [{"modelId": "SmolLM", "fileCount": 2, "totalBytes": 3}],
                            "verified": True, "verificationScope": "selected-file-integrity"},
                           response_schema("ArtifactVerification"), self.document)
        error_schema = response_schema("FutureError")
        for payload in (
                {"code": "invalid_request", "message": "bad", "retryable": False},
                {"code": "unknown_volume", "message": "missing", "retryable": False},
                {"code": "overloaded", "message": "busy", "retryable": True},
                {"code": "verification_timeout", "message": "timed out", "retryable": True},
                {"code": "artifact_verification_failed", "message": "failed", "retryable": False},
        ):
            self._validate_ref(payload, error_schema, self.document)

    def test_scheduler_error_stop_and_sse_wire_schemas_match_adapter(self):
        document = deepcopy(self.document)
        # OpenAPI 3's ``nullable`` is not understood by jsonschema's
        # JSON-Schema validator. Translate it throughout the document so
        # referenced component schemas receive the same treatment.
        def normalize_nullable(node):
            if isinstance(node, dict):
                if node.pop("nullable", False) and "type" in node:
                    node["type"] = [node["type"], "null"]
                for child in node.values():
                    normalize_nullable(child)
            elif isinstance(node, list):
                for child in node:
                    normalize_nullable(child)
        normalize_nullable(document)
        schemas = document["components"]["schemas"]
        for name, value in (
            ("Error", {"code": "invalid_request", "message": "bad", "retryable": False}),
            ("SchedulerStop", {"accepted": True, "cancelled": False, "affected": 1}),
            ("SseEnvelope", {"id": "12", "event": "status", "data": {
                "requestId": "r", "schedulerId": "s", "modelId": "SmolLM", "status": "done",
                "sequence": 12, "runningAt": None, "doneAt": None, "runningToDoneMs": None,
                "timeOnGpuMs": None, "gpuTimingComplete": False, "errorCode": None,
                "resultReference": None, "cancellation": False,
            }}),
        ):
            self._validate_ref(value, f"#/components/schemas/{name}", document)
