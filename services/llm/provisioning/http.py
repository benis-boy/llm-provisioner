"""Bounded read-only HTTP adapter for artifact-volume verification."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import math
from types import MappingProxyType
from typing import Callable, Mapping

from aiohttp import web

from .artifacts import SPECS
from .volume import verify_current
from .profile_validation import ProfileSyntaxError, ProfileValidationBinding, validate_profile
from services.llm.queue.contracts import ModelId


def create_app(volume_roots: Mapping[str, str | Path], *, verifier: Callable[[str | Path], dict] = verify_current,
               max_concurrent: int = 2, max_body_bytes: int = 4096,
               timeout_seconds: float = 30.0, profile_registries: Mapping[str, ProfileValidationBinding] | None = None,
               profile_max_body_bytes: int = 4 * 1024 * 1024) -> web.Application:
    """Create an app with an operator-supplied volume-id to absolute-root map."""
    if (type(max_concurrent) is not int or max_concurrent < 1 or
               type(max_body_bytes) is not int or max_body_bytes < 1 or
               type(profile_max_body_bytes) is not int or profile_max_body_bytes < 1 or profile_max_body_bytes > 4 * 1024 * 1024 or
            not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("invalid verification bounds")
    roots = {}
    for volume_id, root in volume_roots.items():
        if not isinstance(volume_id, str) or not volume_id or not isinstance(root, (str, Path)):
            raise ValueError("invalid artifact volume configuration")
        path = Path(root)
        if not path.is_absolute():
            raise ValueError("artifact volume roots must be absolute")
        roots[volume_id] = path
    profile_bindings = None
    if profile_registries is not None:
        if any(not isinstance(k, str) or k not in {m.value for m in ModelId}
                                         or not isinstance(v, ProfileValidationBinding)
                                         for k, v in profile_registries.items()):
            raise ValueError("invalid profile registry configuration")
        profile_bindings = MappingProxyType(dict(profile_registries))
    slots = asyncio.Semaphore(max_concurrent)
    workers: set[asyncio.Task] = set()
    profile_workers: set[asyncio.Task] = set()
    profile_slots = asyncio.Semaphore(max_concurrent)

    def error(code: str, message: str, status: int, retryable: bool = False):
        return web.json_response({"code": code, "message": message, "retryable": retryable}, status=status)

    def finish(task: asyncio.Task):
        workers.discard(task)
        # Retrieve failures even when the request timed out and no awaiter remains.
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        slots.release()

    async def handle(request: web.Request) -> web.Response:
        if request.content_type != "application/json":
            return error("invalid_request", "application/json is required", 400)
        key = request.headers.get("Idempotency-Key")
        if key is None or not key.strip():
            return error("invalid_request", "Idempotency-Key is required", 400)
        if request.content_length is not None and request.content_length > max_body_bytes:
            return error("request_too_large", "request body is too large", 413)
        try:
            chunks = []
            size = 0
            async for chunk in request.content.iter_chunked(min(64 * 1024, max_body_bytes + 1)):
                size += len(chunk)
                if size > max_body_bytes:
                    return error("request_too_large", "request body is too large", 413)
                chunks.append(chunk)
            raw = b"".join(chunks)
        except ConnectionError:
            return error("invalid_request", "request body could not be read", 400)

        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate request key")
                result[key] = value
            return result

        def reject_nonfinite(value):
            raise ValueError("non-finite request value")
        try:
            body = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates,
                              parse_constant=reject_nonfinite)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
            return error("invalid_request", "malformed JSON body", 400)
        if not isinstance(body, dict) or set(body) - {"volumeId", "expectedManifestSha256"}:
            return error("invalid_request", "invalid verification request", 400)
        volume_id = body.get("volumeId")
        expected_present = "expectedManifestSha256" in body
        expected = body.get("expectedManifestSha256")
        if not isinstance(volume_id, str) or not volume_id or (expected_present and
                (not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected))):
            return error("invalid_request", "invalid verification request", 400)
        if volume_id not in roots:
            return error("unknown_volume", "unknown artifact volume", 404)
        if slots.locked():
            return error("overloaded", "artifact verification is busy", 429, True)
        await slots.acquire()
        task = asyncio.create_task(asyncio.to_thread(verifier, roots[volume_id]))
        workers.add(task)
        task.add_done_callback(finish)
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout_seconds)
        except asyncio.TimeoutError:
            return error("verification_timeout", "artifact verification timed out", 504, True)
        except Exception:
            return error("artifact_verification_failed", "artifact volume verification failed", 422)
        if (not isinstance(result, dict) or
                not {"manifestSha256", "models", "verified", "verificationScope"}.issubset(result) or
                not set(result).issubset(
                {"volumeId", "manifestSha256", "models", "verified", "verificationScope"}) or
                (not isinstance(result.get("manifestSha256"), str) or len(result["manifestSha256"]) != 64 or
                 any(c not in "0123456789abcdef" for c in result["manifestSha256"])) or
                result.get("verified") is not True or result.get("verificationScope") != "selected-file-integrity" or
                not isinstance(result.get("models"), list)):
            return error("artifact_verification_failed", "artifact volume verification failed", 422)
        summaries = []
        for item in result["models"]:
            if (not isinstance(item, dict) or set(item) != {"modelId", "fileCount", "totalBytes"} or
                    not isinstance(item["modelId"], str) or item["modelId"] not in SPECS or
                    type(item["fileCount"]) is not int or item["fileCount"] < 0 or
                    type(item["totalBytes"]) is not int or item["totalBytes"] < 0):
                return error("artifact_verification_failed", "artifact volume verification failed", 422)
            summaries.append(item)
        if [item["modelId"] for item in summaries] != sorted({item["modelId"] for item in summaries}):
            return error("artifact_verification_failed", "artifact volume verification failed", 422)
        result = {"volumeId": volume_id, "manifestSha256": result["manifestSha256"],
                  "models": summaries, "verified": True,
                  "verificationScope": "selected-file-integrity"}
        if expected_present and result["manifestSha256"] != expected:
            return error("manifest_mismatch", "manifest digest does not match expectation", 422)
        return web.json_response(result)

    async def cleanup(_app):
        pending = tuple(workers | profile_workers)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    app = web.Application(client_max_size=max(max_body_bytes, profile_max_body_bytes))
    app.router.add_post("/provisioning/verify-artifacts", handle)
    async def validate_handle(request: web.Request) -> web.Response:
        if profile_bindings is None:
            return error("profile_validation_failed", "profile validation failed", 422)
        if (request.content_type != "application/json" or
                not request.headers.get("Idempotency-Key", "").strip()):
            return error("invalid_request", "invalid profile validation request", 400)
        if profile_slots.locked():
            return error("overloaded", "profile validation is busy", 429, True)
        await profile_slots.acquire()
        acquired = True
        worker = None
        try:
            if request.content_length is not None and request.content_length > profile_max_body_bytes:
                return error("request_too_large", "request body is too large", 413)
            chunks, size = [], 0
            async for chunk in request.content.iter_chunked(min(64 * 1024, profile_max_body_bytes + 1)):
                size += len(chunk)
                if size > profile_max_body_bytes:
                    return error("request_too_large", "request body is too large", 413)
                chunks.append(chunk)
            raw = b"".join(chunks)
        except (ConnectionError, asyncio.IncompleteReadError):
            return error("invalid_request", "request body could not be read", 400)
        finally:
            # A successfully read body continues into the worker hand-off below.
            # Read errors and size rejections have no worker ownership.
            if acquired and worker is None and "raw" not in locals():
                profile_slots.release()
                acquired = False

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate key")
                result[key] = value
            return result
        try:
            body = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            worker = asyncio.create_task(asyncio.to_thread(validate_profile, body, profile_bindings))
            profile_workers.add(worker)
            def finish_profile(task):
                profile_workers.discard(task)
                # Consume a late worker failure before releasing its capacity.
                try:
                    task.exception()
                except asyncio.CancelledError:
                    pass
                profile_slots.release()
            worker.add_done_callback(finish_profile)
            operation_id = await asyncio.wait_for(asyncio.shield(worker), timeout_seconds)
        except asyncio.TimeoutError:
            return error("profile_validation_timeout", "profile validation timed out", 504, True)
        except (UnicodeDecodeError, json.JSONDecodeError, ProfileSyntaxError, RecursionError):
            return error("invalid_request", "invalid profile validation request", 400)
        except (ValueError, OverflowError):
            return error("profile_validation_failed", "profile validation failed", 422)
        except TypeError:
            return error("invalid_request", "invalid profile validation request", 400)
        finally:
            if acquired and worker is None:
                profile_slots.release()
        return web.json_response({"accepted": True, "operationId": operation_id})
    app.router.add_post("/provisioning/validate-profile", validate_handle)
    app.on_cleanup.append(cleanup)
    return app
