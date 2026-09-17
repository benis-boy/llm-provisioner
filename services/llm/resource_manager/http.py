"""Bounded HTTP/SSE adapter for the server-owned ResourceManager authority."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from aiohttp import ClientSession, ClientTimeout, web
from urllib.parse import quote

from services.llm.queue.contracts import ModelId
from services.llm.queue.results import ResultError, ResultStore
from services.llm.resource_manager.contracts import CapacityProfile, SampleMetadata
from services.llm.resource_manager.core import ResourceManager, ResourceManagerError
from services.llm.resource_manager.profiles import ProfileStore
from services.llm.resource_manager.protocol import Capacity, EventKind, Failure, ProgressEvent, SessionInfo, Submission

MAX_BODY = MAX_FRAME = 8 * 1024 * 1024
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class HttpError(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False):
        self.status, self.code, self.message, self.retryable = status, code, message, retryable


def _json_loads(value: bytes) -> Any:
    def pairs(items):
        answer = {}
        for key, item in items:
            if key in answer:
                raise ValueError("duplicate JSON object key")
            answer[key] = item
        return answer
    return json.loads(value, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON number")))


def _obj(value: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or any(key not in allowed for key in value):
        raise HttpError(400, "invalid_request", "request must be an object with known fields")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise HttpError(400, "invalid_request", f"{name} must be non-empty text")
    return value


def _integer(value: Any, name: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise HttpError(400, "invalid_request", f"{name} must be an integer")
    return value


def _failure(exc: BaseException) -> tuple[int, str, str, bool]:
    if isinstance(exc, HttpError): return exc.status, exc.code, exc.message, exc.retryable
    if isinstance(exc, ResourceManagerError):
        failure = exc.failure
        status = 429 if failure.code == "capacity_backpressure" else 409 if failure.code in {
            "idempotency_conflict", "scheduler_superseded", "request_in_flight", "cursor_expired", "invalid_cursor"} else 400
        return status, failure.code, failure.message, failure.retryable
    return 500, "resource_manager_failure", "resource manager operation failed", True


def _response(status: int, value: dict[str, Any]) -> web.Response:
    return web.json_response(value, status=status, dumps=lambda x: json.dumps(x, allow_nan=False, separators=(",", ":")))


def _error(request: web.Request, exc: BaseException) -> web.Response:
    status, code, message, retryable = _failure(exc)
    return _response(status, {"error": {"code": code, "message": message, "retryable": retryable}})


@dataclass(frozen=True)
class ModelBinding:
    model_id: ModelId
    gpu_uuid: str
    artifact_manifest_hash: str
    model_hash: str
    runtime_identity: str
    adapter_identity: str
    profile_store: ProfileStore
    provider: Any

    def resolve(self, *, context_size: int | None, bucket_identity: str | None) -> tuple[CapacityProfile, Any]:
        profile = self.profile_store.lookup(self.model_id, self.gpu_uuid, self.artifact_manifest_hash,
            self.model_hash, self.runtime_identity, self.adapter_identity, context_size=context_size, bucket_identity=bucket_identity)
        if profile is None: raise HttpError(409, "profile_unavailable", "no measured exact profile is available")
        return profile, self.provider


class ResourceManagerHttpServer:
    def __init__(self, core: ResourceManager, *, bindings: dict[ModelId, ModelBinding] | None = None,
                 resolver: Callable[..., Any] | None = None, result_store: ResultStore, max_body: int = MAX_BODY,
                 max_frame: int = MAX_FRAME, watch_timeout: float = 30.0, max_watches: int = 16,
                 write_timeout: float = 5.0):
        if (bindings is None) == (resolver is None): raise ValueError("provide exactly one server-side binding resolver")
        if not isinstance(result_store, ResultStore): raise ValueError("a shared ResultStore is required")
        if min(max_body, max_frame, max_watches) < 1 or not all(math.isfinite(x) and x > 0 for x in (watch_timeout, write_timeout)):
            raise ValueError("invalid HTTP bounds")
        self.core, self.bindings, self.resolver, self.result_store = core, bindings, resolver, result_store
        self.max_body, self.max_frame, self.watch_timeout, self.write_timeout = max_body, max_frame, watch_timeout, write_timeout
        self._watches = asyncio.BoundedSemaphore(max_watches)
        self.app = web.Application(client_max_size=max_body)
        self.app.add_routes([web.post("/resource-manager/sessions", self.start), web.post("/resource-manager/sessions/{sessionToken}/submissions", self.submit), web.post("/resource-manager/sessions/{sessionToken}/requests/{requestId}/cancel", self.cancel), web.post("/resource-manager/sessions/{sessionToken}/stop", self.stop), web.get("/resource-manager/sessions/{sessionToken}/capacity", self.capacity), web.get("/resource-manager/sessions/{sessionToken}/watch", self.watch)])

    async def _resolve(self, model: ModelId, context: int | None, bucket: str | None):
        if self.bindings is not None:
            binding = self.bindings.get(model)
            if binding is None: raise HttpError(400, "unknown_model", "model is not server configured")
            return binding.resolve(context_size=context, bucket_identity=bucket)
        value = self.resolver(model, context, bucket)
        return await value if inspect.isawaitable(value) else value

    async def _body(self, request: web.Request) -> dict[str, Any]:
        if request.content_length is not None and request.content_length > self.max_body: raise HttpError(413, "body_too_large", "request body exceeds limit")
        chunks, total = [], 0
        try:
            async for chunk in request.content.iter_chunked(64 * 1024):
                total += len(chunk)
                if total > self.max_body: raise HttpError(413, "body_too_large", "request body exceeds limit")
                chunks.append(chunk)
            value = _json_loads(b"".join(chunks))
            if not isinstance(value, dict): raise HttpError(400, "invalid_request", "request must be an object")
            return value
        except web.HTTPException as exc:
            if exc.status == 413: raise HttpError(413, "body_too_large", "request body exceeds limit") from exc
            raise
        except (ValueError, json.JSONDecodeError): raise HttpError(400, "invalid_json", "invalid strict JSON")

    @staticmethod
    def _selectors(body: dict[str, Any], *, required: bool) -> tuple[int | None, str | None]:
        context, bucket = body.get("contextSizeEstimate"), body.get("bucketIdentity")
        if "contextSizeEstimate" in body: _integer(context, "contextSizeEstimate")
        if "bucketIdentity" in body: _text(bucket, "bucketIdentity")
        if context is not None and bucket is not None: raise HttpError(400, "invalid_request", "profile selectors are exclusive")
        if required and context is None and bucket is None: raise HttpError(400, "invalid_request", "exactly one profile selector is required")
        return context, bucket

    def _read_reference(self, reference: str) -> bytes:
        if not _DIGEST.fullmatch(reference): raise HttpError(400, "invalid_reference", "inputReference must be sha256:<digest>")
        digest = reference[7:]
        try:
            fd = os.open(self.result_store.root / digest, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size > self.max_body: raise ResultError("result reference is unavailable or too large")
                blocks, remaining = [], self.max_body + 1
                while remaining:
                    block = os.read(fd, min(64 * 1024, remaining))
                    if not block: break
                    blocks.append(block); remaining -= len(block)
                data = b"".join(blocks)
            finally: os.close(fd)
        except (OSError, ResultError) as exc: raise HttpError(400, "invalid_reference", "result reference is unavailable or corrupted") from exc
        if len(data) > self.max_body or hashlib.sha256(data).hexdigest() != digest: raise HttpError(400, "invalid_reference", "result reference is unavailable or corrupted")
        return data

    async def start(self, request):
        try:
            body = _obj(await self._body(request), {"schedulerId", "modelId", "contextSizeEstimate", "bucketIdentity"})
            scheduler = _text(body.get("schedulerId"), "schedulerId")
            try: model = ModelId(_text(body.get("modelId"), "modelId"))
            except ValueError as exc: raise HttpError(400, "unknown_model", "model is not server configured") from exc
            context, bucket = self._selectors(body, required=True)
            profile, provider = await self._resolve(model, context, bucket)
            info = await self.core.start_session(scheduler, model, profile, provider, idempotency_key=_text(request.headers.get("Idempotency-Key"), "Idempotency-Key"))
            return _response(201, {"sessionToken": info.session_token, "schedulerId": info.scheduler_id, "modelId": info.model_id.value, "residencyGeneration": info.generation})
        except Exception as exc: return _error(request, exc)

    async def submit(self, request):
        try:
            token = _text(request.match_info["sessionToken"], "sessionToken")
            # Fence stale tokens before touching shared-storage bytes.
            capacity = await self.core.get_capacity(token)
            body = _obj(await self._body(request), {"requestId", "attemptToken", "inputReference", "contextSizeEstimate", "bucketIdentity"})
            context, bucket = self._selectors(body, required=False)
            if (context is not None or bucket is not None) and not capacity.profile.accepts_request(context, bucket):
                raise HttpError(400, "invalid_input", "request does not match exact capacity profile")
            if context is None and bucket is None:
                # The transport-neutral protocol's normal scheduler path has no
                # selector; carry the active server-owned profile selector into
                # the core rather than trusting a client default.
                context, bucket = capacity.profile.context_size, capacity.profile.bucket_identity
            payload = await asyncio.to_thread(self._read_reference, _text(body.get("inputReference"), "inputReference"))
            result = await self.core.submit(token, _text(body.get("requestId"), "requestId"), _text(body.get("attemptToken"), "attemptToken"), payload, idempotency_key=_text(request.headers.get("Idempotency-Key"), "Idempotency-Key"), context_size=context, bucket_identity=bucket)
            return _response(429 if result.backpressure else 202, {"accepted": result.accepted, "requestId": result.request_id, "attemptToken": result.attempt, "sessionToken": result.session_token, "residencyGeneration": result.generation, "backpressure": result.backpressure})
        except Exception as exc: return _error(request, exc)

    async def cancel(self, request):
        try:
            acknowledged = await self.core.cancel_request(request.match_info["sessionToken"], request.match_info["requestId"], idempotency_key=_text(request.headers.get("Idempotency-Key"), "Idempotency-Key"))
            return _response(200, {"acknowledged": acknowledged})
        except Exception as exc: return _error(request, exc)

    async def stop(self, request):
        try:
            body = {} if request.content_length == 0 else _obj(await self._body(request), {"reason"}); reason = body.get("reason", "stopped")
            _text(reason, "reason")
            await self.core.stop_session(request.match_info["sessionToken"], reason=reason, idempotency_key=_text(request.headers.get("Idempotency-Key"), "Idempotency-Key"))
            return _response(202, {"acknowledged": True})
        except Exception as exc: return _error(request, exc)

    async def capacity(self, request):
        try:
            cap = await self.core.get_capacity(request.match_info["sessionToken"])
            return _response(200, _capacity_wire(cap))
        except Exception as exc: return _error(request, exc)

    async def watch(self, request):
        iterator = response = pending_task = None
        acquired = False
        try:
            raw = request.headers.get("Last-Event-ID", "0")
            if not raw.isdigit(): raise HttpError(400, "invalid_cursor", "Last-Event-ID must be nonnegative integer")
            if self._watches.locked(): raise HttpError(429, "watch_limit", "too many progress watches", True)
            # asyncio semaphore acquisition does not suspend while unlocked;
            # no other handler can interleave between this check and acquisition.
            await self._watches.acquire(); acquired = True
            iterator = self.core.watch_progress(request.match_info["sessionToken"], int(raw))
            # Enter the generator before headers so invalid/future/expired cursors are HTTP errors.
            pending_task = asyncio.create_task(iterator.__anext__())
            done, _ = await asyncio.wait((pending_task,), timeout=.05)
            first = pending_task.result() if done else None
            response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await response.prepare(request)
            async def send(event):
                wire = _event_wire(event)
                line = f"id: {event.sequence}\nevent: progress\ndata: {json.dumps(wire, allow_nan=False, separators=(',', ':'))}\n\n".encode()
                if len(line) > self.max_frame: raise HttpError(413, "frame_too_large", "SSE frame exceeds limit")
                await asyncio.wait_for(response.write(line), self.write_timeout)
            if first is not None: await send(first); pending_task = None
            while True:
                if pending_task is None: pending_task = asyncio.create_task(iterator.__anext__())
                done, _ = await asyncio.wait((pending_task,), timeout=self.watch_timeout)
                if not done:
                    await asyncio.wait_for(response.write(b": heartbeat\n\n"), self.write_timeout); continue
                event = pending_task.result(); pending_task = None
                await send(event)
        except StopAsyncIteration:
            return response
        except asyncio.CancelledError: raise
        except ConnectionResetError:
            return response
        except Exception as exc:
            if response is not None and response.prepared:
                status, code, message, retryable = _failure(exc)
                frame = f"event: error\ndata: {json.dumps({'error': {'code': code, 'message': message, 'retryable': retryable}})}\n\n".encode()
                try:
                    if len(frame) <= self.max_frame:
                        await asyncio.wait_for(response.write(frame), self.write_timeout)
                except (ConnectionResetError, RuntimeError, asyncio.TimeoutError): pass
                return response
            return _error(request, exc)
        finally:
            if pending_task is not None and not pending_task.done(): pending_task.cancel(); await asyncio.gather(pending_task, return_exceptions=True)
            if iterator is not None and hasattr(iterator, "aclose"): await iterator.aclose()
            if acquired: self._watches.release()


def _sample_wire(sample): return {"concurrency": sample.concurrency, "wave": sample.wave, "successfulRequests": sample.successful_requests, "wallTimeMs": sample.wall_time_ms, "peakVramBytes": sample.peak_vram_bytes, "latencyMs": list(sample.latency_ms)}
def _profile_wire(p): return {"modelId": p.model_id.value, "gpuUuid": p.gpu_uuid, "artifactManifestHash": p.artifact_manifest_hash, "modelHash": p.model_hash, "runtimeIdentity": p.runtime_identity, "adapterIdentity": p.adapter_identity, "profileIdentity": p.profile_identity, "optimalParallelism": p.optimal_parallelism, "memorySafeN": p.memory_safe_n, "bufferCapacity": p.buffer_capacity, "safetyReservePercent": p.safety_reserve_percent, "rawSamples": [_sample_wire(s) for s in p.raw_samples], "contextSize": p.context_size, "bucketIdentity": p.bucket_identity}
def _capacity_wire(c): return {"profile": _profile_wire(c.profile), "executionSlots": c.execution_slots, "bufferSlots": c.buffer_slots, "freeSlots": c.free_slots}
def _event_wire(e): return {"sequence": e.sequence, "completionSequence": e.completion_sequence, "kind": e.kind.value, "requestId": e.request_id, "attemptToken": e.attempt, "sessionToken": e.session_token, "residencyGeneration": e.generation, "resultBase64": base64.b64encode(e.result).decode("ascii") if e.result is not None else None, "failure": None if e.failure is None else {"code": e.failure.code, "message": e.failure.message, "retryable": e.failure.retryable}, "timeOnGpuMs": e.time_on_gpu_ms, "gpuTimingComplete": e.gpu_timing_complete}


class ResourceManagerHTTPClient:
    """Typed protocol client. Provider/profile objects are never serialized."""
    def __init__(self, base_url: str, *, result_store: ResultStore | None, timeout: float = 30.0, max_frame: int = MAX_FRAME):
        if result_store is None: raise ValueError("HTTP submission requires a shared ResultStore")
        if not math.isfinite(timeout) or timeout <= 0 or max_frame < 1: raise ValueError("invalid client bounds")
        self.base_url, self.result_store, self.timeout, self.max_frame = base_url.rstrip("/"), result_store, timeout, max_frame

    @staticmethod
    def _path_segment(value: str) -> str:
        return quote(_text(value, "path segment"), safe="")

    async def _bounded_json(self, response):
        chunks, size = [], 0
        async for chunk in response.content.iter_chunked(64 * 1024):
            size += len(chunk)
            if size > self.max_frame: raise ResourceManagerError(Failure("response_too_large", "HTTP response exceeds limit", True))
            chunks.append(chunk)
        try: return _json_loads(b"".join(chunks))
        except ValueError as exc: raise ResourceManagerError(Failure("invalid_response", "HTTP response is not strict JSON", True)) from exc

    async def _json(self, method, path, body=None, key=None, headers=None, expected_statuses=()):
        headers = dict(headers or {})
        if key is not None: headers["Idempotency-Key"] = _text(key, "idempotency key")
        async with ClientSession(timeout=ClientTimeout(total=self.timeout), read_bufsize=self.max_frame) as session:
            async with session.request(method, self.base_url + path, json=body, headers=headers) as response:
                value = await self._bounded_json(response)
                if response.status >= 400 and response.status not in expected_statuses:
                    error = value.get("error", {}) if isinstance(value, dict) else {}
                    raise ResourceManagerError(Failure(error.get("code", "http_error"), error.get("message", "HTTP operation failed"), bool(error.get("retryable", False))))
                return value

    async def start_session(self, scheduler_id, model_id, profile, provider, *, idempotency_key):
        del provider
        if not isinstance(profile, CapacityProfile) or profile.model_id != ModelId(model_id): raise ValueError("profile must match model")
        body = {"schedulerId": scheduler_id, "modelId": ModelId(model_id).value}
        if profile.context_size is not None: body["contextSizeEstimate"] = profile.context_size
        else: body["bucketIdentity"] = profile.bucket_identity
        v = await self._json("POST", "/resource-manager/sessions", body, idempotency_key)
        answer = SessionInfo(v["schedulerId"], v["sessionToken"], ModelId(v["modelId"]), _integer(v["residencyGeneration"], "residencyGeneration"))
        return answer

    async def submit(self, session_token, request_id, attempt, payload, *, idempotency_key, context_size=None, bucket_identity=None):
        if not isinstance(payload, bytes) or len(payload) > MAX_BODY: raise ValueError("payload must be bounded bytes")
        if context_size is not None and bucket_identity is not None: raise ValueError("profile selectors are exclusive")
        reference = "sha256:" + await asyncio.to_thread(self.result_store.write, payload)
        body = {"requestId": request_id, "attemptToken": attempt, "inputReference": reference}
        if context_size is not None: body["contextSizeEstimate"] = context_size
        elif bucket_identity is not None: body["bucketIdentity"] = bucket_identity
        v = await self._json("POST", f"/resource-manager/sessions/{self._path_segment(session_token)}/submissions", body, idempotency_key, expected_statuses=(429,))
        failure = _decode_failure(v.get("failure")) if "failure" in v else None
        return Submission(v["accepted"], v["requestId"], v["attemptToken"], v["sessionToken"], v["residencyGeneration"], v["backpressure"], failure)

    async def cancel_request(self, session_token, request_id, *, idempotency_key): return (await self._json("POST", f"/resource-manager/sessions/{self._path_segment(session_token)}/requests/{self._path_segment(request_id)}/cancel", None, idempotency_key))["acknowledged"]
    async def stop_session(self, session_token, *, reason="stopped", idempotency_key): await self._json("POST", f"/resource-manager/sessions/{self._path_segment(session_token)}/stop", {"reason": reason}, idempotency_key)
    async def get_capacity(self, session_token):
        v = await self._json("GET", f"/resource-manager/sessions/{self._path_segment(session_token)}/capacity"); p = v["profile"]
        samples = tuple(SampleMetadata(s["concurrency"], s["wave"], s["successfulRequests"], s["wallTimeMs"], s["peakVramBytes"], tuple(s["latencyMs"])) for s in p["rawSamples"])
        profile = CapacityProfile(ModelId(p["modelId"]), p["gpuUuid"], p["artifactManifestHash"], p["modelHash"], p["runtimeIdentity"], p["adapterIdentity"], p["profileIdentity"], p["optimalParallelism"], p["memorySafeN"], p["bufferCapacity"], p["safetyReservePercent"], samples, p.get("contextSize"), p.get("bucketIdentity"))
        return Capacity(profile, v["executionSlots"], v["bufferSlots"], v["freeSlots"])
    async def watch_progress(self, session_token, after_sequence=0) -> AsyncIterator[ProgressEvent]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0: raise ValueError("cursor must be nonnegative integer")
        async with ClientSession(timeout=ClientTimeout(total=None, sock_read=self.timeout), read_bufsize=self.max_frame) as session:
            async with session.get(self.base_url + f"/resource-manager/sessions/{self._path_segment(session_token)}/watch", headers={"Last-Event-ID": str(after_sequence)}) as response:
                if response.status >= 400:
                    value = await self._bounded_json(response); error = value.get("error", {}) if isinstance(value, dict) else {}
                    raise ResourceManagerError(Failure(error.get("code", "http_error"), error.get("message", "HTTP operation failed"), bool(error.get("retryable", False))))
                buffer = b""
                async for chunk in response.content.iter_chunked(64 * 1024):
                    buffer += chunk
                    while b"\n\n" in buffer:
                        frame, buffer = buffer.split(b"\n\n", 1)
                        if len(frame) > self.max_frame: raise ResourceManagerError(Failure("frame_too_large", "SSE frame exceeds limit", True))
                        lines = frame.split(b"\n")
                        if not any(line.startswith(b"data: ") for line in lines): continue
                        data = _json_loads(next(line[6:] for line in lines if line.startswith(b"data: ")))
                        if "error" in data: e = data["error"]; raise ResourceManagerError(Failure(e["code"], e["message"], e["retryable"]))
                        event = _decode_event(data)
                        try: event_id = int(next(line[4:] for line in lines if line.startswith(b"id: ")))
                        except (StopIteration, ValueError) as exc: raise ResourceManagerError(Failure("invalid_response", "invalid SSE event id", True)) from exc
                        if event_id != event.sequence: raise ResourceManagerError(Failure("invalid_response", "SSE event id differs from sequence", True))
                        yield event
                    if len(buffer) > self.max_frame: raise ResourceManagerError(Failure("frame_too_large", "SSE frame exceeds limit", True))
                if buffer: raise ResourceManagerError(Failure("invalid_response", "truncated SSE frame", True))


def _decode_event(v):
    required = {"sequence", "completionSequence", "kind", "requestId", "attemptToken", "sessionToken", "residencyGeneration", "resultBase64", "failure", "timeOnGpuMs", "gpuTimingComplete"}
    if not isinstance(v, dict) or set(v) != required: raise ResourceManagerError(Failure("invalid_response", "invalid progress event", True))
    result = None
    if v["resultBase64"] is not None:
        try: result = base64.b64decode(v["resultBase64"], validate=True)
        except (ValueError, TypeError) as exc: raise ResourceManagerError(Failure("invalid_response", "invalid result encoding", True)) from exc
        if len(result) > MAX_FRAME: raise ResourceManagerError(Failure("frame_too_large", "SSE result exceeds limit", True))
    failure = _decode_failure(v["failure"])
    try:
        if any(item is not None and not isinstance(item, str) for item in (v["requestId"], v["attemptToken"])) or isinstance(v["gpuTimingComplete"], bool) is False:
            raise ValueError("invalid event field")
        if v["timeOnGpuMs"] is not None: _integer(v["timeOnGpuMs"], "timeOnGpuMs", zero=True)
        return ProgressEvent(_integer(v["sequence"], "sequence", zero=True), _integer(v["completionSequence"], "completionSequence", zero=True), EventKind(v["kind"]), v["requestId"], v["attemptToken"], _text(v["sessionToken"], "sessionToken"), _integer(v["residencyGeneration"], "residencyGeneration"), result, failure, v["timeOnGpuMs"], v["gpuTimingComplete"])
    except (ValueError, HttpError) as exc: raise ResourceManagerError(Failure("invalid_response", "invalid progress event", True)) from exc


def _decode_failure(value):
    if value is None: return None
    if not isinstance(value, dict) or set(value) != {"code", "message", "retryable"} or not isinstance(value["code"], str) or not value["code"] or not isinstance(value["message"], str) or not isinstance(value["retryable"], bool):
        raise ResourceManagerError(Failure("invalid_response", "invalid failure", True))
    return Failure(value["code"], value["message"], value["retryable"])


ResourceManagerClient = ResourceManagerHTTPClient
