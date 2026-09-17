"""Small, bounded HTTP adapter for application-owned queue schedulers.

The registry is deliberately supplied by the application.  This module never
imports a scheduler, resolves a dotted path, or reads a payload reference.
"""
from __future__ import annotations

import asyncio
import json
import math
from typing import Any, AsyncIterator
from urllib.parse import quote

from aiohttp import ClientSession, ClientTimeout, web

from .contracts import FunctionDescriptor, InsertionMode, ModelId
from .scheduler import QueueScheduler
from .store import (DependencyError, DuplicateRequest, IdempotencyConflict,
                    OperationStale, QueueError, SessionError)

MAX_BODY = 1024 * 1024
MAX_FRAME = 256 * 1024
EVENT_BATCH = 64


class SchedulerHttpError(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False):
        self.status, self.code, self.message, self.retryable = status, code, message, retryable


def _strict(data: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite number")))


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchedulerHttpError(400, "invalid_request", f"{name} must be non-empty text")
    return value


def _error(exc: BaseException) -> web.Response:
    if isinstance(exc, SchedulerHttpError):
        status, code, message, retryable = exc.status, exc.code, exc.message, exc.retryable
    elif isinstance(exc, (DuplicateRequest, IdempotencyConflict, OperationStale)):
        status, code, message, retryable = 409, type(exc).__name__.lower(), "operation conflicts with durable state", False
    elif isinstance(exc, QueueError):
        status, code, message, retryable = 404 if "unknown" in str(exc) else 409, "queue_error", "queue operation was rejected", False
    elif isinstance(exc, (ValueError, SessionError)):
        status, code, message, retryable = 400, "invalid_request", "request could not be accepted", False
    else:
        status, code, message, retryable = 500, "scheduler_failure", "scheduler operation failed", True
    return web.json_response({"code": code, "message": message, "retryable": retryable}, status=status)


class SchedulerHttpServer:
    def __init__(self, schedulers: dict[str, QueueScheduler], *, max_body: int = MAX_BODY,
                 max_frame: int = MAX_FRAME, max_watches: int = 16,
                 heartbeat: float = 15.0, write_timeout: float = 5.0):
        if not schedulers or any(not isinstance(k, str) or not k or not isinstance(v, QueueScheduler) for k, v in schedulers.items()):
            raise ValueError("schedulers must be a named scheduler registry")
        if min(max_body, max_frame, max_watches) < 1 or not all(math.isfinite(x) and x > 0 for x in (heartbeat, write_timeout)):
            raise ValueError("invalid HTTP bounds")
        self.schedulers, self.max_body, self.max_frame = schedulers, max_body, max_frame
        self.heartbeat, self.write_timeout = heartbeat, write_timeout
        self.watches = asyncio.BoundedSemaphore(max_watches)
        self.app = web.Application(client_max_size=max_body)
        self.app.add_routes([
            web.post('/schedulers/{schedulerId}/start', self.start),
            web.post('/schedulers/{schedulerId}/requests', self.enqueue),
            web.get('/schedulers/{schedulerId}/requests/{requestId}', self.get),
            web.get('/schedulers/{schedulerId}/requests/{requestId}/watch', self.watch),
            web.post('/schedulers/{schedulerId}/requests/{requestId}/cancel', self.cancel),
            web.post('/schedulers/{schedulerId}/stop', self.stop),
        ])

    def _scheduler(self, request) -> QueueScheduler:
        scheduler_id = _text(request.match_info.get('schedulerId'), 'schedulerId')
        scheduler = self.schedulers.get(scheduler_id)
        if scheduler is None:
            raise SchedulerHttpError(404, 'unknown_scheduler', 'scheduler is not configured')
        return scheduler

    async def _body(self, request, allowed: set[str]) -> dict[str, Any]:
        if request.content_length is not None and request.content_length > self.max_body:
            raise SchedulerHttpError(413, 'body_too_large', 'request body exceeds limit')
        chunks, total = [], 0
        async for chunk in request.content.iter_chunked(64 * 1024):
            total += len(chunk)
            if total > self.max_body:
                raise SchedulerHttpError(413, 'body_too_large', 'request body exceeds limit')
            chunks.append(chunk)
        try: value = _strict(b''.join(chunks))
        except (ValueError, json.JSONDecodeError):
            raise SchedulerHttpError(400, 'invalid_json', 'invalid strict JSON')
        if not isinstance(value, dict) or any(k not in allowed for k in value):
            raise SchedulerHttpError(400, 'invalid_request', 'request contains unknown fields')
        return value

    def _model(self, scheduler, body) -> None:
        try: model = ModelId(_text(body.get('modelId'), 'modelId'))
        except ValueError as exc: raise SchedulerHttpError(400, 'unknown_model', 'model is not configured') from exc
        if model != scheduler.store.model_id:
            raise SchedulerHttpError(400, 'model_mismatch', 'model does not match scheduler')

    @staticmethod
    def _wire(scheduler, row) -> dict[str, Any]:
        return scheduler.store._projection(row['request_id'])

    @staticmethod
    def _descriptor(value: Any) -> FunctionDescriptor:
        if not isinstance(value, dict) or set(value) != {'name', 'args', 'dependencyResultIds'}:
            raise SchedulerHttpError(400, 'invalid_request', 'malformed function descriptor')
        if not isinstance(value['args'], dict):
            raise SchedulerHttpError(400, 'invalid_request', 'descriptor args must be an object')
        ids = value['dependencyResultIds']
        if not isinstance(ids, list) or any(not isinstance(item, str) or not item for item in ids):
            raise SchedulerHttpError(400, 'invalid_request', 'descriptor dependencyResultIds must be text array')
        return FunctionDescriptor(_text(value['name'], 'descriptor name'), value['args'], tuple(ids))

    async def start(self, request):
        try:
            scheduler = self._scheduler(request)
            body = await self._body(request, {'schedulerId', 'modelId'})
            if body.get('schedulerId') != request.match_info['schedulerId']: raise SchedulerHttpError(400, 'scheduler_mismatch', 'body schedulerId differs from path')
            self._model(scheduler, body)
            key = _text(request.headers.get('Idempotency-Key'), 'Idempotency-Key')
            session = await scheduler.start(idempotency_key=key)
            return web.json_response({'schedulerId': scheduler.store.scheduler_id, 'modelId': scheduler.store.model_id.value, 'sessionToken': session.session_token, 'residencyGeneration': session.generation}, status=200)
        except Exception as exc: return _error(exc)

    async def enqueue(self, request):
        try:
            scheduler = self._scheduler(request); body = await self._body(request, {'schedulerId','requestId','modelId','payloadReference','dependencies','insertionMode','resultTarget','ready','template'})
            if body.get('schedulerId') != request.match_info['schedulerId']: raise SchedulerHttpError(400, 'scheduler_mismatch', 'body schedulerId differs from path')
            self._model(scheduler, body)
            dependencies = body.get('dependencies', [])
            if not isinstance(dependencies, list) or any(not isinstance(item, str) or not item for item in dependencies):
                raise SchedulerHttpError(400, 'invalid_request', 'dependencies must be an array of non-empty text')
            result_target = body.get('resultTarget', 'local')
            _text(result_target, 'resultTarget')
            descriptors = {}
            for name in ('ready', 'template'):
                if name in body:
                    descriptors[name] = self._descriptor(body[name])
            key = _text(request.headers.get('Idempotency-Key'), 'Idempotency-Key')
            try: mode = InsertionMode(body.get('insertionMode', 'append'))
            except (TypeError, ValueError) as exc: raise SchedulerHttpError(400, 'invalid_request', 'invalid insertionMode') from exc
            row = await scheduler.enqueue(_text(body.get('requestId'), 'requestId'), _text(body.get('payloadReference'), 'payloadReference'), dependencies, mode, result_target, key, **descriptors)
            return web.json_response(self._wire(scheduler, row), status=201)
        except Exception as exc: return _error(exc)

    async def get(self, request):
        try:
            scheduler = self._scheduler(request); row = await scheduler.get(_text(request.match_info['requestId'], 'requestId'))
            if not row: raise SchedulerHttpError(404, 'unknown_request', 'request is not known')
            return web.json_response(self._wire(scheduler, row))
        except Exception as exc: return _error(exc)

    async def cancel(self, request):
        try:
            scheduler = self._scheduler(request); key = _text(request.headers.get('Idempotency-Key'), 'Idempotency-Key')
            row = await scheduler.cancel(_text(request.match_info['requestId'], 'requestId'), idempotency_key=key)
            return web.json_response(self._wire(scheduler, row))
        except Exception as exc: return _error(exc)

    async def stop(self, request):
        try:
            scheduler = self._scheduler(request); body = await self._body(request, {'outcome','reason'})
            if 'outcome' not in body:
                raise SchedulerHttpError(400, 'invalid_request', 'outcome is required')
            outcome = body['outcome']
            if outcome not in ('error', 'cancelled'): raise SchedulerHttpError(400, 'invalid_request', 'outcome must be error or cancelled')
            reason = body.get('reason', 'stopped')
            _text(reason, 'reason')
            result = await scheduler.stop(reason, cancelled=outcome == 'cancelled', idempotency_key=_text(request.headers.get('Idempotency-Key'), 'Idempotency-Key'))
            return web.json_response({'accepted': True, 'cancelled': outcome == 'cancelled', 'affected': result})
        except Exception as exc: return _error(exc)

    async def watch(self, request):
        response = iterator = None; acquired = False
        try:
            scheduler = self._scheduler(request); request_id = _text(request.match_info['requestId'], 'requestId')
            raw = request.headers.get('Last-Event-ID', '0')
            if not raw.isdigit(): raise SchedulerHttpError(400, 'invalid_cursor', 'Last-Event-ID must be nonnegative integer')
            cursor = int(raw); high = scheduler.store.event_highwater()
            if cursor > high: raise SchedulerHttpError(409, 'cursor_future', 'cursor is ahead of durable history')
            if self.watches.locked(): raise SchedulerHttpError(429, 'watch_limit', 'too many watchers', True)
            await self.watches.acquire(); acquired = True
            current = scheduler.store._projection(request_id)
            if scheduler.store.request_history_has_legacy_event(request_id, cursor):
                raise SchedulerHttpError(409, 'legacy_event_unavailable', 'historical event has no immutable snapshot')
            rows = scheduler.store.request_events(request_id, cursor, EVENT_BATCH)
            def frame_for(row):
                data = json.loads(row['data']); snapshot = data['snapshot']; payload = json.dumps(snapshot, allow_nan=False, separators=(',', ':'))
                frame = f"id: {row['cursor']}\nevent: {row['kind']}\ndata: {payload}\n\n".encode()
                if len(frame) > self.max_frame: raise SchedulerHttpError(413, 'frame_too_large', 'SSE frame exceeds limit')
                return frame
            # Validate the first pending frame before SSE headers make a JSON
            # 413 impossible. Later frames remain bounded before every write.
            first_frame = frame_for(rows[0]) if rows else None
            response = web.StreamResponse(headers={'Content-Type':'text/event-stream','Cache-Control':'no-cache'})
            await response.prepare(request)
            async def send(row, frame=None):
                frame = frame_for(row) if frame is None else frame
                await asyncio.wait_for(response.write(frame), self.write_timeout)
            while rows:
                for index, row in enumerate(rows):
                    cursor = row['cursor']; await send(row, first_frame if index == 0 else None)
                first_frame = None
                if json.loads(rows[-1]['data'])['snapshot']['status'] in ('done', 'error', 'cancelled'):
                    return response
                rows = scheduler.store.request_events(request_id, cursor, EVENT_BATCH)
            if current['status'] in ('done', 'error', 'cancelled'):
                return response
            next_heartbeat = asyncio.get_running_loop().time() + self.heartbeat
            while True:
                row = scheduler.store.request_events(request_id, cursor, EVENT_BATCH)
                if row:
                    for item in row: cursor = item['cursor']; await send(item)
                    if json.loads(row[-1]['data'])['snapshot']['status'] in ('done','error','cancelled'): break
                else:
                    now = asyncio.get_running_loop().time()
                    if now >= next_heartbeat:
                        await asyncio.wait_for(response.write(b': heartbeat\n\n'), self.write_timeout)
                        next_heartbeat = now + self.heartbeat
                    await asyncio.sleep(min(.05, max(0, next_heartbeat - now)))
            return response
        except ConnectionResetError:
            return response
        except asyncio.CancelledError: raise
        except Exception as exc:
            if response is not None and response.prepared: return response
            return _error(exc)
        finally:
            if acquired: self.watches.release()


class SchedulerHttpClient:
    def __init__(self, base_url: str, *, timeout: float = 30, max_frame: int = MAX_FRAME):
        self.base_url, self.timeout, self.max_frame = base_url.rstrip('/'), timeout, max_frame
    @staticmethod
    def _p(value): return quote(_text(value, 'path segment'), safe='')
    async def _response_json(self, response):
        data = await response.content.read(self.max_frame + 1)
        if len(data) > self.max_frame:
            raise SchedulerHttpError(502, 'response_too_large', 'HTTP response exceeds limit')
        try:
            value = _strict(data)
        except (ValueError, json.JSONDecodeError) as exc:
            raise SchedulerHttpError(502, 'invalid_response', 'HTTP response is not strict JSON') from exc
        if not isinstance(value, dict):
            raise SchedulerHttpError(502, 'invalid_response', 'HTTP response has invalid shape')
        return value
    @staticmethod
    def _error_value(status, value):
        if set(value) != {'code', 'message', 'retryable'} or not isinstance(value['code'], str) or not isinstance(value['message'], str) or not isinstance(value['retryable'], bool):
            raise SchedulerHttpError(502, 'invalid_response', 'HTTP error response has invalid shape')
        return SchedulerHttpError(status, value['code'], value['message'], value['retryable'])
    async def _json(self, method, path, body=None, key=None):
        headers = {} if key is None else {'Idempotency-Key': _text(key, 'Idempotency-Key')}
        async with ClientSession(timeout=ClientTimeout(total=self.timeout)) as session:
            async with session.request(method, self.base_url + path, json=body, headers=headers) as response:
                value = await self._response_json(response)
                if response.status >= 400: raise self._error_value(response.status, value)
                return value
    async def start(self, scheduler_id, model_id, *, idempotency_key): return await self._json('POST', f'/schedulers/{self._p(scheduler_id)}/start', {'schedulerId':scheduler_id,'modelId':ModelId(model_id).value}, idempotency_key)
    async def enqueue(self, scheduler_id, body, *, idempotency_key): return await self._json('POST', f'/schedulers/{self._p(scheduler_id)}/requests', body, idempotency_key)
    async def get(self, scheduler_id, request_id): return await self._json('GET', f'/schedulers/{self._p(scheduler_id)}/requests/{self._p(request_id)}')
    async def cancel(self, scheduler_id, request_id, *, idempotency_key): return await self._json('POST', f'/schedulers/{self._p(scheduler_id)}/requests/{self._p(request_id)}/cancel', None, idempotency_key)
    async def stop(self, scheduler_id, *, outcome='error', reason='stopped', idempotency_key): return await self._json('POST', f'/schedulers/{self._p(scheduler_id)}/stop', {'outcome':outcome,'reason':reason}, idempotency_key)

    async def watch(self, scheduler_id, request_id, *, after: int = 0) -> AsyncIterator[dict[str, Any]]:
        if isinstance(after, bool) or not isinstance(after, int) or after < 0: raise ValueError('after must be non-negative')
        async with ClientSession(timeout=ClientTimeout(total=None)) as session:
            async with session.get(self.base_url + f'/schedulers/{self._p(scheduler_id)}/requests/{self._p(request_id)}/watch', headers={'Last-Event-ID': str(after)}) as response:
                if response.status >= 400:
                    raise self._error_value(response.status, await self._response_json(response))
                buffer = b''
                cursor = after
                async for chunk in response.content.iter_chunked(64 * 1024):
                    buffer += chunk
                    while b'\n\n' in buffer:
                        frame, buffer = buffer.split(b'\n\n', 1)
                        if len(frame) > self.max_frame:
                            raise SchedulerHttpError(502, 'frame_too_large', 'SSE frame exceeds limit')
                        lines = frame.split(b'\n')
                        if frame.startswith(b':'):
                            if any(line and not line.startswith(b':') for line in lines):
                                raise SchedulerHttpError(502, 'invalid_sse', 'SSE frame has invalid shape')
                            continue
                        fields = {}
                        for line in lines:
                            if line.count(b': ') != 1:
                                raise SchedulerHttpError(502, 'invalid_sse', 'SSE frame has invalid shape')
                            key, value = line.split(b': ', 1)
                            if key in fields:
                                raise SchedulerHttpError(502, 'invalid_sse', 'SSE frame has invalid shape')
                            fields[key] = value
                        if set(fields) != {b'id', b'event', b'data'}:
                            raise SchedulerHttpError(502, 'invalid_sse', 'SSE frame has invalid shape')
                        try:
                            cursor = fields[b'id'].decode('ascii')
                            event = fields[b'event'].decode('ascii')
                            numeric_cursor = int(cursor)
                            if not cursor.isdigit() or not event or numeric_cursor <= after:
                                raise ValueError
                            payload = _strict(fields[b'data'])
                            if (not isinstance(payload, dict) or payload.get('sequence') != numeric_cursor
                                    or payload.get('requestId') != request_id or payload.get('schedulerId') != scheduler_id):
                                raise ValueError
                        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                            raise SchedulerHttpError(502, 'invalid_sse', 'SSE frame has invalid shape')
                        yield {'id': cursor, 'event': event, 'data': payload}
                        after = numeric_cursor
                    if len(buffer) > self.max_frame:
                        raise SchedulerHttpError(502, 'frame_too_large', 'SSE frame exceeds limit')
                if buffer:
                    raise SchedulerHttpError(502, 'truncated_sse', 'SSE stream ended mid-frame')
