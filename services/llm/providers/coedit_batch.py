"""Bounded native-batch collection for CoEdIT."""
from __future__ import annotations
import asyncio
import json
import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class AllocatorObservation:
    baseline_allocated: int
    baseline_reserved: int
    peak_allocated: int
    peak_reserved: int
    final_allocated: int
    final_reserved: int

    def valid(self) -> bool:
        values = (self.baseline_allocated, self.baseline_reserved, self.peak_allocated,
                  self.peak_reserved, self.final_allocated, self.final_reserved)
        return (all(type(value) is int and value >= 0 for value in values)
                and self.baseline_allocated <= self.baseline_reserved
                and self.peak_allocated <= self.peak_reserved
                and self.final_allocated <= self.final_reserved
                and self.peak_allocated >= self.baseline_allocated
                and self.peak_reserved >= self.baseline_reserved
                and self.peak_allocated >= self.final_allocated
                and self.peak_reserved >= self.final_reserved)


@dataclass(frozen=True)
class NativeBatchObservation:
    batch_size: int
    execution_started: int
    execution_ended: int
    cuda_synchronized: bool
    allocator: AllocatorObservation
    decoder_steps: tuple[int, ...] = ()
    max_output_tokens: int = 0
    request_ids: tuple[str, ...] = ()


class CoEdITBatcher:
    MAX_NATIVE_BATCH_SIZE = 32
    _MAX_RPC_ID = 2 ** 63 - 1
    def __init__(self, worker, maximum: int, delay: float, expected_max_output_tokens: int):
        if type(maximum) is not int or not 1 <= maximum <= self.MAX_NATIVE_BATCH_SIZE:
            raise ValueError(f"native batch maximum must be between 1 and {self.MAX_NATIVE_BATCH_SIZE}")
        if not isinstance(delay, (int, float)) or isinstance(delay, bool) or not math.isfinite(delay) or not 0 <= delay <= 1:
            raise ValueError("native batch delay must be between zero and one second")
        if type(expected_max_output_tokens) is not int or expected_max_output_tokens < 1:
            raise ValueError("expected output maximum must be a positive integer")
        self.worker, self.maximum, self.delay = worker, maximum, delay
        self.expected_max_output_tokens = expected_max_output_tokens
        frame_limit = getattr(worker, "frame_limit", 256 * 1024)
        if type(frame_limit) is not int or frame_limit <= 0:
            raise ValueError("worker frame limit must be a positive integer")
        self.frame_limit = frame_limit
        self.pending = []
        self.timer = None
        self.running = None
        self.active = []
        self.closed = False
        self.generation = 0
        # Observations are diagnostic evidence, not an unbounded second request
        # queue.  Retain a bounded recent window and expose every eviction.
        self.observations = deque(maxlen=maximum)
        self.dropped_observations = 0

    async def submit(self, request_id, instruction, text):
        candidate = {"id": self._MAX_RPC_ID, "op": "execute_batch", "items": [
            {"instruction": instruction, "texts": [text]}]}
        if self._encoded_size(candidate) > self.frame_limit:
            raise ValueError("CoEdIT batch item exceeds RPC frame bound")
        if self.closed or len(self.pending) + len(self.active) >= self.maximum:
            raise RuntimeError("CoEdIT native batch queue is full")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request ID is required")
        future = asyncio.get_running_loop().create_future()
        item = (request_id, instruction, text, future)
        self.pending.append(item)
        if len(self.pending) == self.maximum:
            await self._flush()
        elif self.timer is None:
            self.timer = asyncio.create_task(self._delay_flush())
        try:
            return await future
        except asyncio.CancelledError:
            self.pending = [x for x in self.pending if x[3] is not future]
            future.cancel()
            raise

    async def _delay_flush(self):
        await asyncio.sleep(self.delay)
        await self._flush()

    async def _flush(self):
        if self.timer is not None and self.timer is not asyncio.current_task():
            task, self.timer = self.timer, None
            if not task.done(): task.cancel()
        elif self.timer is asyncio.current_task():
            self.timer = None
        if self.running is not None or not self.pending or self.closed:
            return
        items = []
        for item in self.pending[:self.maximum]:
            candidate = {"id": self._MAX_RPC_ID, "op": "execute_batch", "items": [
                {"instruction": instruction, "texts": [text]}
                for _, instruction, text, _ in (*items, item)]}
            if self._encoded_size(candidate) > self.frame_limit:
                break
            items.append(item)
        if not items:
            raise RuntimeError("CoEdIT batch item exceeds RPC frame bound")
        del self.pending[:len(items)]
        self.active = items
        generation = self.generation

        async def run():
            try:
                value = await self.worker.call("execute_batch", items=[{"instruction": i, "texts": [t]} for _, i, t, _ in items])
                if not isinstance(value, dict) or set(value) != {"outputs", "observation"}:
                    raise RuntimeError("malformed CoEdIT batch response")
                outputs, obs = value["outputs"], value["observation"]
                if not isinstance(outputs, list) or len(outputs) != len(items):
                    raise RuntimeError("invalid CoEdIT batch output cardinality")
                required = {"batch_size", "execution_started", "execution_ended", "cuda_synchronized", "allocator", "decoder_steps", "max_output_tokens"}
                if (not isinstance(obs, dict) or set(obs) != required or type(obs["batch_size"]) is not int or obs["batch_size"] != len(items)
                        or type(obs["cuda_synchronized"]) is not bool
                        or type(obs["execution_started"]) is not int or type(obs["execution_ended"]) is not int
                        or obs["execution_started"] < 0 or obs["execution_ended"] < 0
                        or obs["cuda_synchronized"] is not True
                        or obs["execution_ended"] < obs["execution_started"]):
                    raise RuntimeError("malformed CoEdIT batch observation")
                allocator_values = obs["allocator"]
                allocator_keys = {"baseline_allocated", "baseline_reserved", "peak_allocated", "peak_reserved", "final_allocated", "final_reserved"}
                if not isinstance(allocator_values, dict) or set(allocator_values) != allocator_keys:
                    raise RuntimeError("malformed CoEdIT allocator observation")
                allocator = AllocatorObservation(*(allocator_values[key] for key in ("baseline_allocated", "baseline_reserved", "peak_allocated", "peak_reserved", "final_allocated", "final_reserved")))
                if not allocator.valid():
                    raise RuntimeError("inconsistent CoEdIT allocator observation")
                steps = obs["decoder_steps"]
                maximum = obs["max_output_tokens"]
                if (type(maximum) is not int or maximum != self.expected_max_output_tokens
                        or not isinstance(steps, list) or len(steps) != len(items)
                        or any(type(value) is not int or value < 0 or value > maximum for value in steps)):
                    raise RuntimeError("malformed CoEdIT decoder workload observation")
                observation = NativeBatchObservation(len(items), obs["execution_started"], obs["execution_ended"], obs["cuda_synchronized"], allocator, tuple(steps), maximum, tuple(item[0] for item in items))
                for (_, _, _, future), output in zip(items, outputs):
                    if generation == self.generation and not future.done():
                        future.set_result((output, observation))
                if generation == self.generation:
                    if len(self.observations) == self.observations.maxlen:
                        self.dropped_observations += 1
                    self.observations.append({"batch_size": observation.batch_size, "execution_started": observation.execution_started, "execution_ended": observation.execution_ended, "cuda_synchronized": observation.cuda_synchronized, "allocator": observation.allocator, "decoder_steps": observation.decoder_steps, "max_output_tokens": observation.max_output_tokens, "request_ids": tuple(item[0] for item in items)})
            except BaseException as exc:
                for _, _, _, future in items:
                    if not future.done(): future.set_exception(exc)
            finally:
                self.active = []
                self.running = None
                if self.pending and not self.closed:
                    await self._flush()
        self.running = asyncio.create_task(run())

    async def close(self):
        self.fence()
        await self.join()

    def fence(self):
        self.closed = True
        self.generation += 1
        if self.timer is not None:
            self.timer.cancel()
        for _, _, _, future in self.pending:
            if not future.done(): future.set_exception(RuntimeError("CoEdIT batcher closed"))
        self.pending.clear()
        for _, _, _, future in self.active:
            if not future.done(): future.set_exception(RuntimeError("CoEdIT batcher closed"))

    async def join(self):
        if self.timer is not None:
            timer, self.timer = self.timer, None
            await asyncio.gather(timer, return_exceptions=True)
        if self.running is not None:
            await self.running

    def drain_observations(self):
        values = tuple(self.observations)
        self.observations.clear()
        return values

    def cancel(self, request_id):
        for item in self.pending:
            if item[0] == request_id and not item[3].done():
                item[3].set_exception(asyncio.CancelledError())
        self.pending = [item for item in self.pending if item[0] != request_id]
        for item in self.active:
            if item[0] == request_id and not item[3].done():
                item[3].set_exception(asyncio.CancelledError())

    @staticmethod
    def _encoded_size(value):
        return len(json.dumps(value, separators=(",", ":"), ensure_ascii=True,
                              allow_nan=False).encode()) + 4
