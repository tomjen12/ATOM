# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Low-overhead GPU envelope timing for long-running serving workloads.

This deliberately uses CUDA/HIP events instead of the PyTorch profiler. Event
resolution and JSONL I/O happen on a background thread, so the model thread
only records events and enqueues one item per scheduler batch.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import torch

from atom.utils import envs

logger = logging.getLogger("atom")


@dataclass
class _ForwardTiming:
    mode: str
    role: str
    input_token_count: int
    sequence_count: int
    start: torch.cuda.Event
    end: torch.cuda.Event | None = None


@dataclass
class _BatchTiming:
    batch_id: int
    replay_case_id: int | None
    is_dummy: bool
    num_prefill_tokens: int
    num_decode_tokens: int
    num_sequences: int
    host_start_ns: int
    host_end_ns: int | None
    start_events: list[torch.cuda.Event]
    end_events: list[torch.cuda.Event] = field(default_factory=list)
    forwards: list[_ForwardTiming] = field(default_factory=list)


class GpuTimingRecorder:
    """Record one GPU envelope and its target/draft forwards per runner batch."""

    def __init__(self, runner: Any, output_dir: str):
        self.runner = runner
        self.output_dir = output_dir
        self._next_batch_id = 0
        self._active_batch: _BatchTiming | None = None
        self._queue: queue.Queue[_BatchTiming | None] = queue.Queue()
        self._closed = False

        os.makedirs(output_dir, exist_ok=True)
        pp_rank = runner.config.parallel_config.pipeline_parallel_rank
        dp_rank = runner.config.parallel_config.data_parallel_rank
        filename = (
            f"gpu_timing_dp{dp_rank}_pp{pp_rank}_tp{runner.rank}_"
            f"pid{os.getpid()}.jsonl"
        )
        self.path = os.path.join(output_dir, filename)
        self._file = open(self.path, "a", encoding="utf-8", buffering=1)

        self._anchor = torch.cuda.Event(enable_timing=True)
        self._anchor.record(torch.cuda.current_stream(runner.device))
        self._anchor_epoch_ns = time.time_ns()
        for stream in self._streams():
            if stream != torch.cuda.current_stream(runner.device):
                stream.wait_event(self._anchor)

        self._writer = threading.Thread(
            target=self._writer_loop,
            name="atom-gpu-timing-writer",
            daemon=True,
        )
        self._writer.start()
        logger.info("GPU envelope timing enabled: %s", self.path)

    def _streams(self) -> list[torch.cuda.Stream]:
        streams = [torch.cuda.current_stream(self.runner.device)]
        copy_stream = getattr(
            getattr(self.runner, "tokenID_processor", None),
            "async_copy_stream",
            None,
        )
        if copy_stream is not None and copy_stream not in streams:
            streams.append(copy_stream)
        return streams

    @staticmethod
    def _event(stream: torch.cuda.Stream) -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def begin_batch(self, batch: Any) -> None:
        if self._active_batch is not None:
            logger.warning("Discarding unfinished GPU timing batch")
        self._next_batch_id += 1
        self._active_batch = _BatchTiming(
            batch_id=self._next_batch_id,
            replay_case_id=getattr(batch, "replay_case_id", None),
            is_dummy=bool(getattr(batch, "is_dummy_run", False)),
            num_prefill_tokens=int(getattr(batch, "total_tokens_num_prefill", 0)),
            num_decode_tokens=int(getattr(batch, "total_tokens_num_decode", 0)),
            num_sequences=int(getattr(batch, "total_seqs_num", 0)),
            host_start_ns=time.time_ns(),
            host_end_ns=None,
            start_events=[self._event(stream) for stream in self._streams()],
        )

    def end_batch(self) -> None:
        timing = self._active_batch
        self._active_batch = None
        if timing is None:
            return
        timing.host_end_ns = time.time_ns()
        timing.end_events = [self._event(stream) for stream in self._streams()]
        self._queue.put(timing)

    def begin_forward(
        self,
        mode: str,
        role: str,
        input_token_count: int,
        sequence_count: int,
    ) -> _ForwardTiming | None:
        batch = self._active_batch
        if batch is None:
            return None
        timing = _ForwardTiming(
            mode=mode,
            role=role,
            input_token_count=input_token_count,
            sequence_count=sequence_count,
            start=self._event(torch.cuda.current_stream(self.runner.device)),
        )
        batch.forwards.append(timing)
        return timing

    def end_forward(self, timing: _ForwardTiming | None) -> None:
        if timing is not None:
            timing.end = self._event(torch.cuda.current_stream(self.runner.device))

    def _offset_ms(self, event: torch.cuda.Event) -> float:
        return float(self._anchor.elapsed_time(event))

    def _serialize(self, timing: _BatchTiming) -> dict[str, Any]:
        for event in timing.end_events:
            event.synchronize()

        start_offsets = [self._offset_ms(event) for event in timing.start_events]
        end_offsets = [self._offset_ms(event) for event in timing.end_events]
        forwards = []
        for forward in timing.forwards:
            if forward.end is None:
                continue
            start_ms = self._offset_ms(forward.start)
            end_ms = self._offset_ms(forward.end)
            forwards.append(
                {
                    "mode": forward.mode,
                    "role": forward.role,
                    "input_token_count": forward.input_token_count,
                    "sequence_count": forward.sequence_count,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "start_epoch_ns": self._anchor_epoch_ns
                    + round(start_ms * 1_000_000),
                    "end_epoch_ns": self._anchor_epoch_ns
                    + round(end_ms * 1_000_000),
                }
            )
        gpu_start_ms = min(start_offsets)
        gpu_end_ms = max(end_offsets)
        return {
            "schema_version": 2,
            "batch_id": timing.batch_id,
            "replay_case_id": timing.replay_case_id,
            "is_dummy": timing.is_dummy,
            "num_prefill_tokens": timing.num_prefill_tokens,
            "num_decode_tokens": timing.num_decode_tokens,
            "num_sequences": timing.num_sequences,
            "host_start_ns": timing.host_start_ns,
            "host_end_ns": timing.host_end_ns,
            "gpu_start_ms": gpu_start_ms,
            "gpu_end_ms": gpu_end_ms,
            "gpu_start_epoch_ns": self._anchor_epoch_ns
            + round(gpu_start_ms * 1_000_000),
            "gpu_end_epoch_ns": self._anchor_epoch_ns
            + round(gpu_end_ms * 1_000_000),
            "stream_start_ms": start_offsets,
            "stream_end_ms": end_offsets,
            "forwards": forwards,
        }

    def _writer_loop(self) -> None:
        while True:
            timing = self._queue.get()
            if timing is None:
                break
            try:
                self._file.write(
                    json.dumps(self._serialize(timing), separators=(",", ":")) + "\n"
                )
            except Exception:
                logger.exception("Failed to write GPU timing batch")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._writer.join()
        self._file.close()


def create_gpu_timing_recorder(runner: Any) -> GpuTimingRecorder | None:
    """Create a recorder on one representative TP rank when explicitly enabled."""

    output_dir = os.getenv("ATOM_GPU_TIMING_DIR")
    if not envs.ATOM_WORKLOAD_RECORD_ALL or not output_dir or runner.rank != 0:
        return None
    return GpuTimingRecorder(runner, output_dir)


class HostTimingRecorder:
    """Write EngineCore host-stage intervals on a background thread."""

    def __init__(self, output_dir: str, dp_rank: int, pp_rank: int):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(
            output_dir,
            f"scheduler_host_timing_dp{dp_rank}_pp{pp_rank}_pid{os.getpid()}.jsonl",
        )
        self._file = open(self.path, "a", encoding="utf-8", buffering=1)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._idle_start_ns: int | None = None
        self._closed = False
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="atom-host-timing-writer",
            daemon=True,
        )
        self._writer.start()
        logger.info("Scheduler host timing enabled: %s", self.path)

    def record(self, stage: str, start_ns: int, end_ns: int) -> None:
        if end_ns <= start_ns:
            return
        self._queue.put(
            {
                "schema_version": 1,
                "stage": stage,
                "start_ns": start_ns,
                "end_ns": end_ns,
            }
        )

    def start_idle(self) -> None:
        if self._idle_start_ns is None:
            self._idle_start_ns = time.time_ns()

    def end_idle(self) -> None:
        if self._idle_start_ns is not None:
            end_ns = time.time_ns()
            self.record("scheduler_idle_or_stalled", self._idle_start_ns, end_ns)
            self._idle_start_ns = None

    def _writer_loop(self) -> None:
        while True:
            record = self._queue.get()
            if record is None:
                break
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.end_idle()
        self._queue.put(None)
        self._writer.join()
        self._file.close()


def create_host_timing_recorder(config: Any) -> HostTimingRecorder | None:
    output_dir = os.getenv("ATOM_GPU_TIMING_DIR")
    parallel = config.parallel_config
    if (
        not envs.ATOM_WORKLOAD_RECORD_ALL
        or not output_dir
        or parallel.data_parallel_rank != 0
    ):
        return None
    return HostTimingRecorder(
        output_dir,
        parallel.data_parallel_rank,
        parallel.pipeline_parallel_rank,
    )


def gpu_timing_batch_method(method):
    @wraps(method)
    def wrapped(self, batch, *args, **kwargs):
        prefill_only_trace = envs.ATOM_EXTEND_TRACE
        collect_trace = bool(
            int(getattr(batch, "total_tokens_num_prefill", 0)) > 0
            and int(getattr(batch, "total_tokens_num_decode", 0)) == 0
            and int(getattr(batch, "total_seqs_num_decode", 0)) == 0
        )
        start_extend_trace = getattr(
            self, "_start_extend_trace_profiler", None
        )
        stop_extend_trace = getattr(
            self, "_stop_extend_trace_profiler", None
        )
        extend_trace_started = False
        if (
            prefill_only_trace
            and collect_trace
            and start_extend_trace is not None
        ):
            extend_trace_started = bool(start_extend_trace(batch))
        recorder = getattr(self, "_gpu_timing_recorder", None)
        if recorder is not None:
            recorder.begin_batch(batch)
        try:
            return method(self, batch, *args, **kwargs)
        finally:
            try:
                if recorder is not None:
                    recorder.end_batch()
            finally:
                try:
                    advance_profiler = getattr(
                        self, "_advance_runtime_trace_profiler", None
                    )
                    if advance_profiler is not None and not prefill_only_trace:
                        advance_profiler()
                finally:
                    if extend_trace_started and stop_extend_trace is not None:
                        stop_extend_trace()

    return wrapped


def gpu_timing_forward_method(role: str):
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            recorder = getattr(self, "_gpu_timing_recorder", None)
            if recorder is None:
                return method(self, *args, **kwargs)

            batch = kwargs.get("batch")
            if batch is None and len(args) >= 2 and role == "target":
                batch = args[1]
            if batch is None and args and role == "draft":
                batch = args[0]
            is_prefill = bool(
                batch is not None
                and getattr(batch, "total_tokens_num_prefill", 0) > 0
            )
            mode = (
                "TARGET_VERIFY"
                if role == "draft"
                else ("EXTEND" if is_prefill else "TARGET_VERIFY")
            )
            sequence_count = int(getattr(batch, "total_seqs_num", 0))
            if role == "draft":
                mtp_k = int(getattr(getattr(self, "drafter", None), "mtp_k", 0))
                input_token_count = sequence_count * mtp_k
            else:
                input_token_count = int(getattr(batch, "total_tokens_num", 0))
            timing = recorder.begin_forward(
                mode,
                role,
                input_token_count,
                sequence_count,
            )
            try:
                return method(self, *args, **kwargs)
            finally:
                recorder.end_forward(timing)

        return wrapped

    return decorate
