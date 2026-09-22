# SPDX-License-Identifier: MIT

"""Low-overhead capture of native ATOM prefill batches for offline replay."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import Sequence, SequenceType
from atom.model_engine.state_runtime import StateMaintenanceOps
from atom.sampling_params import SamplingParams
from atom.utils import envs

logger = logging.getLogger("atom")

_REPLAY_RUNTIME_ENV_NAMES = (
    "PYTHONHASHSEED",
    "LMCACHE_LOCAL_CPU",
    "LMCACHE_MAX_LOCAL_CPU_SIZE",
    "LMCACHE_CHUNK_SIZE",
    "LMCACHE_NUMA_MODE",
    "LMCACHE_LOCAL_DISK",
    "LMCACHE_MAX_LOCAL_DISK_SIZE",
    "LMCACHE_EC_PIN_TIMEOUT_SEC",
    "OFFLOAD_KV_FOR_HYBRID",
    "OFFLOAD_PROFILE",
    "OFFLOAD_GPU_STAGING_CHUNKS",
    "ATOM_NUMA_BIND",
    "ATOM_NUMA_NODE",
    "ATOM_AUTO_NUMA_BIND",
    "ATOM_STATE_CHECKPOINT_DEMAND",
    "ATOM_EXTEND_TRACE",
    "ATOM_PROFILER_MORE",
    "ATOM_PROFILER_RECORD_SHAPES",
    "ATOM_PROFILER_WITH_STACK",
    "ATOM_PROFILER_PROFILE_MEMORY",
    "AITER_LOG_LEVEL",
    "AITER_SITUV2_A4W4",
    "AITER_QUICK_REDUCE_QUANTIZATION",
    "AITER_FLYDSL_STAGE2_FP8",
    "PYTHONNOUSERSITE",
)


def build_synthetic_replay_sequences(
    case: dict[str, Any],
) -> dict[int, Sequence]:
    """Build logical sequences without reusing captured physical cache IDs."""
    seqs: dict[int, Sequence] = {}
    for request in case["requests"]:
        sampling = request["sampling"]
        params = SamplingParams(
            temperature=float(sampling["temperature"]),
            top_k=int(sampling["top_k"]),
            top_p=float(sampling["top_p"]),
            max_tokens=1,
            ignore_eos=True,
            logprobs=bool(sampling["return_logprobs"]),
        )
        request_id = int(request["request_id"])
        if request_id in seqs:
            raise ValueError(
                f"synthetic replay has duplicate request id {request_id}"
            )
        seq = Sequence(
            [int(token) for token in request["prompt_token_ids"]],
            block_size=int(request["block_size"]),
            sampling_params=params,
            id=request_id,
            request_id=str(request["external_request_id"]),
            has_per_req_cache=bool(
                request.get(
                    "requires_state_cache",
                    bool(request.get("state_slots")),
                )
            ),
        )
        seq.type = SequenceType.PREFILL
        seqs[request_id] = seq
    return seqs


def allocate_synthetic_replay_resources(
    block_manager: Any,
    seqs: dict[int, Sequence],
    case: dict[str, Any],
) -> None:
    """Allocate fresh local KV blocks and state slots with captured geometry."""
    for seq, request in zip(seqs.values(), case["requests"], strict=True):
        if request.get("allocate_kv_for_context", False):
            expected_blocks = int(
                block_manager.num_pool_blocks(
                    int(request["context_token_count"])
                )
            )
        else:
            expected_blocks = len(request["block_table"])
        if request.get("requires_state_cache", False):
            expected_slots = int(block_manager.state_slots_per_req)
        else:
            expected_slots = len(request.get("state_slots", []))
        if int(request["last_block_num_tokens"]) != seq.last_block_num_tokens:
            raise ValueError(
                "synthetic replay last-block geometry differs from capture"
            )
        if expected_slots:
            actual_width = int(block_manager.state_slots_per_req)
            if expected_slots != actual_width:
                raise ValueError(
                    "synthetic replay state-slot width differs from runtime: "
                    f"captured={expected_slots} configured={actual_width}"
                )
            if not block_manager.state.has_free(expected_slots):
                raise RuntimeError(
                    f"not enough state slots for replay request {seq.id}"
                )
            seq.state_slots = block_manager.state.pop_many(expected_slots)

        try:
            for _ in range(expected_blocks):
                seq.block_table.append(block_manager._fresh_block())
        except BaseException:
            block_manager.deallocate(seq)
            raise

        seq.num_cached_tokens = int(request["prefix_kv_token_count"])
        # Captured fork-source IDs are process-local. Preserve the branch and
        # index-tensor shape with a valid local alias; state values are outside
        # the performance-replay contract.
        seq.state_fork_src = (
            seq.state_slot
            if int(request.get("state_fork_src", -1)) >= 0
            else -1
        )


def release_synthetic_replay_resources(
    block_manager: Any,
    seqs: dict[int, Sequence],
) -> None:
    for seq in reversed(list(seqs.values())):
        block_manager.deallocate(seq)


def build_synthetic_replay_batch(
    case: dict[str, Any],
    seqs: dict[int, Sequence],
) -> ScheduledBatch:
    """Recreate captured ScheduledBatch geometry over fresh local resources."""
    requests = case["requests"]
    # Captured state-maintenance operations prepare process-local state values
    # before model execution. Synthetic performance replay intentionally uses
    # fresh state slots and does not compare state values or output values, so
    # do not recreate captured relocation/checkpoint operations here. The
    # ScheduledBatch default remains an empty StateMaintenanceOps instance.
    query_counts = [int(row["query_token_count"]) for row in requests]
    cached_counts = [int(row["prefix_kv_token_count"]) for row in requests]
    total_tokens = sum(query_counts)
    batch = ScheduledBatch(
        seqs=seqs,
        num_scheduled_tokens=query_counts,
        total_tokens_num=total_tokens,
        total_tokens_num_prefill=total_tokens,
        total_seqs_num=len(seqs),
        total_seqs_num_prefill=len(seqs),
        num_cached_tokens=cached_counts,
        is_final_chunk=[bool(row["is_final_chunk"]) for row in requests],
        next_token_ids=[
            int(row.get("next_token_id", -1)) for row in requests
        ],
    )
    expected = case["batch"]
    if [int(token) for token in batch.scheduled_tokens] != [
        int(token) for token in expected["scheduled_tokens"]
    ]:
        raise ValueError("synthetic replay scheduled token payload mismatch")
    if [int(value) for value in batch.context_lens] != [
        int(value) for value in expected["context_token_counts"]
    ]:
        raise ValueError("synthetic replay context geometry mismatch")
    batch.replay_case_id = int(case["case_id"])
    return batch


def _sampling_record(seq: Any) -> dict[str, Any]:
    return {
        "temperature": float(getattr(seq, "temperature", 0.0)),
        "top_k": int(getattr(seq, "top_k", -1)),
        "top_p": float(getattr(seq, "top_p", 1.0)),
        "return_logprobs": bool(getattr(seq, "return_logprobs", False)),
    }


def build_prefill_case_record(
    case_id: int,
    batch: Any,
    seqs: dict[int, Any],
) -> dict[str, Any]:
    """Snapshot one scheduled prefill batch before runtime mutates its sequences."""
    requests = []
    for index, (req_id, seq) in enumerate(seqs.items()):
        token_ids = [int(token) for token in seq.token_ids]
        requests.append(
            {
                "request_id": int(req_id),
                "external_request_id": str(
                    getattr(seq, "external_request_id", req_id)
                ),
                "prompt_token_ids": token_ids,
                "prompt_token_count": int(
                    getattr(seq, "num_prompt_tokens", len(token_ids))
                ),
                "block_size": int(seq.block_size),
                "query_token_count": int(batch.num_scheduled_tokens[index]),
                "prefix_kv_token_count": int(batch.num_cached_tokens[index]),
                "context_token_count": int(batch.context_lens[index]),
                "is_final_chunk": bool(batch.is_final_chunk[index]),
                "next_token_id": (
                    int(batch.next_token_ids[index])
                    if batch.next_token_ids is not None
                    else -1
                ),
                "block_table": [int(block) for block in seq.block_table],
                "last_block_num_tokens": int(seq.last_block_num_tokens),
                "state_slots": [
                    int(slot) for slot in getattr(seq, "state_slots", [])
                ],
                "state_fork_src": int(
                    getattr(seq, "state_fork_src", -1)
                ),
                "sampling": _sampling_record(seq),
            }
        )

    scheduled_tokens = [int(token) for token in batch.scheduled_tokens]
    state_ops = getattr(
        batch,
        "state_maintenance_ops",
        StateMaintenanceOps(),
    )
    query_tokens = [request["query_token_count"] for request in requests]
    prefix_tokens = [request["prefix_kv_token_count"] for request in requests]
    context_tokens = [request["context_token_count"] for request in requests]
    if sum(query_tokens) != len(scheduled_tokens):
        raise ValueError(
            "prefill replay capture token mismatch: "
            f"query_sum={sum(query_tokens)} scheduled={len(scheduled_tokens)}"
        )
    if any(
        prefix + query != context
        for prefix, query, context in zip(
            prefix_tokens, query_tokens, context_tokens
        )
    ):
        raise ValueError("prefill replay capture has inconsistent context lengths")

    record = {
        "schema_version": 1,
        "record_type": "atom_prefill_case",
        "status": "scheduled",
        "case_id": case_id,
        "timestamp_ns": time.time_ns(),
        "mode": "EXTEND",
        "batch": {
            "request_ids": [int(req_id) for req_id in batch.req_ids],
            "scheduled_tokens": scheduled_tokens,
            "query_token_counts": query_tokens,
            "prefix_kv_token_counts": prefix_tokens,
            "context_token_counts": context_tokens,
            "total_prefill_tokens": int(batch.total_tokens_num_prefill),
            "sequence_count": int(batch.total_seqs_num_prefill),
            "state_maintenance": {
                "relocation_count": len(state_ops.relocations),
                "checkpoint_store_count": len(state_ops.checkpoint_stores),
                "checkpoint_restore_count": len(
                    state_ops.checkpoint_restores
                ),
            },
        },
        "requests": requests,
        "moe_routes": {
            "case_id": case_id,
            "metadata": "moe_routes.json",
            "forwards": "moe_routes.jsonl",
            "counts": "moe_counts.uint16.bin",
        },
    }
    if envs.ATOM_EXTEND_TRACE:
        record["trace"] = {
            "rank": 0,
            "path": (
                "runtime_trace/rank_0/cases/"
                f"case_{case_id:06d}.trace.json.gz"
            ),
        }
    return record


def build_prefill_result_record(
    case_id: int, output: Any
) -> dict[str, Any]:
    token_ids = [
        [int(token) for token in row]
        for row in (getattr(output, "token_ids", None) or [])
    ]
    return {
        "schema_version": 1,
        "record_type": "atom_prefill_result",
        "status": "completed",
        "case_id": case_id,
        "timestamp_ns": time.time_ns(),
        "request_ids": [
            int(req_id) for req_id in getattr(output, "req_ids", [])
        ],
        "token_ids": token_ids,
        "is_deferred_output": bool(
            getattr(output, "is_deferred_out", False)
        ),
        "is_previous_prefill": bool(
            getattr(output, "is_prev_prefill", False)
        ),
    }


class PrefillReplayRecorder:
    """Asynchronously write replayable logical prefill cases and results."""

    _STOP = object()

    def __init__(self, output_dir: str, max_cases: int, config: Any):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "prefill_cases.jsonl"
        metadata_path = self.output_dir / "run_metadata.json"
        parallel = config.parallel_config
        speculative = getattr(config, "speculative_config", None)
        engine_args_json = os.getenv(
            "ATOM_PREFILL_REPLAY_ENGINE_ARGS_JSON", ""
        )
        engine_args = json.loads(engine_args_json) if engine_args_json else None
        resolved_config = {
            "max_model_len": int(getattr(config, "max_model_len", 0)),
            "kv_cache_dtype": str(getattr(config, "kv_cache_dtype", "")),
            "index_cache_dtype": getattr(config, "index_cache_dtype", None),
            "block_size": int(getattr(config, "kv_cache_block_size", 0)),
        }
        if engine_args is not None:
            engine_args = dict(engine_args)
            engine_args.update(resolved_config)
        metadata = {
            "schema_version": 1,
            "record_type": "atom_prefill_run_metadata",
            "model": str(getattr(config, "model", "")),
            "max_model_len": int(getattr(config, "max_model_len", 0)),
            "max_num_batched_tokens": int(
                getattr(config, "max_num_batched_tokens", 0)
            ),
            "max_num_seqs": int(getattr(config, "max_num_seqs", 0)),
            "parallel": {
                "tensor_parallel_size": int(
                    getattr(config, "tensor_parallel_size", 1)
                ),
                "data_parallel_size": int(
                    getattr(parallel, "data_parallel_size", 1)
                ),
                "pipeline_parallel_size": int(
                    getattr(config, "pipeline_parallel_size", 1)
                ),
                "decode_context_parallel_size": int(
                    getattr(config, "decode_context_parallel_size", 1)
                ),
                "prefill_context_parallel_size": int(
                    getattr(config, "prefill_context_parallel_size", 1)
                ),
            },
            "speculative": {
                "enabled": speculative is not None,
                "num_speculative_tokens": int(
                    getattr(speculative, "num_speculative_tokens", 0)
                ),
                "draft_model": str(getattr(speculative, "model", "")),
            },
            "engine_args": engine_args,
            "resolved_config": resolved_config,
            "runtime_env": {
                name: os.environ[name]
                for name in _REPLAY_RUNTIME_ENV_NAMES
                if name in os.environ
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._file = self.path.open("a", encoding="utf-8", buffering=1024 * 1024)
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._max_cases = max_cases
        self._next_case_id = 0
        self._trace_only = envs.ATOM_RUNTIME_TRACE or envs.ATOM_EXTEND_TRACE
        self._trace_active = not self._trace_only
        self._closed = False
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="atom-prefill-replay-writer",
            daemon=True,
        )
        self._thread.start()
        logger.info("ATOM prefill replay capture enabled: %s", self.path)

    def record_batch(self, batch: Any, seqs: dict[int, Any]) -> int | None:
        if (
            self._closed
            or not self._trace_active
            or int(batch.total_tokens_num_prefill) <= 0
            or int(getattr(batch, "total_tokens_num_decode", 0)) != 0
            or int(getattr(batch, "total_seqs_num_decode", 0)) != 0
        ):
            return None
        if self._max_cases > 0 and self._next_case_id >= self._max_cases:
            return None
        self._next_case_id += 1
        case_id = self._next_case_id
        self._queue.put(build_prefill_case_record(case_id, batch, seqs))
        return case_id

    def start_trace_capture(self) -> None:
        if self._trace_only and not self._closed:
            self._trace_active = True

    def stop_trace_capture(self) -> None:
        if self._trace_only:
            self._trace_active = False

    def record_result(self, case_id: int | None, output: Any) -> None:
        if not self._closed and case_id is not None:
            self._queue.put(build_prefill_result_record(case_id, output))

    def _writer_loop(self) -> None:
        while True:
            record = self._queue.get()
            if record is self._STOP:
                break
            self._file.write(
                json.dumps(record, separators=(",", ":")) + "\n"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._STOP)
        self._thread.join()
        self._file.flush()
        self._file.close()


def create_prefill_replay_recorder(config: Any) -> PrefillReplayRecorder | None:
    output_dir = os.getenv("ATOM_PREFILL_REPLAY_DIR")
    if not envs.ATOM_WORKLOAD_RECORD_PREFILL or not output_dir:
        return None
    parallel = config.parallel_config
    if (
        parallel.data_parallel_rank != 0
        or parallel.pipeline_parallel_rank != 0
    ):
        return None
    if envs.ATOM_PREFILL_REPLAY_MAX_CASES < 0:
        raise ValueError("ATOM_PREFILL_REPLAY_MAX_CASES must be non-negative")
    return PrefillReplayRecorder(
        output_dir,
        envs.ATOM_PREFILL_REPLAY_MAX_CASES,
        config,
    )
