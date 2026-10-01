import json
import queue
from types import SimpleNamespace

import numpy as np

from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.prefill_replay import (
    PrefillReplayRecorder,
    allocate_synthetic_replay_resources,
    build_prefill_case_record,
    build_prefill_result_record,
    build_synthetic_replay_batch,
    build_synthetic_replay_sequences,
    release_synthetic_replay_resources,
)


def _sequence(req_id, tokens, *, cached, block_table):
    return SimpleNamespace(
        id=req_id,
        external_request_id=f"req-{req_id}",
        token_ids=tokens,
        num_prompt_tokens=len(tokens),
        block_size=16,
        block_table=block_table,
        last_block_num_tokens=len(tokens) % 16 or 16,
        state_slots=[],
        state_fork_src=-1,
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        return_logprobs=False,
        num_cached_tokens=cached,
    )


def test_build_prefill_case_captures_replayable_prompt_windows():
    seqs = {
        7: _sequence(7, list(range(20)), cached=16, block_table=[3, 4]),
        9: _sequence(9, list(range(100, 108)), cached=0, block_table=[5]),
    }
    batch = SimpleNamespace(
        req_ids=[7, 9],
        num_scheduled_tokens=np.asarray([4, 8], dtype=np.int32),
        num_cached_tokens=[16, 0],
        context_lens=np.asarray([20, 8], dtype=np.int32),
        is_final_chunk=[True, True],
        next_token_ids=None,
        scheduled_tokens=np.asarray(
            list(range(16, 20)) + list(range(100, 108)),
            dtype=np.int32,
        ),
        total_tokens_num_prefill=12,
        total_seqs_num_prefill=2,
    )

    record = build_prefill_case_record(11, batch, seqs)

    assert record["case_id"] == 11
    assert record["batch"]["scheduled_tokens"] == (
        list(range(16, 20)) + list(range(100, 108))
    )
    assert record["batch"]["context_token_counts"] == [20, 8]
    assert record["requests"][0]["prompt_token_ids"] == list(range(20))


def test_build_prefill_result_is_json_safe():
    output = SimpleNamespace(
        req_ids=[7],
        token_ids=[(42,)],
        is_deferred_out=False,
        is_prev_prefill=True,
    )

    record = build_prefill_result_record(3, output)

    assert record["request_ids"] == [7]
    assert record["token_ids"] == [[42]]
    assert record["is_previous_prefill"] is True


def test_recorder_captures_engine_args_and_runtime_env(tmp_path, monkeypatch):
    engine_args = {
        "model": "/models/Kimi-K3",
        "tensor_parallel_size": 8,
        "kv_cache_dtype": "fp8",
        "online_quant_config": {"global_quant_config": "ptpc_fp8"},
    }
    monkeypatch.setenv(
        "ATOM_PREFILL_REPLAY_ENGINE_ARGS_JSON", json.dumps(engine_args)
    )
    monkeypatch.setenv("LMCACHE_LOCAL_CPU", "True")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "128")
    config = SimpleNamespace(
        model="/models/Kimi-K3",
        max_model_len=262144,
        max_num_batched_tokens=4096,
        max_num_seqs=32,
        kv_cache_dtype="fp8",
        index_cache_dtype="fp4",
        kv_cache_block_size=128,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        decode_context_parallel_size=8,
        prefill_context_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        speculative_config=None,
    )

    recorder = PrefillReplayRecorder(str(tmp_path), 1, config)
    recorder.close()

    metadata = json.loads(
        (tmp_path / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["engine_args"]["model"] == engine_args["model"]
    assert metadata["engine_args"]["tensor_parallel_size"] == 8
    assert metadata["engine_args"]["max_model_len"] == 262144
    assert metadata["engine_args"]["index_cache_dtype"] == "fp4"
    assert metadata["engine_args"]["block_size"] == 128
    assert metadata["resolved_config"]["kv_cache_dtype"] == "fp8"
    assert metadata["runtime_env"]["LMCACHE_LOCAL_CPU"] == "True"
    assert metadata["runtime_env"]["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "128"


def test_trace_only_recorder_activates_with_profiler(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOM_PREFILL_KERNEL_TRACE", "1")
    monkeypatch.delenv("ATOM_PREFILL_REPLAY_ENGINE_ARGS_JSON", raising=False)
    config = SimpleNamespace(
        model="/models/Kimi-K3",
        max_model_len=262144,
        max_num_batched_tokens=4096,
        max_num_seqs=32,
        kv_cache_dtype="fp8",
        index_cache_dtype=None,
        kv_cache_block_size=128,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        decode_context_parallel_size=8,
        prefill_context_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        speculative_config=None,
    )

    recorder = PrefillReplayRecorder(str(tmp_path), 20, config)
    assert recorder._trace_active is False
    recorder.start_trace_capture()
    assert recorder._trace_active is True
    recorder.stop_trace_capture()
    assert recorder._trace_active is False
    recorder.close()


def test_profile_utility_toggles_injected_prefill_recorder():
    recorder_calls = []
    recorder = SimpleNamespace(
        start_trace_capture=lambda: recorder_calls.append("start"),
        stop_trace_capture=lambda: recorder_calls.append("stop"),
    )
    runner_mgr = SimpleNamespace(
        call_func=lambda *args, **kwargs: True,
    )
    handler = EngineUtilityHandler(
        runner_mgr,
        queue.Queue(),
        prefill_replay_recorder=recorder,
    )

    handler._handle_start_profile({})
    handler._handle_stop_profile({})

    assert recorder_calls == ["start", "stop"]


def test_synthetic_replay_replaces_physical_cache_ids():
    case = {
        "case_id": 4,
        "batch": {
            "scheduled_tokens": [16, 17, 18, 19],
            "context_token_counts": [20],
            "state_maintenance": {
                "relocation_count": 0,
                "checkpoint_store_count": 1,
                "checkpoint_restore_count": 1,
            },
        },
        "requests": [
            {
                "request_id": 7,
                "external_request_id": "req-7",
                "prompt_token_ids": list(range(20)),
                "block_size": 16,
                "query_token_count": 4,
                "prefix_kv_token_count": 16,
                "is_final_chunk": True,
                "next_token_id": -1,
                "block_table": [91, 92],
                "last_block_num_tokens": 4,
                "state_slots": [81],
                "sampling": {
                    "temperature": 0.0,
                    "top_k": -1,
                    "top_p": 1.0,
                    "return_logprobs": False,
                },
            }
        ],
    }
    released = []
    state = SimpleNamespace(
        has_free=lambda count: count == 1,
        pop_many=lambda count: [51],
    )
    block_ids = iter([41, 42])
    block_manager = SimpleNamespace(
        state_slots_per_req=1,
        state=state,
        _fresh_block=lambda: next(block_ids),
    )

    def deallocate(seq):
        released.append(seq.id)
        del seq.block_table[:]
        seq.state_slots = []

    block_manager.deallocate = deallocate
    seqs = build_synthetic_replay_sequences(case)
    allocate_synthetic_replay_resources(block_manager, seqs, case)
    batch = build_synthetic_replay_batch(case, seqs)

    seq = seqs[7]
    assert list(seq.block_table) == [41, 42]
    assert list(seq.block_table) != case["requests"][0]["block_table"]
    assert seq.state_slots == [51]
    assert batch.replay_case_id == 4
    assert batch.context_lens.tolist() == [20]
    assert batch.state_maintenance_ops.empty
    release_synthetic_replay_resources(block_manager, seqs)
    assert released == [7]


def test_synthetic_replay_derives_resources_for_external_case():
    case = {
        "case_id": 1013,
        "batch": {
            "scheduled_tokens": [9, 9, 9, 9],
            "context_token_counts": [20],
        },
        "requests": [
            {
                "request_id": 1,
                "external_request_id": "sglang-1013-0",
                "prompt_token_ids": [9] * 20,
                "block_size": 16,
                "query_token_count": 4,
                "prefix_kv_token_count": 16,
                "context_token_count": 20,
                "is_final_chunk": True,
                "next_token_id": -1,
                "block_table": [],
                "allocate_kv_for_context": True,
                "last_block_num_tokens": 4,
                "state_slots": [],
                "requires_state_cache": True,
                "state_fork_src": 0,
                "sampling": {
                    "temperature": 0.0,
                    "top_k": -1,
                    "top_p": 1.0,
                    "return_logprobs": False,
                },
            }
        ],
    }
    state = SimpleNamespace(
        has_free=lambda count: count == 2,
        pop_many=lambda count: [51, 52],
    )
    block_ids = iter([41, 42])
    block_manager = SimpleNamespace(
        state_slots_per_req=2,
        state=state,
        num_pool_blocks=lambda context: 2 if context == 20 else 0,
        _fresh_block=lambda: next(block_ids),
        deallocate=lambda seq: None,
    )

    seqs = build_synthetic_replay_sequences(case)
    allocate_synthetic_replay_resources(block_manager, seqs, case)

    seq = seqs[1]
    assert seq.has_per_req_cache is True
    assert list(seq.block_table) == [41, 42]
    assert seq.state_slots == [51, 52]
    assert seq.state_fork_src == seq.state_slot
