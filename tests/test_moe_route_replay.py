import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_engine.moe_route_replay import (
    MoERouteHistogramProvider,
    MoERouteHistogramRecorder,
    MoERouteReplayError,
    begin_forward,
    finish_forward,
    histogram_to_topk_ids,
    record,
    set_active_route_lifecycle,
    validate_capture_config,
)


def test_histogram_to_topk_ids_is_deterministic_and_exact():
    counts = torch.tensor([3, 2, 2, 1])

    first = histogram_to_topk_ids(counts, num_tokens=4, topk=2)
    second = histogram_to_topk_ids(counts, num_tokens=4, topk=2)

    assert torch.equal(first, second)
    assert first.shape == (4, 2)
    assert all(len(set(row.tolist())) == 2 for row in first)
    assert torch.equal(torch.bincount(first.flatten(), minlength=4), counts)


@pytest.mark.parametrize(
    ("counts", "num_tokens", "topk", "message"),
    [
        ([1, 1], 2, 2, "does not equal"),
        ([3, 1], 2, 2, "exceeds num_tokens"),
        ([2, -1, 3], 2, 2, "cannot be negative"),
    ],
)
def test_histogram_to_topk_ids_rejects_invalid_histograms(
    counts, num_tokens, topk, message
):
    with pytest.raises(MoERouteReplayError, match=message):
        histogram_to_topk_ids(counts, num_tokens=num_tokens, topk=topk)


def test_provider_preserves_shape_dtype_and_validates_lifecycle():
    provider = MoERouteHistogramProvider(
        {7: {"model.layers.1.experts": [2, 1, 1]}}
    )
    natural_ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)

    provider.begin_forward(7)
    replay_ids = provider.route("model.layers.1.experts", natural_ids, 3)

    assert replay_ids.shape == natural_ids.shape
    assert replay_ids.dtype == natural_ids.dtype
    assert torch.equal(
        torch.bincount(replay_ids.long().flatten(), minlength=3),
        torch.tensor([2, 1, 1]),
    )
    provider.finish_forward()

    with pytest.raises(MoERouteReplayError, match="outside an active"):
        provider.route("model.layers.1.experts", natural_ids, 3)


def test_provider_prepares_structured_routes_before_forward():
    provider = MoERouteHistogramProvider(
        {
            7: {
                "layer": {
                    "counts": [2, 1, 1],
                    "num_tokens": 2,
                    "topk": 2,
                    "dtype": "torch.int32",
                }
            }
        },
        device="cpu",
    )
    natural_ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)

    provider.begin_forward(7)
    assert "layer" in provider._prepared
    prepared_ptr = provider._prepared["layer"].data_ptr()
    replay_ids = provider.route("layer", natural_ids, 3)
    provider.finish_forward()

    assert replay_ids.data_ptr() == prepared_ptr
    assert provider.last_verified_case_id == 7
    assert provider.last_verified_layers == ("layer",)


def test_provider_rejects_duplicate_and_missing_layers():
    cases = {1: {"layer.0": [1, 1], "layer.1": [1, 1]}}
    ids = torch.tensor([[0, 1]])

    duplicate = MoERouteHistogramProvider(cases)
    duplicate.begin_forward(1)
    duplicate.route("layer.0", ids, 2)
    with pytest.raises(MoERouteReplayError, match="duplicate"):
        duplicate.route("layer.0", ids, 2)

    missing = MoERouteHistogramProvider(cases)
    missing.begin_forward(1)
    missing.route("layer.0", ids, 2)
    with pytest.raises(MoERouteReplayError, match="missing MoE layers"):
        missing.finish_forward()


def test_process_local_api_overrides_ids_without_touching_weights():
    provider = MoERouteHistogramProvider({4: {"layer": [2, 1, 1]}})
    natural_ids = torch.tensor([[0, 1], [0, 1]])
    set_active_route_lifecycle(provider)
    try:
        begin_forward(4)
        replay_ids = record("layer", natural_ids, 3)
        finish_forward()
    finally:
        set_active_route_lifecycle(None)

    assert torch.equal(
        torch.bincount(replay_ids.flatten(), minlength=3),
        torch.tensor([2, 1, 1]),
    )


def test_recorder_writes_layer_major_uint16_sidecar(tmp_path):
    recorder = MoERouteHistogramRecorder(tmp_path)
    recorder.begin_forward(12)
    recorder.route("layer.0", torch.tensor([[0, 1], [0, 2]]), 3)
    recorder.route("layer.1", torch.tensor([[2, 2], [1, 2]]), 3)
    recorder.finish_forward()
    recorder.close()

    counts = np.fromfile(tmp_path / "moe_counts.uint16.bin", dtype="<u2")
    assert counts.tolist() == [2, 1, 1, 0, 1, 3]
    route_record = json.loads(
        (tmp_path / "moe_routes.jsonl").read_text(encoding="utf-8")
    )
    assert route_record["case_id"] == 12
    assert route_record["moe_counts"]["shape"] == [2, 3]
    metadata = json.loads(
        (tmp_path / "moe_routes.json").read_text(encoding="utf-8")
    )
    assert [layer["key"] for layer in metadata["layers"]] == [
        "layer.0",
        "layer.1",
    ]


def test_recorder_ignores_routes_outside_active_forward(tmp_path):
    recorder = MoERouteHistogramRecorder(tmp_path)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)

    assert recorder.route("layer.0", ids, 2) is ids

    recorder.begin_forward(1)
    recorder.route("layer.0", ids, 2)
    recorder.finish_forward()
    recorder.close()

    route_record = json.loads(
        (tmp_path / "moe_routes.jsonl").read_text(encoding="utf-8")
    )
    assert route_record["case_id"] == 1
    assert route_record["moe_counts"]["shape"] == [1, 2]


def test_recorder_rejects_duplicate_and_missing_layers(tmp_path):
    duplicate = MoERouteHistogramRecorder(tmp_path / "duplicate")
    duplicate.begin_forward(1)
    duplicate.route("layer.0", torch.tensor([[0]]), 1)
    with pytest.raises(MoERouteReplayError, match="duplicate"):
        duplicate.route("layer.0", torch.tensor([[0]]), 1)
    duplicate.abort_forward()
    duplicate.close()

    missing = MoERouteHistogramRecorder(tmp_path / "missing")
    missing.begin_forward(1)
    missing.route("layer.0", torch.tensor([[0]]), 1)
    missing.route("layer.1", torch.tensor([[0]]), 1)
    missing.finish_forward()
    missing.begin_forward(2)
    missing.route("layer.0", torch.tensor([[0]]), 1)
    with pytest.raises(MoERouteReplayError, match="layout mismatch"):
        missing.finish_forward()
    missing.close()


def test_recorder_rejects_uint16_overflow(tmp_path):
    recorder = MoERouteHistogramRecorder(tmp_path)
    recorder.begin_forward(1)
    recorder.route("layer.0", torch.zeros((65536, 1), dtype=torch.int64), 1)
    recorder.finish_forward()

    with pytest.raises(MoERouteReplayError, match="writer failed"):
        recorder.close()


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(
            enable_expert_parallel=True,
            eplb_enable=False,
            fake_eplb=False,
        ),
        SimpleNamespace(
            enable_expert_parallel=False,
            eplb_enable=True,
            fake_eplb=False,
        ),
    ],
)
def test_capture_rejects_ep_and_eplb(config, monkeypatch):
    monkeypatch.setenv("ATOM_WORKLOAD_RECORD_PREFILL", "1")
    monkeypatch.setenv("ATOM_PREFILL_REPLAY_DIR", "/tmp/replay")

    with pytest.raises(MoERouteReplayError, match="does not support"):
        validate_capture_config(config)
