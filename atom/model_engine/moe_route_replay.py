"""MoE expert-load histogram capture and deterministic route replay."""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("atom")

_COUNTS_BINARY = "moe_counts.uint16.bin"
_FORWARDS_JSONL = "moe_routes.jsonl"
_METADATA_JSON = "moe_routes.json"


class MoERouteReplayError(RuntimeError):
    """Invalid MoE histogram capture or replay state."""


def histogram_to_topk_ids(
    counts: torch.Tensor | list[int] | tuple[int, ...],
    *,
    num_tokens: int,
    topk: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Reconstruct deterministic unique-per-token IDs with exact expert loads."""
    counts = torch.as_tensor(counts, device="cpu", dtype=torch.int64)
    if counts.ndim != 1:
        raise MoERouteReplayError(
            f"expert counts must be 1-D, got shape {tuple(counts.shape)}"
        )
    if num_tokens < 0 or topk <= 0:
        raise MoERouteReplayError(
            f"invalid routing shape: num_tokens={num_tokens}, topk={topk}"
        )
    if bool(torch.any(counts < 0)):
        raise MoERouteReplayError("expert counts cannot be negative")
    expected = num_tokens * topk
    actual = int(counts.sum().item())
    if actual != expected:
        raise MoERouteReplayError(
            f"expert count sum {actual} does not equal "
            f"num_tokens * topk ({num_tokens} * {topk})"
        )
    if counts.numel() and int(counts.max().item()) > num_tokens:
        raise MoERouteReplayError(
            "an expert count exceeds num_tokens; unique-per-token routing "
            "cannot be reconstructed"
        )

    flat = torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=dtype),
        counts,
    )
    ids = flat.reshape(topk, num_tokens).transpose(0, 1).contiguous()
    if device is not None:
        ids = ids.to(device=device, non_blocking=True)
    return ids


class _RouteLifecycle(Protocol):
    def begin_forward(self, case_id: int) -> None: ...

    def route(
        self, layer_key: str, ids: torch.Tensor, num_experts: int
    ) -> torch.Tensor: ...

    def finish_forward(self) -> None: ...

    def abort_forward(self) -> None: ...


class MoERouteHistogramProvider:
    """Process-local provider of captured histograms for synthetic forwards."""

    def __init__(
        self,
        cases: Mapping[int, Mapping[str, Any]],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.int32,
    ):
        self._cases = cases
        self._device = device
        self._dtype = dtype
        self._case_id: int | None = None
        self._consumed: set[str] = set()
        self._prepared: dict[str, torch.Tensor] = {}
        self.last_verified_case_id: int | None = None
        self.last_verified_layers: tuple[str, ...] = ()

    def begin_forward(self, case_id: int) -> None:
        if self._case_id is not None:
            raise MoERouteReplayError("previous MoE replay forward was not finished")
        if case_id not in self._cases:
            raise MoERouteReplayError(f"missing MoE histogram case {case_id}")
        self._case_id = int(case_id)
        self._consumed = set()
        self._prepared = {}
        if self._device is not None:
            for layer_key, route in self._cases[self._case_id].items():
                if not isinstance(route, Mapping):
                    continue
                route_dtype = self._dtype
                dtype_name = str(route.get("dtype", ""))
                if dtype_name:
                    route_dtype = getattr(
                        torch,
                        dtype_name.removeprefix("torch."),
                        None,
                    )
                    if not isinstance(route_dtype, torch.dtype):
                        raise MoERouteReplayError(
                            f"unsupported route ID dtype {dtype_name!r}"
                        )
                self._prepared[layer_key] = histogram_to_topk_ids(
                    route["counts"],
                    num_tokens=int(route["num_tokens"]),
                    topk=int(route["topk"]),
                    device=self._device,
                    dtype=route_dtype,
                )

    def route(
        self, layer_key: str, ids: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        if self._case_id is None:
            raise MoERouteReplayError(
                "MoE route replay requested outside an active forward"
            )
        if layer_key in self._consumed:
            raise MoERouteReplayError(
                f"case {self._case_id}: duplicate MoE layer {layer_key!r}"
            )
        case = self._cases[self._case_id]
        if layer_key not in case:
            raise MoERouteReplayError(
                f"case {self._case_id}: unexpected MoE layer {layer_key!r}"
            )
        route = case[layer_key]
        if isinstance(route, Mapping) and len(route["counts"]) != num_experts:
            raise MoERouteReplayError(
                f"case {self._case_id}, layer {layer_key!r}: expected "
                f"{num_experts} expert counts, got {len(route['counts'])}"
            )
        if ids.ndim != 2:
            raise MoERouteReplayError(
                f"case {self._case_id}, layer {layer_key!r}: expected 2-D "
                f"top-k IDs, got shape {tuple(ids.shape)}"
            )
        replay_ids = self._prepared.get(layer_key)
        if replay_ids is None:
            counts_value = (
                route["counts"] if isinstance(route, Mapping) else route
            )
            counts = torch.as_tensor(counts_value, device="cpu")
            if counts.ndim != 1 or counts.numel() != num_experts:
                raise MoERouteReplayError(
                    f"case {self._case_id}, layer {layer_key!r}: expected "
                    f"{num_experts} expert counts, got shape {tuple(counts.shape)}"
                )
            replay_ids = histogram_to_topk_ids(
                counts,
                num_tokens=ids.shape[0],
                topk=ids.shape[1],
                device=ids.device,
                dtype=ids.dtype,
            )
        if replay_ids.shape != ids.shape:
            raise MoERouteReplayError(
                f"case {self._case_id}, layer {layer_key!r}: replay shape "
                f"{tuple(replay_ids.shape)} differs from natural route "
                f"{tuple(ids.shape)}"
            )
        if replay_ids.device != ids.device or replay_ids.dtype != ids.dtype:
            raise MoERouteReplayError(
                f"case {self._case_id}, layer {layer_key!r}: prepared replay "
                "route device/dtype differs from natural route"
            )
        self._consumed.add(layer_key)
        return replay_ids

    def finish_forward(self) -> None:
        if self._case_id is None:
            raise MoERouteReplayError("no active MoE replay forward")
        missing = set(self._cases[self._case_id]) - self._consumed
        case_id = self._case_id
        consumed = tuple(sorted(self._consumed))
        self.abort_forward()
        if missing:
            raise MoERouteReplayError(
                f"case {case_id}: missing MoE layers {sorted(missing)}"
            )
        self.last_verified_case_id = case_id
        self.last_verified_layers = consumed

    def abort_forward(self) -> None:
        self._case_id = None
        self._consumed = set()
        self._prepared = {}


class MoERouteHistogramRecorder:
    """Capture GPU histograms and serialize them after the profiled forward."""

    _STOP = object()

    def __init__(self, output_dir: str | os.PathLike[str]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.binary_path = self.output_dir / _COUNTS_BINARY
        self.forwards_path = self.output_dir / _FORWARDS_JSONL
        self.metadata_path = self.output_dir / _METADATA_JSON
        self._binary = self.binary_path.open("ab")
        self._forwards = self.forwards_path.open("a", encoding="utf-8")
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._case_id: int | None = None
        self._layers: list[
            tuple[str, int, torch.Tensor, int, int, int, str]
        ] = []
        self._seen: set[str] = set()
        self._layout: tuple[tuple[str, int], ...] | None = None
        self._closed = False
        self._thread_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="atom-moe-route-writer",
            daemon=True,
        )
        self._thread.start()

    def _check_healthy(self) -> None:
        if self._closed:
            raise MoERouteReplayError("MoE route recorder is closed")
        if self._thread_error is not None:
            raise MoERouteReplayError("MoE route writer failed") from self._thread_error

    def begin_forward(self, case_id: int) -> None:
        self._check_healthy()
        if self._case_id is not None:
            raise MoERouteReplayError("previous MoE capture forward was not finished")
        if case_id is None:
            raise MoERouteReplayError("MoE capture requires a prefill replay case ID")
        self._case_id = int(case_id)
        self._layers = []
        self._seen = set()

    def route(
        self, layer_key: str, ids: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        if self._case_id is None:
            # The recorder is installed for the ModelRunner lifetime, while
            # model startup can invoke MoE routing directly (for example
            # during CUDA graph capture) without entering the per-case
            # forward lifecycle. Those forwards are intentionally unrecorded.
            return ids
        if not layer_key:
            raise MoERouteReplayError("MoE layer key must be non-empty")
        if layer_key in self._seen:
            raise MoERouteReplayError(
                f"case {self._case_id}: duplicate MoE layer {layer_key!r}"
            )
        if num_experts <= 0:
            raise MoERouteReplayError(
                f"layer {layer_key!r}: num_experts must be positive"
            )
        if ids.ndim != 2:
            raise MoERouteReplayError(
                f"layer {layer_key!r}: expected 2-D top-k IDs, "
                f"got shape {tuple(ids.shape)}"
            )

        # Keep only a reference while the model forward is profiled. Histogram
        # kernels are launched by finish_forward(), after the GPU timing
        # decorator has disabled dynamic trace collection.
        self._seen.add(layer_key)
        self._layers.append(
            (
                layer_key,
                num_experts,
                ids,
                ids.numel(),
                int(ids.shape[0]),
                int(ids.shape[1]),
                str(ids.dtype),
            )
        )
        return ids

    def finish_forward(self) -> None:
        if self._case_id is None:
            raise MoERouteReplayError("no active MoE capture forward")
        if not self._layers:
            self.abort_forward()
            raise MoERouteReplayError("MoE capture forward recorded no layers")
        layout = tuple(
            (key, num_experts)
            for key, num_experts, *_ in self._layers
        )
        if self._layout is None:
            self._layout = layout
            self._write_metadata(layout)
        elif layout != self._layout:
            expected = [key for key, _ in self._layout]
            actual = [key for key, _ in layout]
            case_id = self._case_id
            self.abort_forward()
            raise MoERouteReplayError(
                f"case {case_id}: MoE layer layout mismatch; "
                f"expected {expected}, got {actual}"
            )

        self._layers = [
            (
                layer_key,
                num_experts,
                torch.bincount(
                    ids.reshape(-1).to(dtype=torch.int64),
                    minlength=num_experts,
                ),
                expected_sum,
                num_tokens,
                topk,
                dtype,
            )
            for (
                layer_key,
                num_experts,
                ids,
                expected_sum,
                num_tokens,
                topk,
                dtype,
            ) in self._layers
        ]
        event = None
        if self._layers[0][2].device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self._layers[0][2].device))
        self._queue.put(
            {
                "case_id": self._case_id,
                "layers": self._layers,
                "event": event,
            }
        )
        self.abort_forward()

    def abort_forward(self) -> None:
        self._case_id = None
        self._layers = []
        self._seen = set()

    def _write_metadata(self, layout: tuple[tuple[str, int], ...]) -> None:
        metadata = {
            "schema_version": 1,
            "record_type": "atom_moe_route_histograms",
            "counts_binary": _COUNTS_BINARY,
            "forwards": _FORWARDS_JSONL,
            "dtype": "uint16",
            "order": "forward-major/layer-major/expert-minor",
            "layers": [
                {"key": key, "num_experts": num_experts}
                for key, num_experts in layout
            ],
        }
        temporary = self.metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.metadata_path)

    def _write_forward(self, item: dict[str, Any]) -> None:
        event = item["event"]
        if event is not None:
            event.synchronize()

        offset_bytes = self._binary.tell()
        layer_refs = []
        total_values = 0
        uniform_num_experts: int | None = None
        for (
            layer_key,
            num_experts,
            gpu_counts,
            expected_sum,
            num_tokens,
            topk,
            dtype,
        ) in item["layers"]:
            counts = gpu_counts.to(device="cpu", dtype=torch.int64)
            if counts.ndim != 1 or counts.numel() != num_experts:
                raise MoERouteReplayError(
                    f"case {item['case_id']}, layer {layer_key!r}: histogram "
                    f"shape {tuple(counts.shape)} does not match {num_experts} experts"
                )
            actual_sum = int(counts.sum().item())
            if actual_sum != expected_sum:
                raise MoERouteReplayError(
                    f"case {item['case_id']}, layer {layer_key!r}: histogram "
                    f"sum {actual_sum} does not match selected IDs {expected_sum}"
                )
            if bool(torch.any(counts < 0)) or (
                counts.numel() and int(counts.max().item()) > 65535
            ):
                raise MoERouteReplayError(
                    f"case {item['case_id']}, layer {layer_key!r}: histogram "
                    "cannot be represented as uint16"
                )
            layer_refs.append(
                {
                    "key": layer_key,
                    "num_experts": num_experts,
                    "offset_elements": total_values,
                    "num_tokens": num_tokens,
                    "topk": topk,
                    "dtype": dtype,
                }
            )
            total_values += num_experts
            uniform_num_experts = (
                num_experts
                if uniform_num_experts in (None, num_experts)
                else -1
            )
            self._binary.write(counts.numpy().astype("<u2").tobytes())

        reference: dict[str, Any] = {
            "offset_bytes": offset_bytes,
            "length_bytes": total_values * 2,
            "dtype": "uint16",
            "layers": layer_refs,
        }
        if uniform_num_experts is not None and uniform_num_experts >= 0:
            reference["shape"] = [len(layer_refs), uniform_num_experts]
        record = {
            "schema_version": 1,
            "record_type": "atom_moe_route_forward",
            "case_id": item["case_id"],
            "moe_counts": reference,
        }
        self._forwards.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._binary.flush()
        self._forwards.flush()

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is self._STOP:
                    return
                self._write_forward(item)
        except BaseException as exc:
            self._thread_error = exc
            logger.exception("MoE route histogram writer failed")

    def close(self) -> None:
        if self._closed:
            return
        if self._case_id is not None:
            self.abort_forward()
        self._queue.put(self._STOP)
        self._thread.join()
        self._binary.close()
        self._forwards.close()
        self._closed = True
        if self._thread_error is not None:
            raise MoERouteReplayError("MoE route writer failed") from self._thread_error


_active: _RouteLifecycle | None = None


def set_active_route_lifecycle(lifecycle: _RouteLifecycle | None) -> None:
    """Install one process-local recorder or replay provider."""
    global _active
    if lifecycle is not None and _active is not None and lifecycle is not _active:
        raise MoERouteReplayError("a MoE route recorder/provider is already active")
    _active = lifecycle


def begin_forward(case_id: int) -> None:
    if _active is not None:
        _active.begin_forward(case_id)


def record(layer_key: str, ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Capture or replace logical IDs; callers keep naturally computed weights."""
    if _active is None:
        return ids
    return _active.route(layer_key, ids, num_experts)


def finish_forward() -> None:
    if _active is not None:
        _active.finish_forward()


def abort_forward() -> None:
    if _active is not None:
        _active.abort_forward()


def validate_capture_config(config: Any) -> None:
    """Reject route capture modes whose logical loads are not yet supported."""
    if not (
        os.getenv("ATOM_WORKLOAD_RECORD_PREFILL", "0") == "1"
        and os.getenv("ATOM_PREFILL_REPLAY_DIR")
    ):
        return
    if bool(getattr(config, "enable_expert_parallel", False)):
        raise MoERouteReplayError(
            "ATOM MoE route capture does not support expert parallelism yet"
        )
    if bool(getattr(config, "eplb_enable", False)) or bool(
        getattr(config, "fake_eplb", False)
    ):
        raise MoERouteReplayError(
            "ATOM MoE route capture does not support EPLB or fake EPLB yet"
        )


def create_moe_route_recorder(
    config: Any, *, tensor_parallel_rank: int
) -> MoERouteHistogramRecorder | None:
    """Create the TP-rank-0 recorder aligned with native prefill capture."""
    validate_capture_config(config)
    output_dir = os.getenv("ATOM_PREFILL_REPLAY_DIR")
    if not (
        os.getenv("ATOM_WORKLOAD_RECORD_PREFILL", "0") == "1" and output_dir
    ):
        return None
    if tensor_parallel_rank != 0:
        return None
    parallel = config.parallel_config
    if int(getattr(parallel, "data_parallel_rank", 0) or 0) != 0:
        return None
    if int(getattr(parallel, "pipeline_parallel_rank", 0) or 0) != 0:
        return None

    recorder = MoERouteHistogramRecorder(output_dir)
    set_active_route_lifecycle(recorder)
    atexit.register(recorder.close)
    logger.info("ATOM MoE route histogram capture enabled: %s", output_dir)
    return recorder


def moe_route_forward_method(func: "Callable") -> "Callable":
    """Bracket worker forwards carrying a native prefill replay case ID."""

    @wraps(func)
    def wrapped(self, batch, *args, **kwargs):
        case_id = getattr(batch, "replay_case_id", None)
        if _active is None or case_id is None:
            return func(self, batch, *args, **kwargs)
        begin_forward(int(case_id))
        try:
            result = func(self, batch, *args, **kwargs)
        except BaseException:
            abort_forward()
            raise
        finish_forward()
        return result

    return wrapped
