from types import SimpleNamespace

from atom.model_engine.gpu_timing import gpu_timing_batch_method


class _Runner:
    def __init__(self):
        self.events = []

    def _start_extend_trace_profiler(self, batch):
        if batch.replay_case_id is None:
            return False
        self.events.append(("trace_start", batch.replay_case_id))
        return True

    def _stop_extend_trace_profiler(self):
        self.events.append(("trace_stop",))

    def _advance_runtime_trace_profiler(self):
        self.events.append(("advance",))

    @gpu_timing_batch_method
    def forward(self, batch):
        self.events.append(("forward",))


def _batch(*, prefill, decode, replay_case_id=None):
    return SimpleNamespace(
        total_tokens_num_prefill=prefill,
        total_tokens_num_decode=decode,
        total_seqs_num_decode=int(decode > 0),
        replay_case_id=replay_case_id,
    )


def test_prefill_only_trace_profiles_only_pure_extend(monkeypatch):
    monkeypatch.setenv("ATOM_EXTEND_TRACE", "1")
    runner = _Runner()

    runner.forward(_batch(prefill=16, decode=0, replay_case_id=7))
    assert runner.events == [
        ("trace_start", 7),
        ("forward",),
        ("trace_stop",),
    ]

    runner.events.clear()
    runner.forward(_batch(prefill=0, decode=8))
    assert runner.events == [("forward",)]

    runner.events.clear()
    runner.forward(_batch(prefill=16, decode=0))
    assert runner.events == [("forward",)]
