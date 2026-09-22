from unittest.mock import Mock

import pytest

from atom.model_engine import async_proc


def test_rpc_exception_always_runs_worker_cleanup():
    worker = async_proc.AsyncIOProc.__new__(async_proc.AsyncIOProc)
    cleanup = Mock()
    worker.busy_loop = Mock(side_effect=RuntimeError("rpc failed"))
    worker.exit = cleanup

    with pytest.raises(RuntimeError, match="rpc failed"):
        worker._run_busy_loop_with_cleanup()

    cleanup.assert_called_once_with()


def test_manager_requests_graceful_exit_before_forced_shutdown(monkeypatch):
    manager = async_proc.AsyncIOProcManager.__new__(
        async_proc.AsyncIOProcManager
    )
    manager.still_running = True
    manager.label = "test"
    manager.rpc_broadcast_mq = Mock()
    manager._cleanup_shared_memory = Mock()
    manager.outputs_queue = Mock()
    manager.output_thread = Mock()
    manager.kv_output_threads = []
    manager.parent_finalizer = Mock()
    proc = Mock()
    proc.is_alive.return_value = True
    manager.procs = [proc]
    shutdown = Mock()
    monkeypatch.setattr(async_proc, "shutdown_all_processes", shutdown)

    manager.exit()

    manager.rpc_broadcast_mq.enqueue.assert_called_once_with(("exit",))
    proc.join.assert_called_once()
    manager._cleanup_shared_memory.assert_called_once_with()
    shutdown.assert_called_once_with([proc], allowed_seconds=10)
