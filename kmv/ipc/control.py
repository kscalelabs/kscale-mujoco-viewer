"""Inter-process-communication for lower bandwidth control messages."""

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any

__all__ = ["ControlPipe", "make_metrics_queue"]


class ControlPipe:
    """Single-direction pipe with leak-proof handle management.

    The wrapper hands out a **send-only** end for the child and keeps a
    **recv-only** end in the parent, with convenience helpers for polling.
    """

    def __init__(self) -> None:
        parent_end, child_end = mp.Pipe(duplex=False)
        self._recv: Connection = parent_end
        self._send: Connection = child_end

    def sender(self) -> Connection:
        """Return the write-only handle to pass into the worker process."""
        return self._send

    def poll(self) -> bool:
        """Non-blocking check: `True` when a message is waiting."""
        try:
            return self._recv.poll()
        except (OSError, EOFError):
            return False

    def recv(self) -> tuple[str, Any]:
        """Blocking receive.  Returns `(tag, payload)`."""
        tag, payload = self._recv.recv()
        return tag, payload

    def close(self) -> None:
        """Close the read end held by the parent process."""
        self._recv.close()


def make_metrics_queue(maxsize: int = 1024) -> mp.Queue:
    """Bounded process-safe queue for tiny telemetry dicts.

    Limiting `maxsize` prevents unbounded memory growth if the GUI stalls.
    """
    return mp.get_context().Queue(maxsize=maxsize)
