# kmv/ipc/control.py
"""
Tiny wrappers around `multiprocessing` primitives used for *control* traffic:

* `ControlPipe` – one-way, single-producer/single-consumer, low latency.
* `make_metrics_queue` – bounded `multiprocessing.Queue` for telemetry.

Both classes are deliberately minimal; they add just enough sugar to keep the
rest of the code clean and to remain start-method agnostic (“spawn”, “fork”,
“forkserver” all work).
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any

__all__ = ["ControlPipe", "make_metrics_queue"]


# -----------------------------------------------------------------------------#
# 1.  Pipe wrapper
# -----------------------------------------------------------------------------#

class ControlPipe:
    """
    One-way pipe: **child writes → parent reads** (or vice-versa – you choose).

    Why a wrapper?
    --------------
    * Ensures we never leak the write-end in the wrong process.
    * Provides non-blocking poll/recv convenience.
    * Keeps type hints tidy.
    """

    def __init__(self) -> None:
        parent_end, child_end = mp.Pipe(duplex=False)
        self._recv: Connection = parent_end   # used by the receiver process
        self._send: Connection = child_end    # handed to the sender process

    # -- handle access -------------------------------------------------------- #

    def sender(self) -> Connection:
        """Return the *send*-only end – pass this to the worker process."""
        return self._send

    # -- receiver side convenience ------------------------------------------- #

    def poll(self) -> bool:
        """`True` if a message is waiting (non-blocking)."""
        try:
            return self._recv.poll()
        except (OSError, EOFError):
            return False

    def recv(self) -> tuple[str, Any]:
        """Blocking receive.  Returns `(tag, payload)`."""
        tag, payload = self._recv.recv()
        return tag, payload

    # -- clean-up ------------------------------------------------------------- #

    def close(self) -> None:
        self._recv.close()
        # sender end is closed by whichever process owns it


# -----------------------------------------------------------------------------#
# 2.  Telemetry queue
# -----------------------------------------------------------------------------#

def make_metrics_queue(maxsize: int = 1024) -> mp.Queue:
    """
    Return a *bounded*, process-safe queue for scalar telemetry.

    Using `maxsize` avoids unbounded memory growth if the GUI hangs.
    Feel free to tune the limit; metrics are tiny.
    """
    return mp.get_context().Queue(maxsize=maxsize)
