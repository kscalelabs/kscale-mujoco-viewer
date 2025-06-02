# kmv/core/buffer.py
"""
A **generic, in-process, single-producer / single-consumer ring buffer**.

* No multiprocessing.shared_memory — this lives entirely inside one Python
  process and is protected only by the GIL.
* Fast enough for kHz loops (deque + atomic pointer update).
* Type-parameterised so you can store Frame, np.ndarray, or anything else.

Used in two places:

1.  Inside the GUI process to decouple the Qt timer from the OpenGL repaint.
2.  Unit tests, where shared memory isn’t convenient.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Lock-light¹, overwrite-on-overflow ring.

    ¹We still take a tiny `threading.Lock` because deque operations are
    only *mostly* atomic in CPython; the lock adds ~50 ns and avoids edge-cases.
    """

    def __init__(self, size: int = 8) -> None:
        if size < 1:
            raise ValueError("RingBuffer size must be ≥ 1")
        self._buf: deque[T] = deque(maxlen=size)
        self._lock          = Lock()
        self._push_ctr      = 0          # total writes (debug / stats)
        self._pop_ctr       = 0          # total successful reads

    # ------------------------------------------------------------------ #
    # producer API
    # ------------------------------------------------------------------ #

    def push(self, item: T) -> None:
        """Append one element; overwrites the oldest on overflow."""
        with self._lock:
            self._buf.append(item)
            self._push_ctr += 1

    # ------------------------------------------------------------------ #
    # consumer API
    # ------------------------------------------------------------------ #

    def latest(self) -> T | None:
        """Return the **newest** element (or None if empty). Non-blocking."""
        with self._lock:
            if not self._buf:
                return None
            self._pop_ctr += 1
            return self._buf[-1]

    # ------------------------------------------------------------------ #
    # extras (optional, but handy for telemetry)
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Number of items currently in the buffer."""
        with self._lock:
            return len(self._buf)

    @property
    def push_count(self) -> int:        # total pushes since startup
        return self._push_ctr

    @property
    def pop_count(self) -> int:         # total successful `latest()` calls
        return self._pop_ctr
