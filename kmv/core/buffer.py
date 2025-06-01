"""Very small SPSC ring-buffer for passing Frames from the sim-thread to Qt.

The implementation is intentionally simple – a `deque` protected by the GIL is
more than fast enough at 1 k Hz and avoids extra C-extensions.  Producer ⇢
consumer is wait-free as long as the ring is big enough to absorb bursts.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Single-producer / single-consumer ring buffer."""

    def __init__(self, size: int = 8) -> None:
        self._buf: deque[T] = deque(maxlen=size)
        self._lock = Lock()
        self._push_ctr: int = 0        # total writes
        self._pop_ctr:  int = 0        # total successful reads

    def push(self, item: T) -> None:
        with self._lock:
            self._buf.append(item)          # overwrites oldest on overflow
            self._push_ctr += 1

    def latest(self) -> T | None:
        with self._lock:
            if self._buf:
                self._pop_ctr += 1
                return self._buf[-1]
            return None

    def __len__(self) -> int:               # optional, for debugging
        with self._lock:
            return len(self._buf)

    @property
    def push_count(self) -> int:               # total pushes since start
        return self._push_ctr

    @property
    def pop_count(self) -> int:                # total successful pops
        return self._pop_ctr