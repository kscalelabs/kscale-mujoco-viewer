"""Lock-light ring buffer helpers.

Defines:
- `Ring` - structural protocol for a single-producer / single-consumer queue
- `InProcessRing` - deque-based implementation for threads in one process
"""

from collections import deque
from threading import Lock
from typing import (
    Generic,
    Iterable,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

T = TypeVar("T")


@runtime_checkable
class Ring(Protocol[T]):
    """Minimal API any ring must expose."""

    def push(self, item: T) -> None: ...
    def latest(self) -> Optional[T]: ...
    def __len__(self) -> int: ...

    @property
    def push_count(self) -> int: ...

    @property
    def pop_count(self) -> int: ...


class InProcessRing(Generic[T]):
    """Thread-safe deque ring for *in-process* producer/consumer work.

    Overwrites the oldest element on overflow.
    """

    __slots__ = ("_buf", "_lock", "_push_ctr", "_pop_ctr")

    def __init__(self, size: int = 8, init: Iterable[T] | None = None) -> None:
        if size < 1:
            raise ValueError("Ring size must be ≥ 1")
        self._buf: deque[T] = deque(init or (), maxlen=size)
        self._lock = Lock()
        self._push_ctr = 0
        self._pop_ctr = 0

    def push(self, item: T) -> None:
        with self._lock:
            self._buf.append(item)
            self._push_ctr += 1

    def latest(self) -> Optional[T]:
        with self._lock:
            if not self._buf:
                return None
            self._pop_ctr += 1
            return self._buf[-1]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def push_count(self) -> int:
        return self._push_ctr

    @property
    def pop_count(self) -> int:
        return self._pop_ctr


__all__ = ["Ring", "InProcessRing"]
