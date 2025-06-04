"""Shared-memory ring buffer for single-producer/single-consumer IPC.

The implementation stores items in an anonymous `multiprocessing.SharedMemory`
block so a physics loop and GUI process can exchange high-rate NumPy arrays
without pickling or extra copies.
"""

import ctypes
import gc
import warnings
from multiprocessing import Lock, shared_memory
from typing import Tuple

import numpy as np

__all__ = ["SharedMemoryRing"]

_CAPACITY_DEFAULT = 64  # Needs to be a power of two so we can mask indices
_DTYPE = np.float64


class SharedMemoryRing:
    """Fixed-capacity ring in a `SharedMemory` block.

    A single writer pushes data with `push()`, a single reader fetches the
    newest value via `latest()`.  Overflow silently overwrites the oldest
    element – ideal for high-rate streaming where "latest frame wins".
    """

    HEADER_BYTES = ctypes.sizeof(ctypes.c_uint32)  # 4

    def __init__(
        self,
        *,
        shape: Tuple[int, ...],
        capacity: int = _CAPACITY_DEFAULT,
        name: str | None = None,
        create: bool = True,
    ) -> None:
        """Allocate or attach to a shared ring with the given *shape*.

        Number of elements in the ring must be a power of two so the
        writer can wrap the index with `(idx + 1) & (capacity - 1)` instead
        of the slower `% capacity` modulo.

        The caller is responsible for unlinking the block (`unlink()`) when done if `create=True`.
        """
        if capacity < 1 or (capacity & (capacity - 1)) != 0:
            raise ValueError(f"capacity must be a power of two and ≥1 (got {capacity})")
        self._mask = capacity - 1

        self.shape = shape
        self.capacity = capacity
        self.elem_size = int(np.prod(shape))
        self._bytes = self.elem_size * _DTYPE().nbytes
        shm_bytes = self.HEADER_BYTES + capacity * self._bytes

        self._shm = shared_memory.SharedMemory(name=name, create=create, size=shm_bytes)

        self._idx = ctypes.c_uint32.from_buffer(self._shm.buf, 0)

        buf_start = self.HEADER_BYTES
        self._buf: np.ndarray = np.ndarray(
            (capacity, self.elem_size),
            dtype=_DTYPE,
            buffer=self._shm.buf,
            offset=buf_start,
        )

        self._lock = Lock()

        self._push_ctr = 0
        self._pop_ctr = 0
        self._current_size = 0

    def push(self, arr: np.ndarray) -> None:
        """Append *arr* (must match `shape`).

        Oldest entry is dropped on wrap.
        """
        if arr.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got {arr.shape}")

        with self._lock:
            i = (self._idx.value + 1) & self._mask
            self._buf[i, :] = arr.ravel()
            self._idx.value = i
            self._push_ctr += 1
            if self._current_size < self.capacity:
                self._current_size += 1

    def latest(self) -> np.ndarray:
        """Return a **copy** of the most recent element; thread-safe and wait-free."""
        i = self._idx.value & self._mask
        out = self._buf[i].copy().reshape(self.shape)
        self._pop_ctr += 1
        return out

    def __len__(self) -> int:
        """Return current number of elements in the ring."""
        with self._lock:
            return self._current_size

    @property
    def push_count(self) -> int:
        """Total pushes since construction (monotonic)."""
        return self._push_ctr

    @property
    def pop_count(self) -> int:
        """Total successful `latest()` calls (monotonic)."""
        return self._pop_ctr

    @property
    def name(self) -> str:
        """SharedMemory block name used by a second process to attach."""
        return self._shm.name

    def close(self) -> None:
        """Detach local NumPy views and close the shared-memory mapping.

        This cleanup is important to avoid memory leaks.
        """
        try:
            del self._buf
            del self._idx
        except AttributeError:
            pass

        gc.collect()

        try:
            self._shm.close()
        except BufferError as err:
            warnings.warn(
                f"SharedMemoryRing.close(): leaked view detected ({err}). Shared memory left mapped.",
                RuntimeWarning,
                stacklevel=2,
            )

    def unlink(self) -> None:
        """Permanently destroy the backing shared-memory block (creator only)."""
        self._shm.unlink()
