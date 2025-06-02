# kmv/ipc/state.py
"""
Shared-memory, fixed-shape ring buffer for **bulk numeric arrays**
(qpos, qvel, RGB frames, …).

Only NumPy + multiprocessing.shared_memory are imported, so this module can be
used by both the sim process (creator / writer) and the GUI process (attacher /
reader) without pulling in Qt or MuJoCo.
"""

from __future__ import annotations

import ctypes
from multiprocessing import Lock, Value, shared_memory
from typing import Tuple

import numpy as np

__all__ = ["SharedArrayRing"]

# -----------------------------------------------------------------------------#
#  Configuration constants
# -----------------------------------------------------------------------------#

_CAPACITY_DEFAULT = 64       # power-of-two so we can mask indices
_DTYPE            = np.float64


# -----------------------------------------------------------------------------#
#  Implementation
# -----------------------------------------------------------------------------#

class SharedArrayRing:
    """
    A **single-producer / single-consumer** ring in anonymous shared memory.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of one element (e.g. `(nq,)` or `(240, 320, 3)`).
    capacity : int, optional
        Number of elements in the ring (power-of-two recommended).
    name : str | None
        Name of an existing `SharedMemory` block (for attach).
    create : bool
        `True`  → allocate new block.  
        `False` → attach to existing block (`name` must be given).

    Notes
    -----
    • Overwrites the oldest frame on overflow (perfectly fine for a viewer).  
    • Uses a tiny `Lock` to protect the producer’s index update; reader is
      wait-free except for a single atomic read.
    """

    # ───────────────────────────────────────────────────────────────────── #

    def __init__(
        self,
        *,
        shape: Tuple[int, ...],
        capacity: int = _CAPACITY_DEFAULT,
        name: str | None = None,
        create: bool = True,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be ≥ 1")

        self.shape      = shape
        self.capacity   = capacity
        self.elem_size  = int(np.prod(shape))
        self._bytes     = self.elem_size * _DTYPE().nbytes
        shm_bytes       = capacity * self._bytes

        # (1) allocate or attach ------------------------------------------------
        self._shm = shared_memory.SharedMemory(
            name=name, create=create, size=shm_bytes
        )

        # (2) expose as NumPy 2-D view  [capacity, *shape] ----------------------
        self._buf = np.ndarray(
            (capacity, self.elem_size), dtype=_DTYPE, buffer=self._shm.buf
        )

        # (3) simple cursor + lock ---------------------------------------------
        self._idx  = Value(ctypes.c_uint32, 0)
        self._lock = Lock()

    # ------------------------------------------------------------------ #
    #  Producer API
    # ------------------------------------------------------------------ #

    def push(self, arr: np.ndarray) -> None:
        """Copy *arr* into the next slot; arr must match `shape`."""
        if arr.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got {arr.shape}")

        with self._lock:
            i = (self._idx.value + 1) & (self.capacity - 1)
            self._idx.value = i
        self._buf[i, :] = arr.ravel()          # memory-copy

    # ------------------------------------------------------------------ #
    #  Consumer API
    # ------------------------------------------------------------------ #

    def latest(self) -> np.ndarray:
        """Return a **copy** of the newest element."""
        i   = self._idx.value                  # racy read is fine
        out = self._buf[i].copy().reshape(self.shape)
        return out

    # ------------------------------------------------------------------ #
    #  House-keeping
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        """SharedMemory block name – needed by the attacher."""
        return self._shm.name

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        """Destroy the backing shared-memory block (creator only)."""
        self._shm.unlink()
