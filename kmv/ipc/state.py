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
from multiprocessing import Lock, shared_memory
from typing import Tuple

import numpy as np        
import gc, warnings

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
    • Uses a tiny `Lock` to protect the producer's index update; reader is
      wait-free except for a single atomic read.
    """

    HEADER_BYTES = ctypes.sizeof(ctypes.c_uint32)   # 4

    # ───────────────────────────────────────────────────────────────────── #

    def __init__(
        self,
        *,
        shape: Tuple[int, ...],
        capacity: int = _CAPACITY_DEFAULT,
        name: str | None = None,
        create: bool = True,
    ) -> None:
        if capacity < 1 or (capacity & (capacity - 1)) != 0:
            raise ValueError(
                "capacity must be a power of two and ≥1 "
                f"(got {capacity})"
            )
        self._mask = capacity - 1          # single xor-able mask

        self.shape      = shape
        self.capacity   = capacity
        self.elem_size  = int(np.prod(shape))
        self._bytes     = self.elem_size * _DTYPE().nbytes
        shm_bytes       = self.HEADER_BYTES + capacity * self._bytes

        # (1) allocate or attach ------------------------------------------------
        self._shm = shared_memory.SharedMemory(
            name=name, create=create, size=shm_bytes
        )

        # 1) map the first 4 bytes to a *shared* uint32 cursor
        self._idx = ctypes.c_uint32.from_buffer(self._shm.buf, 0)

        # 2) the actual ring starts right after the header
        buf_start = self.HEADER_BYTES
        self._buf = np.ndarray(
            (capacity, self.elem_size), 
            dtype=_DTYPE, 
            buffer=self._shm.buf,
            offset=buf_start,
        )

        # (3) simple lock ---------------------------------------------
        self._lock = Lock()

    # ------------------------------------------------------------------ #
    #  Producer API
    # ------------------------------------------------------------------ #

    def push(self, arr: np.ndarray) -> None:
        """Copy *arr* into the next slot; arr must match `shape`."""
        if arr.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got {arr.shape}")

        with self._lock:
            i = (self._idx.value + 1) & self._mask
            self._buf[i, :] = arr.ravel()   # ① copy first
            self._idx.value = i             # ② publish index *after* data

    # ------------------------------------------------------------------ #
    #  Consumer API
    # ------------------------------------------------------------------ #

    def latest(self) -> np.ndarray:
        """Return a **copy** of the newest element."""
        i   = self._idx.value & self._mask     # defensive mask
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
        """
        Idempotent.  Detaches local Python views then closes the
        underlying `SharedMemory` mapping.
        """


        # 1. drop local views we own
        try:
            del self._buf
            del self._idx
        except AttributeError:
            pass

        gc.collect()                      # ensure ref-counts hit zero

        # 2. single close attempt
        try:
            self._shm.close()
        except BufferError as err:
            # probably some external view still alive – emit warning and move on
            warnings.warn(
                f"SharedArrayRing.close(): leaked view detected ({err}). "
                "Shared memory left mapped.", RuntimeWarning, stacklevel=2
            )

    def unlink(self) -> None:
        """Destroy the backing shared-memory block (creator only)."""
        self._shm.unlink()
