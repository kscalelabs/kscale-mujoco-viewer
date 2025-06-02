# SPDX-License-Identifier: MIT
"""
Shared-memory single-producer / single-consumer ring.

Stores only qpos & qvel (float64) – no forces for now.
"""

from multiprocessing import shared_memory, Value, Lock
import ctypes, numpy as np

CAPACITY = 64                       # power-of-two
F64      = np.float64

class SharedRing:
    """
    Ring with fixed element length:
        elem = np.concatenate([qpos, qvel])      (dtype float64)
    Size is decided at runtime from nq, nv.
    """

    def __init__(self, *, nq: int, nv: int,
                 name: str | None = None, create: bool = True):
        self.nq, self.nv = nq, nv
        self.elem_len    = nq + nv + 1        # +1 for sim_time
        self.bytes       = self.elem_len * 8  # float64 = 8 bytes
        size             = CAPACITY * self.bytes

        self.shm  = shared_memory.SharedMemory(name=name,
                                               create=create, size=size)
        self.buf  = np.ndarray((CAPACITY, self.elem_len),
                               dtype=F64, buffer=self.shm.buf)
        self.idx  = Value(ctypes.c_uint32, 0)
        self.lock = Lock()

    #  producer --------------------------------------------------------------
    def push(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        sim_time: float | int = 0.0,          # optional
    ) -> None:
        with self.lock:
            i = (self.idx.value + 1) & (CAPACITY - 1)
            self.idx.value = i
        self.buf[i, :self.nq]            = qpos
        self.buf[i, self.nq : self.nq+self.nv] = qvel
        self.buf[i, -1]                  = sim_time

    #  consumer --------------------------------------------------------------
    def latest(self) -> tuple[np.ndarray, np.ndarray, float]:
        i     = self.idx.value        # racy read is fine
        frame = self.buf[i].copy()    # copy out so producer may overwrite
        return (
            frame[:self.nq],
            frame[self.nq : self.nq + self.nv],
            float(frame[-1]),
        )

    # -----------------------------------------------------------------------
    @property
    def name(self) -> str: return self.shm.name
    def close(self) -> None:
        self.shm.close()
        self.shm.unlink()
