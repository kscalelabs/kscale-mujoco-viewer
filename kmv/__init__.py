import multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")


__version__ = "0.3.13"
