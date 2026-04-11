import hashlib
import numpy as np


def get_id(arr: np.ndarray, *, prefix: str = "", length: int = 8) -> str:
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a numpy.ndarray")
    if not 1 <= length <= 32:
        raise ValueError("length must be between 1 and 32")

    contiguous_arr = np.ascontiguousarray(arr)
    hasher = hashlib.md5()
    hasher.update(contiguous_arr.dtype.str.encode("ascii"))
    hasher.update(np.asarray(contiguous_arr.shape, dtype=np.int64).tobytes())
    hasher.update(memoryview(contiguous_arr))
    return prefix + hasher.hexdigest()[:length]
