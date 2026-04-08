import numpy as np
import pytest
from imid import get_id


class TestGetId:
    def test_deterministic(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        assert get_id(arr) == get_id(arr)

    def test_different_arrays(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.ones((4, 4), dtype=np.uint8)
        assert get_id(a) != get_id(b)

    def test_different_shapes(self):
        a = np.zeros((2, 8), dtype=np.uint8)
        b = np.zeros((4, 4), dtype=np.uint8)
        assert get_id(a) != get_id(b)

    def test_different_dtypes(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.zeros((4, 4), dtype=np.float32)
        assert get_id(a) != get_id(b)

    def test_prefix(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        result = get_id(arr, prefix="img_")
        assert result.startswith("img_")

    def test_length(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        result = get_id(arr, length=16)
        assert len(result) == 16

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            get_id([1, 2, 3])

    def test_non_contiguous(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        sliced = arr[::2]
        assert get_id(sliced) == get_id(np.ascontiguousarray(sliced))
