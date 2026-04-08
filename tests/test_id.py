import numpy as np
import pytest
from imid import from_array


class TestFromArray:
    def test_deterministic(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        assert from_array(arr) == from_array(arr)

    def test_different_arrays(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.ones((4, 4), dtype=np.uint8)
        assert from_array(a) != from_array(b)

    def test_different_shapes(self):
        a = np.zeros((2, 8), dtype=np.uint8)
        b = np.zeros((4, 4), dtype=np.uint8)
        assert from_array(a) != from_array(b)

    def test_different_dtypes(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.zeros((4, 4), dtype=np.float32)
        assert from_array(a) != from_array(b)

    def test_prefix(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        result = from_array(arr, prefix="img_")
        assert result.startswith("img_")

    def test_length(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        result = from_array(arr, length=16)
        assert len(result) == 16

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            from_array([1, 2, 3])

    def test_non_contiguous(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        sliced = arr[::2]
        assert from_array(sliced) == from_array(np.ascontiguousarray(sliced))
