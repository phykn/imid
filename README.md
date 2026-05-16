# imid

Deterministic MD5-based string IDs for NumPy arrays.

The hash incorporates **dtype**, **shape**, and **byte content**, so differences in any of those affect the ID.

## Install

```bash
pip install imid
```

## Usage

```python
import numpy as np
from imid import get_id

arr = np.array([1.0, 2.0, 3.0])

get_id(arr)                 # "70543f9b"
get_id(arr, prefix="img_")  # "img_70543f9b"
get_id(arr, length=16)      # "70543f9b270d20db"
```

## API

### `get_id(arr, *, prefix="", length=8) -> str`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arr` | `np.ndarray` | *(required)* | Input array |
| `prefix` | `str` | `""` | String prepended to the ID |
| `length` | `int` | `8` | Length of the hex digest (1-32) |

- Non-contiguous arrays are handled automatically.
- Raises `TypeError` if `arr` is not a `numpy.ndarray` or `length` is not an `int`.
- Raises `ValueError` if `length` is outside 1-32 (inclusive).

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## License

MIT
