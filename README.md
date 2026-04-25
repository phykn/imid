# imid

Deterministic MD5-based string IDs for NumPy arrays.

The hash incorporates **dtype**, **shape**, and **byte content**, so arrays differing in any of those produce different IDs.

## Install

```bash
pip install imid
```

## Usage

```python
import numpy as np
from imid import get_id

arr = np.array([1.0, 2.0, 3.0])

get_id(arr)                 # "a3f1b2c4"
get_id(arr, prefix="img_")  # "img_a3f1b2c4"
get_id(arr, length=16)      # "a3f1b2c4d5e6f7a8"
```

## API

### `get_id(arr, *, prefix="", length=8) -> str`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arr` | `np.ndarray` | *(required)* | Input array |
| `prefix` | `str` | `""` | String prepended to the ID |
| `length` | `int` | `8` | Length of the hex digest (1-32) |

- Non-contiguous arrays are handled automatically.
- Raises `TypeError` if `arr` is not a `numpy.ndarray`.
- Raises `ValueError` if `length` is outside `1..32`.

## Development

```bash
pip install -e ".[test]"
pytest
```

## License

MIT
