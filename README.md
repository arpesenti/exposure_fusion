# Exposure Fusion

Python implementation of exposure fusion of multiple images, using the algorithm described in:

Mertens, Tom, Jan Kautz, and Frank Van Reeth. "Exposure Fusion." *Computer Graphics and Applications, 2007. PG'07. 15th Pacific Conference on*. IEEE, 2007.

Combines a bracket of differently-exposed images into a single well-exposed result via multi-resolution pyramid blending, without HDR radiance map estimation or tone mapping.

## Features

- **Exposure fusion** — Laplacian pyramid blending with contrast, saturation, and well-exposedness weighting
- **Automatic alignment** — Translation-only alignment via ECC (Enhanced Correlation Coefficient), pure NumPy
- **Time-decay weighting** — Progressive weighting for sequential image stacks (e.g. time-lapses)
- **No OpenCV required** — All image operations reimplemented in NumPy; runtime deps are `numpy` + `Pillow` only

## Requirements

- Python >= 3.10
- `numpy`
- `Pillow`

OpenCV is optional and only needed for running the test suite.

## Installation

```bash
pip install exposure_fusion
```

Or from a local checkout:

```bash
pip install .
```

With test dependencies:

```bash
pip install "exposure_fusion[test]"
```

## CLI Usage

```bash
exposure-fusion [-h] [-o OUTPUT] [-d DEPTH] [--time-decay TIME_DECAY]
                [--align] [-v] IMAGE [IMAGE ...]
```

Arguments:

| Argument | Description |
|----------|-------------|
| `IMAGE` (positional, 2+) | Input image file paths |
| `-o, --output` | Output image path (default: `fusion.jpg`) |
| `-d, --depth` | Pyramid depth (default: 3) |
| `--time-decay` | Time decay factor for sequential images |
| `--align` | Enable translation alignment before fusion |
| `-v, --verbose` | Print progress messages to stderr |

Examples:

```bash
exposure-fusion samples/peyrou_mean.jpg samples/peyrou_under.jpg samples/peyrou_over.jpg -o result.jpg
exposure-fusion --align -d 4 samples/peyrou_mean.jpg samples/peyrou_under.jpg samples/peyrou_over.jpg -o result.jpg
exposure-fusion --time-decay 4 samples/time_decay_1.png samples/time_decay_2.png samples/time_decay_3.png samples/time_decay_4.png -o fusion.png
```

Also invocable as `python -m exposure_fusion`.

## Python API

```python
from exposure_fusion import exposure_fusion, align_images
import numpy as np
from PIL import Image

def _read(path):
    return np.array(Image.open(path))[:, :, ::-1]  # RGB -> BGR

def _write(path, img):
    Image.fromarray(img[:, :, ::-1]).save(path)    # BGR -> RGB

# Load bracket exposures
img1 = _read('samples/peyrou_mean.jpg')
img2 = _read('samples/peyrou_under.jpg')
img3 = _read('samples/peyrou_over.jpg')

# Optional alignment
images = align_images([img1, img2, img3])

# Fuse
fusion = exposure_fusion(images, depth=4)

_write('samples/peyrou_fusion.jpg', fusion)

# Time-decay fusion (e.g. time-lapse)
images = [_read(f'samples/time_decay_{i}.png') for i in range(1, 5)]
fusion = exposure_fusion(images, depth=3, time_decay=4)
_write('samples/time_decay_fusion.png', fusion)
```

## Tests

```bash
pip install "exposure_fusion[test]"
pytest tests/
```

## License

MIT
