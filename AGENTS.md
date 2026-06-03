# exposure_fusion

## Quick start

```bash
pip install -e ".[test]"
pytest tests/
```

## Key facts

- **Images are BGR uint8 ndarrays internally.** Pillow I/O flips with `[:, :, ::-1]` on both read and write. The public API (`exposure_fusion`, `align_images`) expects BGR.
- **CLI**: `exposure-fusion` or `python -m exposure_fusion`.
- **Entrypoints**: `exposure_fusion/__init__.py` (public API: `exposure_fusion`, `align_images`), `exposure_fusion/cli.py` (argparse CLI via console_scripts), `exposure_fusion/__main__.py`.
- **Runtime deps**: only `numpy` + `Pillow`. No OpenCV needed at runtime.
- **Test deps**: `pip install "exposure_fusion[test]"` adds `opencv-python` + `pytest`. The numpy-op tests (`test_numpy_ops.py`, `test_alignment.py`) compare against OpenCV reference — they require `opencv-python` installed. `test_core.py` does not.
- **Architecture**: All image ops reimplemented in pure NumPy in `_numpy_ops.py`. No external image processing library is used.
- **Alignment**: Translation-only ECC alignment in `alignment.py`, pure NumPy. Only the first image is the reference; others are warped to match it.
- **No formatter, linter, or typechecker config** in repo. No CI config.
- **Single package, no monorepo.** Source is `exposure_fusion/`, tests are `tests/`.
