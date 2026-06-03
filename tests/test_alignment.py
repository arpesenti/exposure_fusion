import numpy as np
import pytest
from exposure_fusion.alignment import align_images


def _bgr_image(height=16, width=16):
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestAlignImages:
    def test_not_a_list(self):
        assert align_images("not a list") is None

    def test_single_image(self):
        assert align_images([_bgr_image(8, 8)]) is None

    def test_different_sizes(self):
        imgs = [_bgr_image(8, 8), _bgr_image(16, 16)]
        assert align_images(imgs) is None

    def test_basic_alignment(self):
        img = _bgr_image(16, 16)
        imgs = [img.copy(), img.copy()]
        result = align_images(imgs)
        assert result is not None
        assert len(result) == 2
        assert result[0].shape == (16, 16, 3)
        assert result[1].shape == (16, 16, 3)
        assert result[0].dtype == np.uint8
        assert result[1].dtype == np.uint8
