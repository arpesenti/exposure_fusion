import cv2
import numpy as np
import pytest
from exposure_fusion.alignment import align_images
from exposure_fusion._numpy_ops import bgr_to_gray, find_transform_ecc, MOTION_TRANSLATION, TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT


def _bgr_image(height=16, width=16):
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestAlignImages:
    def test_not_a_list(self):
        with pytest.raises(ValueError):
            align_images("not a list")

    def test_single_image(self):
        with pytest.raises(ValueError):
            align_images([_bgr_image(8, 8)])

    def test_different_sizes(self):
        imgs = [_bgr_image(8, 8), _bgr_image(16, 16)]
        with pytest.raises(ValueError):
            align_images(imgs)

    def test_basic_alignment(self):
        img = _bgr_image(16, 16)
        imgs = [img.copy(), img.copy()]
        result = align_images(imgs)
        assert len(result) == 2
        assert result[0].shape == (16, 16, 3)
        assert result[1].shape == (16, 16, 3)
        assert result[0].dtype == np.uint8
        assert result[1].dtype == np.uint8

    def test_translation_shift(self):
        h, w = 24, 32
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[4:12, 4:12] = 200
        img[8:16, 8:16] = 100

        tx, ty = 3, 2
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(img, M, (w, h))

        result = align_images([img, shifted])
        assert len(result) == 2
        assert result[1].shape == (h, w, 3)
        assert result[0].dtype == np.uint8
        assert result[1].dtype == np.uint8

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 5000, 1e-10)
        gray_img = bgr_to_gray(img)
        gray_shifted = bgr_to_gray(shifted)
        _, M_est = find_transform_ecc(gray_img, gray_shifted, warp_matrix.copy(),
                                      MOTION_TRANSLATION, criteria)
        assert abs(M_est[0, 2] - tx) < 0.5
        assert abs(M_est[1, 2] - ty) < 0.5
