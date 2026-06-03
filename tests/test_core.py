import numpy as np
import pytest
from exposure_fusion.core import (
    compute_weights,
    gaussian_kernel,
    image_reduce,
    image_expand,
    gaussian_pyramid,
    laplacian_pyramid,
    pyramid_collapse,
    exposure_fusion,
)


def _rgb_image(height=16, width=16):
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestGaussianKernel:
    def test_shape(self):
        k = gaussian_kernel(5, 0.4)
        assert k.shape == (5, 1)

    def test_sum_approx_one(self):
        k = gaussian_kernel(5, 0.4)
        assert abs(k.sum() - 1.0) < 1e-4

    def test_symmetric(self):
        k = gaussian_kernel(7, 1.0).flatten()
        assert np.allclose(k, k[::-1])


class TestImageReduce:
    def test_shape_halved(self):
        img = _rgb_image(16, 32)
        out = image_reduce(img)
        assert out.shape == (8, 16, 3)

    def test_output_uint8(self):
        img = _rgb_image(16, 16)
        out = image_reduce(img)
        assert out.dtype == np.uint8


class TestImageExpand:
    def test_shape_doubled(self):
        img = _rgb_image(8, 12)
        out = image_expand(img)
        assert out.shape == (16, 24, 3)

    def test_output_uint8(self):
        img = _rgb_image(8, 8)
        out = image_expand(img)
        assert out.dtype == np.uint8


class TestGaussianPyramid:
    def test_depth(self):
        img = _rgb_image(32, 32)
        gp = gaussian_pyramid(img, 3)
        assert len(gp) == 4  # depth + 1

    def test_level_shapes(self):
        img = _rgb_image(32, 32)
        gp = gaussian_pyramid(img, 3)
        for i, g in enumerate(gp):
            expected_h = 32 // (2**i)
            expected_w = 32 // (2**i)
            assert g.shape == (expected_h, expected_w, 3)


class TestLaplacianPyramid:
    def test_depth(self):
        img = _rgb_image(32, 32)
        lp = laplacian_pyramid(img, 3)
        assert len(lp) == 3  # depth

    def test_collapse_reconstructs(self):
        img = _rgb_image(16, 16)
        for depth in [2, 3]:
            lp = laplacian_pyramid(img, depth)
            reconstructed = pyramid_collapse(lp)
            assert reconstructed.shape == img.shape
            assert reconstructed.dtype == np.uint8
            assert np.allclose(reconstructed.astype(np.float32),
                               img.astype(np.float32), atol=2)


class TestPyramidCollapse:
    def test_output_shape(self):
        lp = [np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8).astype(np.float32),
              np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8).astype(np.float32)]
        out = pyramid_collapse(lp)
        assert out.shape == (8, 8, 3)
        assert out.dtype == np.uint8


class TestComputeWeights:
    def test_basic(self):
        images = [_rgb_image(8, 8), _rgb_image(8, 8)]
        weights = compute_weights(images, time_decay=None)
        assert len(weights) == 2
        assert weights[0].shape == (8, 8)
        assert weights[0].dtype == np.uint8

    def test_with_time_decay(self):
        images = [_rgb_image(8, 8), _rgb_image(8, 8), _rgb_image(8, 8)]
        weights = compute_weights(images, time_decay=3)
        assert len(weights) == 3
        for w in weights:
            assert w.shape == (8, 8)

    def test_normalized_sum(self):
        images = [_rgb_image(8, 8), _rgb_image(8, 8)]
        weights = compute_weights(images, time_decay=None)
        total = weights[0].astype(np.float32) + weights[1].astype(np.float32)
        assert np.allclose(total[total > 0], 255, atol=2)


class TestExposureFusion:
    def test_not_a_list(self):
        assert exposure_fusion("not a list") is None

    def test_single_image(self):
        assert exposure_fusion([_rgb_image(8, 8)]) is None

    def test_different_sizes(self):
        imgs = [_rgb_image(8, 8), _rgb_image(16, 16)]
        assert exposure_fusion(imgs) is None

    def test_basic_fusion(self):
        imgs = [_rgb_image(8, 8), _rgb_image(8, 8)]
        result = exposure_fusion(imgs, depth=2)
        assert result is not None
        assert result.shape == (8, 8, 3)
        assert result.dtype == np.uint8

    def test_fusion_with_time_decay(self):
        imgs = [_rgb_image(16, 16) for _ in range(3)]
        result = exposure_fusion(imgs, depth=2, time_decay=3)
        assert result is not None
        assert result.shape == (16, 16, 3)

    def test_fusion_depth_3(self):
        imgs = [_rgb_image(16, 16) for _ in range(2)]
        result = exposure_fusion(imgs, depth=3)
        assert result is not None
        assert result.shape == (16, 16, 3)
