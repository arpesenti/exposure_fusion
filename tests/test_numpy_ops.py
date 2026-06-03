import cv2
import numpy as np
import pytest

from exposure_fusion._numpy_ops import (
    rgb_to_gray,
    find_transform_ecc,
    get_gaussian_kernel,
    laplacian,
    resize,
    sep_filter2d,
    warp_affine,
    MOTION_TRANSLATION,
    TERM_CRITERIA_EPS,
    TERM_CRITERIA_COUNT,
)


def _gray_image(h=16, w=16):
    return np.random.randint(0, 256, (h, w), dtype=np.uint8)


def _rgb_image(h=16, w=16):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


class TestRgbToGray:
    def test_vs_opencv_uint8(self):
        for _ in range(5):
            img = _rgb_image(32, 48)
            ours = rgb_to_gray(img)
            cv = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            assert np.allclose(ours.astype(np.float32), cv.astype(np.float32), atol=1)

    def test_vs_opencv_float32(self):
        img = np.random.rand(16, 16, 3).astype(np.float32)
        ours = rgb_to_gray(img)
        cv = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        assert np.allclose(ours, cv, atol=1e-6)


class TestGaussianKernel:
    def test_vs_opencv(self):
        for size in [3, 5, 7]:
            for sigma in [0.3, 0.4, 0.5, 1.0]:
                ours = get_gaussian_kernel(size, sigma)
                cv = cv2.getGaussianKernel(size, sigma)
                assert np.allclose(ours, cv, atol=1e-6)


class TestLaplacian:
    def test_vs_opencv_uint8(self):
        img = _gray_image(32, 32)
        ours = laplacian(img.astype(np.float32))
        cv = cv2.Laplacian(img, cv2.CV_32F)
        assert np.allclose(ours, cv, atol=1e-4)

    def test_vs_opencv_random(self):
        for _ in range(5):
            img = _gray_image(np.random.randint(8, 32), np.random.randint(8, 32))
            ours = laplacian(img.astype(np.float32))
            cv = cv2.Laplacian(img, cv2.CV_32F)
            assert np.allclose(ours, cv, atol=1e-4)


class TestSepFilter2d:
    def test_vs_opencv_uint8(self):
        for size in [3, 5, 7]:
            k = cv2.getGaussianKernel(size, 0.5)
            img = _gray_image(32, 48)
            ours = sep_filter2d(img, k, k.T)
            cv = cv2.sepFilter2D(img, -1, k, k.T)
            assert np.allclose(ours.astype(np.float32), cv.astype(np.float32), atol=0.51)

    def test_vs_opencv_rgb(self):
        k = cv2.getGaussianKernel(5, 0.4)
        img = _rgb_image(32, 48)
        ours = sep_filter2d(img, k, k.T)
        cv = cv2.sepFilter2D(img, -1, k, k.T)
        assert np.allclose(ours, cv, atol=1)

    def test_vs_opencv_random(self):
        for _ in range(5):
            size = np.random.choice([3, 5, 7])
            sigma = np.random.uniform(0.2, 1.0)
            k = cv2.getGaussianKernel(size, sigma)
            h = np.random.randint(8, 32)
            w = np.random.randint(8, 32)
            img = _gray_image(h, w)
            ours = sep_filter2d(img, k, k.T)
            cv = cv2.sepFilter2D(img, -1, k, k.T)
            assert np.allclose(ours, cv, atol=1)


class TestResize:
    def test_downsample_vs_opencv_uint8(self):
        for _ in range(5):
            h = np.random.randint(8, 32) * 2
            w = np.random.randint(8, 32) * 2
            img = _gray_image(h, w)
            ours = resize(img, fx=0.5, fy=0.5)
            cv = cv2.resize(img, None, fx=0.5, fy=0.5)
            assert np.allclose(ours, cv, atol=1)

    def test_downsample_vs_opencv_rgb(self):
        for _ in range(5):
            h = np.random.randint(8, 32) * 2
            w = np.random.randint(8, 32) * 2
            img = _rgb_image(h, w)
            ours = resize(img, fx=0.5, fy=0.5)
            cv = cv2.resize(img, None, fx=0.5, fy=0.5)
            assert np.allclose(ours, cv, atol=1)

    def test_upsample_vs_opencv_uint8(self):
        for _ in range(5):
            h = np.random.randint(4, 16)
            w = np.random.randint(4, 16)
            img = _gray_image(h, w)
            ours = resize(img, fx=2, fy=2)
            cv = cv2.resize(img, None, fx=2, fy=2)
            assert np.allclose(ours, cv, atol=1)

    def test_upsample_vs_opencv_rgb(self):
        for _ in range(5):
            h = np.random.randint(4, 16)
            w = np.random.randint(4, 16)
            img = _rgb_image(h, w)
            ours = resize(img, fx=2, fy=2)
            cv = cv2.resize(img, None, fx=2, fy=2)
            assert np.allclose(ours, cv, atol=1)


class TestWarpAffine:
    def test_vs_opencv_identity(self):
        for _ in range(5):
            img = _rgb_image(32, 48)
            M = np.float32([[1, 0, 0], [0, 1, 0]])
            ours = warp_affine(img, M, (48, 32))
            cv = cv2.warpAffine(
                img, M, (48, 32), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            assert np.allclose(ours, cv, atol=1)

    def test_vs_opencv_translation_uint8(self):
        for _ in range(5):
            tx = np.random.randint(-5, 5)
            ty = np.random.randint(-5, 5)
            img = _rgb_image(24, 32)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            ours = warp_affine(img, M, (32, 24))
            cv = cv2.warpAffine(
                img, M, (32, 24), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            assert np.allclose(ours, cv, atol=1)

    def test_vs_opencv_translation_gray(self):
        for _ in range(5):
            tx = np.random.randint(-5, 5)
            ty = np.random.randint(-5, 5)
            img = _gray_image(24, 32)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            ours = warp_affine(img, M, (32, 24))
            cv = cv2.warpAffine(
                img, M, (32, 24), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            assert np.allclose(ours, cv, atol=1)

    def test_vs_opencv_inverse_map(self):
        for _ in range(5):
            tx = np.random.randint(-5, 5)
            ty = np.random.randint(-5, 5)
            img = _rgb_image(24, 32)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            ours = warp_affine(img, M, (32, 24))
            cv = cv2.warpAffine(
                img, M, (32, 24), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            assert np.allclose(ours, cv, atol=1)


class TestFindTransformEcc:
    def test_vs_opencv_identity(self):
        img = _gray_image(32, 32)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 5000, 1e-6)
        cc_ours, M_ours = find_transform_ecc(
            img, img, warp_matrix.copy(), MOTION_TRANSLATION, criteria
        )
        cc_cv, M_cv = cv2.findTransformECC(
            img, img, warp_matrix.copy(), cv2.MOTION_TRANSLATION, criteria
        )
        assert np.allclose(M_ours, np.eye(2, 3, dtype=np.float32), atol=1e-1)
        assert np.allclose(M_cv, np.eye(2, 3, dtype=np.float32), atol=1e-3)

    def test_vs_opencv_translation(self):
        h, w = 32, 32
        img = np.zeros((h, w), dtype=np.uint8)
        img[8:24, 8:24] = 255

        tx, ty = 3, 2
        M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(img, M_shift, (w, h))

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 5000, 1e-10)
        cc_ours, M_ours = find_transform_ecc(
            img, shifted, warp_matrix.copy(), MOTION_TRANSLATION, criteria
        )
        cc_cv, M_cv = cv2.findTransformECC(
            img, shifted, warp_matrix.copy(), cv2.MOTION_TRANSLATION, criteria
        )
        assert np.allclose(M_ours, M_cv, atol=8e-2), f"ours={M_ours}, cv={M_cv}"
        assert np.allclose(M_ours[0, 2], tx, atol=5e-2), f"ours={M_ours}, true tx={tx}"
        assert np.allclose(M_ours[1, 2], ty, atol=5e-2), f"ours={M_ours}, true ty={ty}"
