from typing import Tuple

import numpy as np
from numpy.typing import NDArray


MOTION_TRANSLATION = 0
INTER_LINEAR = 1
WARP_INVERSE_MAP = 16
TERM_CRITERIA_EPS = 2
TERM_CRITERIA_COUNT = 1


def rgb_to_gray(image: NDArray) -> NDArray:
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    result = np.dot(image.astype(np.float32), weights)
    if image.dtype == np.uint8:
        return np.round(result).astype(np.uint8)
    return result


def get_gaussian_kernel(size: int, sigma: float) -> NDArray[np.float32]:
    ax = np.arange(-(size - 1) / 2.0, (size - 1) / 2.0 + 1.0)
    kernel = np.exp(-0.5 * np.square(ax) / (sigma * sigma))
    kernel /= kernel.sum()
    return kernel.reshape(-1, 1).astype(np.float32)


def _reflect_pad(image: NDArray, pad_h: int, pad_w: int) -> NDArray:
    if image.ndim == 2:
        return _reflect_pad_2d(image, pad_h, pad_w)
    else:
        channels = [image[:, :, c] for c in range(image.shape[2])]
        padded_channels = [_reflect_pad_2d(ch, pad_h, pad_w) for ch in channels]
        return np.stack(padded_channels, axis=2)


def _reflect_pad_2d(image: NDArray, pad_h: int, pad_w: int) -> NDArray:
    if pad_h > 0:
        top = image[1 : pad_h + 1][::-1]
        bot = image[-(pad_h + 1) : -1][::-1]
        image = np.concatenate([top, image, bot], axis=0)
    if pad_w > 0:
        left = image[:, 1 : pad_w + 1][:, ::-1]
        right = image[:, -(pad_w + 1) : -1][:, ::-1]
        image = np.concatenate([left, image, right], axis=1)
    return image


def _conv2d_raw(image: NDArray, kernel: NDArray, _dtype=None) -> NDArray:
    if _dtype is None:
        _dtype = np.float64
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = _reflect_pad(image, pad_h, pad_w).astype(_dtype)

    if image.ndim == 2:
        h, w = image.shape[:2]
        shape = (h, w, kh, kw)
        strides = (
            padded.strides[0],
            padded.strides[1],
            padded.strides[0],
            padded.strides[1],
        )
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=shape, strides=strides, writeable=False
        )
        return np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))
    else:
        h, w, c = image.shape[:3]
        shape = (h, w, c, kh, kw)
        strides = (
            padded.strides[0],
            padded.strides[1],
            padded.strides[2],
            padded.strides[0],
            padded.strides[1],
        )
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=shape, strides=strides, writeable=False
        )
        return np.tensordot(windows, kernel, axes=([3, 4], [0, 1]))


def sep_filter2d(image: NDArray, kernel_x: NDArray, kernel_y: NDArray) -> NDArray:
    kernel_x = np.asarray(kernel_x, dtype=np.float32).ravel()
    kernel_y = np.asarray(kernel_y, dtype=np.float32).ravel()
    kernel_2d = np.outer(kernel_x, kernel_y)
    input_dtype = image.dtype
    result = _conv2d_raw(image, kernel_2d, _dtype=np.float32)
    return np.clip(np.round(result), 0, 255).astype(input_dtype)


def laplacian(image: NDArray) -> NDArray[np.float32]:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    result = _conv2d_raw(image.astype(np.float64), kernel)
    return result.astype(np.float32)


def resize(
    image: NDArray,
    dsize: None = None,
    fx: float = 0.5,
    fy: float = 0.5,
) -> NDArray:
    if fx == 0.5 and fy == 0.5:
        return _resize_half(image)
    elif fx == 2.0 and fy == 2.0:
        return _resize_double(image)
    else:
        raise NotImplementedError(
            f"Only fx=fy=0.5 or fx=fy=2.0 are supported, got fx={fx}, fy={fy}"
        )


def _resize_half(image: NDArray) -> NDArray:
    h, w = image.shape[:2]
    if image.ndim == 2:
        out = image.astype(np.float64)
        out = (
            out[0:h:2, 0:w:2]
            + out[1:h:2, 0:w:2]
            + out[0:h:2, 1:w:2]
            + out[1:h:2, 1:w:2]
        ) / 4.0
    else:
        out = image.astype(np.float64)
        out = (
            out[0:h:2, 0:w:2]
            + out[1:h:2, 0:w:2]
            + out[0:h:2, 1:w:2]
            + out[1:h:2, 1:w:2]
        ) / 4.0
    return np.round(out).astype(image.dtype)


def _resize_double(image: NDArray) -> NDArray:
    if image.ndim == 2:
        return _resize_double_2d(image).astype(image.dtype)
    else:
        channels = [
            _resize_double_2d(image[:, :, c]) for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2).astype(image.dtype)


def _resize_double_2d(image: NDArray) -> NDArray:
    h, w = image.shape
    src = image.astype(np.float64)

    rows_upsampled = np.zeros((h * 2, w), dtype=np.float64)
    rows_upsampled[0] = src[0]
    rows_upsampled[2 : 2 * h - 1 : 2] = 0.25 * src[: h - 1] + 0.75 * src[1:]
    rows_upsampled[1 : 2 * h - 2 : 2] = 0.75 * src[: h - 1] + 0.25 * src[1:]
    rows_upsampled[-1] = src[-1]

    out = np.zeros((h * 2, w * 2), dtype=np.float64)
    out[:, 0] = rows_upsampled[:, 0]
    out[:, 2 : 2 * w - 1 : 2] = (
        0.25 * rows_upsampled[:, : w - 1] + 0.75 * rows_upsampled[:, 1:]
    )
    out[:, 1 : 2 * w - 2 : 2] = (
        0.75 * rows_upsampled[:, : w - 1] + 0.25 * rows_upsampled[:, 1:]
    )
    out[:, -1] = rows_upsampled[:, -1]

    return np.round(out)


def warp_affine(
    image: NDArray,
    M: NDArray,
    dsize: Tuple[int, int],
    flags: int = INTER_LINEAR,
) -> NDArray:
    w_out, h_out = dsize[0], dsize[1]
    h_in, w_in = image.shape[:2]

    M_map = M.copy().astype(np.float64)

    xs_out, ys_out = np.meshgrid(
        np.arange(w_out, dtype=np.float64),
        np.arange(h_out, dtype=np.float64),
    )
    xs_in = M_map[0, 0] * xs_out + M_map[0, 1] * ys_out + M_map[0, 2]
    ys_in = M_map[1, 0] * xs_out + M_map[1, 1] * ys_out + M_map[1, 2]

    valid = (
        (xs_in >= 0) & (xs_in < w_in) & (ys_in >= 0) & (ys_in < h_in)
    )

    xs0 = np.floor(xs_in).astype(int)
    ys0 = np.floor(ys_in).astype(int)
    fx = xs_in - xs0
    fy = ys_in - ys0

    xs0c = np.clip(xs0, 0, w_in - 1)
    ys0c = np.clip(ys0, 0, h_in - 1)
    xs1c = np.clip(xs0c + 1, 0, w_in - 1)
    ys1c = np.clip(ys0c + 1, 0, h_in - 1)

    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    if image.ndim == 2:
        out = np.zeros((h_out, w_out), dtype=np.float64)
        out[valid] = (
            w00[valid] * image[ys0c[valid], xs0c[valid]]
            + w10[valid] * image[ys1c[valid], xs0c[valid]]
            + w01[valid] * image[ys0c[valid], xs1c[valid]]
            + w11[valid] * image[ys1c[valid], xs1c[valid]]
        )
        out[~valid] = 0
    else:
        out = np.zeros((h_out, w_out, image.shape[2]), dtype=np.float64)
        out[valid] = (
            w00[valid, None] * image[ys0c[valid], xs0c[valid]]
            + w10[valid, None] * image[ys1c[valid], xs0c[valid]]
            + w01[valid, None] * image[ys0c[valid], xs1c[valid]]
            + w11[valid, None] * image[ys1c[valid], xs1c[valid]]
        )
        out[~valid] = 0

    return np.round(out).astype(image.dtype)


_SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64)
_SCHARR_Y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float64)


def _compute_gradient(image: NDArray, gauss_filt_size: int):
    if gauss_filt_size > 0:
        sigma = 0.3 * (gauss_filt_size - 1) + 0.8
        k = get_gaussian_kernel(gauss_filt_size, sigma).ravel().astype(np.float64)
        kernel_2d = np.outer(k, k)
        smoothed = _conv2d_raw(image, kernel_2d)
    else:
        smoothed = image.astype(np.float64)

    gx = _conv2d_raw(smoothed, _SCHARR_X)
    gy = _conv2d_raw(smoothed, _SCHARR_Y)
    return gx, gy


def find_transform_ecc(
    template_image: NDArray,
    input_image: NDArray,
    warp_matrix: NDArray,
    motion_type: int,
    criteria: Tuple[int, int, float],
    inputMask: None = None,
    gaussFiltSize: int = 3,
) -> Tuple[float, NDArray]:
    max_iters, eps = criteria[1], criteria[2]
    h, w = template_image.shape[:2]

    T = template_image.astype(np.float64)
    I = input_image.astype(np.float64)
    T_mean = T.mean()

    p = np.array([warp_matrix[0, 2], warp_matrix[1, 2]], dtype=np.float64)

    for _ in range(max_iters):
        M = np.array([[1, 0, p[0]], [0, 1, p[1]]], dtype=np.float64)
        I_warped = warp_affine(I, M, (w, h)).astype(np.float64)

        gx, gy = _compute_gradient(I_warped, gaussFiltSize)

        I_w_mean = I_warped.mean()
        I_w_z = I_warped - I_w_mean
        T_z = T - T_mean

        G = np.stack([gx.ravel(), gy.ravel()], axis=1)

        H = G.T @ G
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            break

        norm_Iw = np.sqrt(np.sum(I_w_z**2))
        norm_T = np.sqrt(np.sum(T_z**2))
        if norm_Iw < 1e-10 or norm_T < 1e-10:
            break

        lam = norm_T / norm_Iw
        residual = (T_z - lam * I_w_z).ravel()
        dp = H_inv @ (G.T @ residual)

        p = p + dp
        if np.max(np.abs(dp)) < eps:
            break

    warp_matrix_out = np.array(
        [[1.0, 0.0, p[0]], [0.0, 1.0, p[1]]], dtype=np.float32
    )
    return float(0.0), warp_matrix_out
