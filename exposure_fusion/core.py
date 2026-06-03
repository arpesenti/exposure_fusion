from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from exposure_fusion._numpy_ops import (
    bgr_to_gray,
    get_gaussian_kernel,
    laplacian,
    resize,
    sep_filter2d,
)


def compute_weights(
    images: List[NDArray[np.uint8]],
    time_decay: Optional[float],
    well_exposedness_sigma: float = 0.04,
) -> List[NDArray[np.float32]]:
    w_c, w_s, w_e = 1, 1, 1

    if time_decay is not None:
        tau = len(images)
        time_sigma2 = (tau**2) / (np.float32(time_decay) ** 2)
        t = np.arange(tau - 1, -1, -1)
        decay = np.exp(-((t) ** 2) / (2 * time_sigma2))

    weights: List[NDArray[np.float32]] = []
    weights_sum = np.zeros(images[0].shape[:2], dtype=np.float32)
    for image_uint in images:
        image = np.float32(image_uint) / 255
        W = np.ones(image.shape[:2], dtype=np.float32)

        image_gray = bgr_to_gray(image)
        w_laplacian = laplacian(image_gray)
        W_contrast = np.absolute(w_laplacian) ** w_c + 1
        W *= W_contrast

        W_saturation = image.std(axis=2, dtype=np.float32) ** w_s + 1
        W *= W_saturation

        W_exposedness = (
            np.prod(
                np.exp(-((image - 0.5) ** 2) / (2 * well_exposedness_sigma)),
                axis=2,
                dtype=np.float32,
            )
            ** w_e
            + 1
        )
        W *= W_exposedness

        if time_decay is not None:
            W *= decay[len(weights)]

        weights_sum += W
        weights.append(W)

    nonzero = weights_sum > 0
    for w in weights:
        w[nonzero] /= weights_sum[nonzero]

    return weights


def gaussian_kernel(size: int = 5, sigma: float = 0.4) -> NDArray[np.float32]:
    return get_gaussian_kernel(size=size, sigma=sigma)


def image_reduce(image: NDArray) -> NDArray:
    kernel = gaussian_kernel()
    out = sep_filter2d(image, kernel, kernel.T)
    return resize(out, fx=0.5, fy=0.5)


def image_expand(image: NDArray) -> NDArray:
    kernel = gaussian_kernel()
    out = resize(image, fx=2, fy=2)
    return sep_filter2d(out, kernel, kernel.T)


def gaussian_pyramid(img: NDArray, depth: int) -> List[NDArray]:
    G = img.copy()
    gp = [G]
    for _ in range(depth):
        G = image_reduce(G)
        gp.append(G)
    return gp


def laplacian_pyramid(img: NDArray, depth: int) -> List[NDArray[np.float32]]:
    gp = gaussian_pyramid(img, depth + 1)
    lp: List[NDArray[np.float32]] = [gp[depth - 1].astype(np.float32)]
    for i in range(depth - 1, 0, -1):
        GE = image_expand(gp[i]).astype(np.float32)
        L = gp[i - 1].astype(np.float32) - GE
        lp = [L] + lp
    return lp


def pyramid_collapse(pyramid: List[NDArray]) -> NDArray[np.uint8]:
    depth = len(pyramid)
    collapsed = pyramid[depth - 1].astype(np.float32)
    for i in range(depth - 2, -1, -1):
        collapsed = (
            image_expand(collapsed).astype(np.float32) + pyramid[i].astype(np.float32)
        )
    return np.clip(np.round(collapsed), 0, 255).astype(np.uint8)


def exposure_fusion(
    images: List[NDArray[np.uint8]],
    depth: int = 3,
    time_decay: Optional[float] = None,
    well_exposedness_sigma: float = 0.04,
) -> NDArray[np.uint8]:

    if not isinstance(images, list) or len(images) < 2:
        raise ValueError("Input has to be a list of at least two images")

    if depth < 1:
        raise ValueError("depth must be >= 1")

    size = images[0].shape
    for img in images:
        if img.shape != size:
            raise ValueError("Input images have to be of the same size")

    weights = compute_weights(images, time_decay, well_exposedness_sigma)

    lps = []
    gps = []
    for image, weight in zip(images, weights):
        lps.append(laplacian_pyramid(image, depth))
        gps.append(gaussian_pyramid(weight, depth))

    LS = []
    for l in range(depth):
        ls = np.zeros(lps[0][l].shape, dtype=np.float32)
        for k in range(len(images)):
            gp_3ch = np.dstack((gps[k][l], gps[k][l], gps[k][l]))
            ls += lps[k][l] * gp_3ch
        LS.append(ls)

    return pyramid_collapse(LS)
