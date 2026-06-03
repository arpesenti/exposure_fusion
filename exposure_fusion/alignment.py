from typing import List

import numpy as np
from numpy.typing import NDArray

from exposure_fusion._numpy_ops import (
    MOTION_TRANSLATION,
    TERM_CRITERIA_COUNT,
    TERM_CRITERIA_EPS,
    WARP_INVERSE_MAP,
    rgb_to_gray,
    find_transform_ecc,
    warp_affine,
)


def align_images(images: List[NDArray[np.uint8]]) -> List[NDArray[np.uint8]]:

    if not isinstance(images, list) or len(images) < 2:
        raise ValueError("Input has to be a list of at least two images")

    size = images[0].shape
    for img in images:
        if img.shape != size:
            raise ValueError("Input images have to be of the same size")

    gray_images = [rgb_to_gray(img) for img in images]
    model_image = gray_images[0]
    sz = model_image.shape

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 200, 1e-6)

    aligned_images: List[NDArray[np.uint8]] = [images[0]]
    for i in range(1, len(images)):
        cc, warp_matrix = find_transform_ecc(
            model_image,
            gray_images[i],
            warp_matrix,
            MOTION_TRANSLATION,
            criteria,
            inputMask=None,
            gaussFiltSize=3,
            num_levels=4,
        )
        aligned_image = warp_affine(
            images[i],
            warp_matrix,
            (sz[1], sz[0]),
            flags=WARP_INVERSE_MAP,
        )
        aligned_images.append(aligned_image)

    return aligned_images
