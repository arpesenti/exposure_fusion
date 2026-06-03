from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from exposure_fusion.core import (
    compute_weights,
    gaussian_kernel,
    image_reduce,
    image_expand,
    gaussian_pyramid,
    laplacian_pyramid,
    pyramid_collapse,
    exposure_fusion as _exposure_fusion,
)
from exposure_fusion.alignment import align_images as _align_images
from exposure_fusion.io import load_image, save_image


def exposure_fusion(
    images: List[NDArray[np.uint8]] | List[str],
    depth: int = 3,
    time_decay: Optional[float] = None,
    well_exposedness_sigma: float = 0.04,
) -> NDArray[np.uint8]:
    if images and not isinstance(images[0], np.ndarray):
        images = [load_image(p) for p in images]
    return _exposure_fusion(images, depth=depth, time_decay=time_decay,
                            well_exposedness_sigma=well_exposedness_sigma)


def align_images(
    images: List[NDArray[np.uint8]] | List[str],
) -> List[NDArray[np.uint8]]:
    if images and not isinstance(images[0], np.ndarray):
        images = [load_image(p) for p in images]
    return _align_images(images)
