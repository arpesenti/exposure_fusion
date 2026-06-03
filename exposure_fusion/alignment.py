from typing import List

import cv2
import numpy as np
from numpy.typing import NDArray


def align_images(images: List[NDArray[np.uint8]]) -> List[NDArray[np.uint8]]:

    if not isinstance(images, list) or len(images) < 2:
        raise ValueError("Input has to be a list of at least two images")

    size = images[0].shape
    for img in images:
        if img.shape != size:
            raise ValueError("Input images have to be of the same size")

    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    model_image = gray_images[0]
    sz = model_image.shape

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

    aligned_images: List[NDArray[np.uint8]] = [images[0]]
    for i in range(1, len(images)):
        cc, warp_matrix = cv2.findTransformECC(
            model_image,
            gray_images[i],
            warp_matrix,
            cv2.MOTION_TRANSLATION,
            criteria,
            inputMask=None,
            gaussFiltSize=3,
        )
        aligned_image = cv2.warpAffine(
            images[i],
            warp_matrix,
            (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        aligned_images.append(aligned_image)

    return aligned_images
