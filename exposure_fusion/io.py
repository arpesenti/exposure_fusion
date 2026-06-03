import numpy as np
from numpy.typing import NDArray

from PIL import Image


def load_image(path: str) -> NDArray[np.uint8]:
    return np.array(Image.open(path))


def save_image(path: str, img: NDArray[np.uint8]) -> None:
    Image.fromarray(img).save(path)
