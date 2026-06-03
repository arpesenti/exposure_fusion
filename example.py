from exposure_fusion import align_images, exposure_fusion
import numpy as np
from PIL import Image

def _read(path):
    return np.array(Image.open(path))[:, :, ::-1]

def _write(path, img):
    Image.fromarray(img[:, :, ::-1]).save(path)

img1 = _read('samples/peyrou_mean.jpg')
img2 = _read('samples/peyrou_under.jpg')
img3 = _read('samples/peyrou_over.jpg')

images = [img1, img2, img3]

aligned_images = align_images(images)

fusion = exposure_fusion(aligned_images, depth=4)

_write('samples/peyrou_fusion.jpg', fusion)

images = []
for i in range(1, 5):
    img = _read('samples/time_decay_%d.png' % i)
    images.append(img)

fusion = exposure_fusion(images, depth=3, time_decay=4)

_write('samples/time_decay_fusion.png', fusion)
