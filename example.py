from exposure_fusion import align_images, exposure_fusion
import cv2

img1 = cv2.imread('samples/peyrou_mean.jpg')
img2 = cv2.imread('samples/peyrou_under.jpg')
img3 = cv2.imread('samples/peyrou_over.jpg')

images = [img1, img2, img3]

aligned_images = align_images(images)

fusion = exposure_fusion(aligned_images, depth=4)

cv2.imwrite('samples/peyrou_fusion.jpg', fusion)

images = []
for i in range(1, 5):
    img = cv2.imread('samples/time_decay_%d.png' % i)
    images.append(img)

fusion = exposure_fusion(images, depth=3, time_decay=4)

cv2.imwrite('samples/time_decay_fusion.png', fusion)