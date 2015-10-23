from exposure_fusion import align_images, exposure_fusion
import cv2

img1 = cv2.imread('peyrou_mean.jpg')
img2 = cv2.imread('peyrou_under.jpg')
img3 = cv2.imread('peyrou_over.jpg')

images = [img1, img2, img3]

aligned_images = align_images(images)

fusion = exposure_fusion(aligned_images, depth=4)

cv2.imwrite('peyrou_fusion.jpg', fusion)

