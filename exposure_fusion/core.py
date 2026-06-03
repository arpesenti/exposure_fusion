import cv2
import numpy as np


def compute_weights(images, time_decay):
    (w_c, w_s, w_e) = (1, 1, 1)

    if time_decay is not None:
        tau = len(images)
        sigma2 = (tau**2)/(np.float32(time_decay)**2)
        t = np.array(range(tau-1, -1, -1))
        decay = np.exp(-((t)**2)/(2*sigma2))

    weights = []
    weights_sum = np.zeros(images[0].shape[:2], dtype=np.float32)
    i = 0
    for image_uint in images:
        image = np.float32(image_uint)/255
        W = np.ones(image.shape[:2], dtype=np.float32)

        # contrast
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(image_gray, cv2.CV_32F)
        W_contrast = np.absolute(laplacian) ** w_c + 1
        W = np.multiply(W, W_contrast)

        # saturation
        W_saturation = image.std(axis=2, dtype=np.float32) ** w_s + 1
        W = np.multiply(W, W_saturation)

        # well-exposedness
        sigma2 = 0.4
        W_exposedness = np.prod(np.exp(-((image - 0.5)**2)/(2*sigma2)), axis=2, dtype=np.float32) ** w_e + 1
        W = np.multiply(W, W_exposedness)

        if time_decay is not None:
            W *= decay[i]
            i += 1

        weights_sum += W

        weights.append(W)

    # normalization
    nonzero = weights_sum > 0
    for i in range(len(weights)):
        weights[i][nonzero] /= weights_sum[nonzero]
        weights[i] = np.uint8(weights[i]*255)

    return weights


def gaussian_kernel(size=5, sigma=0.4):
    return cv2.getGaussianKernel(ksize=size, sigma=sigma)


def image_reduce(image):
    kernel = gaussian_kernel()
    out_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    out_image = cv2.resize(out_image, None, fx=0.5, fy=0.5)
    return out_image


def image_expand(image):
    kernel = gaussian_kernel()
    out_image = cv2.resize(image, None, fx=2, fy=2)
    out_image = cv2.filter2D(out_image, cv2.CV_8UC3, kernel)
    return out_image


def gaussian_pyramid(img, depth):
    G = img.copy()
    gp = [G]
    for i in range(depth):
        G = image_reduce(G)
        gp.append(G)
    return gp


def laplacian_pyramid(img, depth):
    gp = gaussian_pyramid(img, depth+1)
    lp = [gp[depth-1]]
    for i in range(depth-1, 0, -1):
        GE = image_expand(gp[i])
        L = cv2.subtract(gp[i-1], GE)
        lp = [L] + lp
    return lp


def pyramid_collapse(pyramid):
    depth = len(pyramid)
    collapsed = pyramid[depth-1]
    for i in range(depth-2, -1, -1):
        collapsed = cv2.add(image_expand(collapsed), pyramid[i])
    return collapsed


def exposure_fusion(images, depth=3, time_decay=None):

    if not isinstance(images, list) or len(images) < 2:
        print("Input has to be a list of at least two images")
        return None

    size = images[0].shape
    for i in range(len(images)):
        if not images[i].shape == size:
            print("Input images have to be of the same size")
            return None

    # compute weights
    weights = compute_weights(images, time_decay)

    # compute pyramids
    lps = []
    gps = []
    for (image, weight) in zip(images, weights):
        lps.append(laplacian_pyramid(image, depth))
        gps.append(gaussian_pyramid(weight, depth))

    # combine pyramids with weights
    LS = []
    for l in range(depth):
        ls = np.zeros(lps[0][l].shape, dtype=np.uint8)
        for k in range(len(images)):
            lp = lps[k][l]
            gps_float = np.float32(gps[k][l])/255
            gp = np.dstack((gps_float, gps_float, gps_float))
            lp_gp = cv2.multiply(lp, gp, dtype=cv2.CV_8UC3)
            ls = cv2.add(ls, lp_gp)
        LS.append(ls)

    # collapse pyramid
    fusion = pyramid_collapse(LS)
    return fusion
