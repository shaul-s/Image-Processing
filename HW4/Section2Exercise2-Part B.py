import numpy as np
from matplotlib import pyplot as plt
import cv2
import Reader as rd
from scipy import linalg as la


### PART B - IMAGE BLENDING USING PYRAMIDS ###

def computeGaussianPyramid(image, n):
    """
    :param image: image for pyramid building
    :param n: number of levels in the pyramid
    :return: list w the gaussian pyramid of the image
    """
    image_copy = image.copy()
    gp = [image_copy]  # first level is the original image
    for i in range(n):
        image_copy = image_copy[::2, ::2]
        # take every second pixel and smooth w 5x5 gaussian filter
        gp.append(cv2.GaussianBlur(image_copy, (5, 5), 1))

    return gp


def computeLaplacianPyramid(gaussianPyramid):
    lp = [gaussianPyramid[-1]]
    for i in range(len(gaussianPyramid) - 1, 0, -1):
        # expand using cv2 :(
        gausssian_expanded = cv2.resize(gaussianPyramid[i],
                                        dsize=(gaussianPyramid[i - 1].shape[1], gaussianPyramid[i - 1].shape[0]),
                                        interpolation=cv2.INTER_CUBIC)
        laplacian = cv2.subtract(gaussianPyramid[i - 1], gausssian_expanded)
        lp.append(laplacian)

    return lp


def generateMask(image):
    image_copy = image.copy()
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if image_copy[j, i].all() != 0:
                image_copy[j, i] = 255

    return image_copy


if __name__ == '__main__':
    ### LOADING PANORAMA IMAGES TO BLEND ###
    left_img = cv2.imread(r'left.jpg', 1)
    center_img = cv2.imread(r'center.jpg', 1)
    right_img = cv2.imread(r'right.jpg', 1)

    mask = generateMask(center_img)

    lpRight = computeLaplacianPyramid(computeGaussianPyramid(right_img, 6))
    lpCenter = computeLaplacianPyramid(computeGaussianPyramid(center_img, 6))
    lpLeft = computeLaplacianPyramid(computeGaussianPyramid(left_img, 6))

    gpMask = computeGaussianPyramid(mask, 6)
    gpMask = gpMask[::-1]
    ### BLEND LEFT AND CENTER PYRAMIDS ###
    LS_left = []
    for la, lb, gm in zip(lpLeft, lpCenter, gpMask):
        ls = lb * gm + la * (1.0 - gm)
        LS_left.append(ls)

    # now reconstruct
    ls_ = LS_left[0]
    for i in range(1, 7):
        ls_ = cv2.resize(ls_, dsize=(LS_left[i].shape[1], LS_left[i].shape[0]), interpolation=cv2.INTER_CUBIC)
        ls_ = cv2.add(ls_, LS_left[i])

    cv2.imwrite("left_center.png", ls_)
