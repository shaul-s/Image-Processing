import cv2
import Reader as rd
import numpy as np

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
    """
    :param gaussianPyramid:
    :return:
    """
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
    """
    :param image: the image you want to create a mask for
    :return: mask binary image
    """
    image_copy = image.copy()
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if image_copy[j, i].all() != 0:
                image_copy[j, i] = 255

    return image_copy


### COPY METHODS FROM PART A ###
def computeHomography(trgtPoints, srcPoints):
    """
    :param srcPoints: source image homologue points
    :param trgtPoints: target image homologue points
    :return: the homography parameters
    """
    srcPoints = np.reshape(srcPoints, (srcPoints.size, 1))
    trgtPoints = np.reshape(trgtPoints, (trgtPoints.size, 1))

    n = srcPoints.shape[0]  # number of observations
    u = 8  # number of unknowns

    A = np.zeros((n, u))
    l = trgtPoints

    for i in range(0, n - 1, 2):
        A[i, 0] = srcPoints[i]
        A[i, 1] = srcPoints[i + 1]
        A[i, 2] = 1
        A[i, 3] = 0
        A[i, 4] = 0
        A[i, 5] = 0
        A[i, 6] = -srcPoints[i] * trgtPoints[i]
        A[i, 7] = -srcPoints[i + 1] * trgtPoints[i]

        A[i + 1, 0] = 0
        A[i + 1, 1] = 0
        A[i + 1, 2] = 0
        A[i + 1, 3] = srcPoints[i]
        A[i + 1, 4] = srcPoints[i + 1]
        A[i + 1, 5] = 1
        A[i + 1, 6] = -srcPoints[i] * trgtPoints[i + 1]
        A[i + 1, 7] = -srcPoints[i + 1] * trgtPoints[i + 1]

    X = np.dot(la.inv(np.dot(A.T, A)), np.dot(A.T, l))
    R = np.array([[float(X[0]), float(X[1]), float(X[2])], [float(X[3]), float(X[4]), float(X[5])],
                  [float(X[6]), float(X[7]), 1]])

    return R


def computeImageCorners(imgShape, R):
    """
    :param imgShape: shape of the center image
    :param R: transformation matrix
    :return: the new image corners
    """
    top_left = np.dot(R, np.array([[0], [0], [1]]))
    bottom_left = np.dot(R, np.array([[0], [imgShape[0]], [1]]))
    top_right = np.dot(R, np.array([[imgShape[1]], [0], [1]]))
    bottom_right = np.dot(R, np.array([[imgShape[1]], [imgShape[0]], [1]]))

    y_min = min(top_left[1], bottom_left[1], top_right[1], bottom_right[1])
    x_min = min(top_left[0], bottom_left[0], top_right[0], bottom_right[0])
    y_max = max(top_left[1], bottom_left[1], top_right[1], bottom_right[1])
    x_max = max(top_left[0], bottom_left[0], top_right[0], bottom_right[0])

    return np.array([x_max, x_min, y_max, y_min])


if __name__ == '__main__':
    """
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

    #cv2.imwrite("left_center.png", ls_)
    """

    ### PLANTING IMAGE INSIDE THE PANORAMA ###
    windows_scene = cv2.imread(r'something_nice.jpeg', 1)
    ### READING HOMOLOGUE POINTS ###
    targetPoints = rd.Reader.ReadSampleFile(r'full_panorama_rgb.json')
    sourcePoints = np.array([[0, windows_scene.shape[1], windows_scene.shape[1], 0],
                             [0, 0, windows_scene.shape[0], windows_scene.shape[0]]]).T
    ### COMPUTING HOMOGRAPHY ###
    T_matrix = computeHomography(targetPoints, sourcePoints)

    print('hi')
