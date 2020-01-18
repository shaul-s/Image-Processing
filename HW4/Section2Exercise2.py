import matplotlib.pyplot as plt
import numpy as np
import cv2
import Reader as rd
from scipy import linalg as la


def computeTransformationMatrix(srcPoints, trgtPoints):
    """
    :param srcPoints:
    :param trgtPoints:
    :return:
    """
    srcPoints = np.reshape(srcPoints.T, (srcPoints.size, 1))
    trgtPoints = np.reshape(trgtPoints.T, (trgtPoints.size, 1))

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
        A[i, 6] = srcPoints[i] * trgtPoints[i]
        A[i, 7] = srcPoints[i + 1] * trgtPoints[i]

        A[i + 1, 0] = 0
        A[i + 1, 1] = 0
        A[i + 1, 2] = 0
        A[i + 1, 3] = srcPoints[i]
        A[i + 1, 4] = srcPoints[i + 1]
        A[i + 1, 5] = 1
        A[i + 1, 6] = srcPoints[i] * trgtPoints[i + 1]
        A[i + 1, 7] = srcPoints[i + 1] * trgtPoints[i + 1]

    X = np.dot(la.inv(np.dot(A.T, A)), np.dot(A.T, l))
    R = np.array([[float(X[0]), float(X[1]), float(X[2])], [float(X[3]), float(X[4]), float(X[5])],
                  [float(X[6]), float(X[7]), 1]])

    return R


def computeImageCorners(imgShape, R):
    """
    :param imgShape: shape of the image
    :param R: transformation matrix
    :return: the new image size and new transform matrix
    """
    top_left = np.dot(R, np.array([[0], [0], [1]]))
    top_left = np.array([top_left[0] / top_left[2], top_left[1] / top_left[2]])

    bottom_left = np.dot(R, np.array([[0], [imgShape[0]], [1]]))
    bottom_left = np.array([bottom_left[0] / bottom_left[2], bottom_left[1] / bottom_left[2]])

    top_right = np.dot(R, np.array([[imgShape[1]], [0], [1]]))
    top_right = np.array([top_right[0] / top_right[2], top_right[1] / top_right[2]])

    bottom_right = np.dot(R, np.array([[imgShape[1]], [imgShape[0]], [1]]))
    bottom_right = np.array([bottom_right[0] / bottom_right[2], bottom_right[1] / bottom_right[2]])

    y_min = min(top_left[1], bottom_left[1], top_right[1], bottom_right[1])
    x_min = min(top_left[0], bottom_left[0], top_right[0], bottom_right[0])
    y_max = max(top_left[1], bottom_left[1], top_right[1], bottom_right[1])
    x_max = max(top_left[0], bottom_left[0], top_right[0], bottom_right[0])

    return np.array([x_max, x_min, y_max, y_min])


if __name__ == '__main__':
    left_img = cv2.imread(r'left_image.jpg', 1)
    center_img = cv2.imread(r'center_image.jpg', 1)
    right_img = cv2.imread(r'right_image.jpg', 1)
    """
    ### REDUCING IMG TO GET BETTER RUN TIMES ###
    r_left_img = cv2.GaussianBlur(left_img[::2, ::2], (5, 5), 1)
    r_left_img = cv2.GaussianBlur(r_left_img[::2, ::2], (5, 5), 1)
    r_center_img = cv2.GaussianBlur(center_img[::2, ::2], (5, 5), 1)
    r_center_img = cv2.GaussianBlur(r_center_img[::2, ::2], (5, 5), 1)
    r_right_img = cv2.GaussianBlur(right_img[::2, ::2], (5, 5), 1)
    r_right_img = cv2.GaussianBlur(r_right_img[::2, ::2], (5, 5), 1)
    """
    ### SPLITTING IMAGES TO 3 CHANNELS ###
    left_b, left_g, left_r = cv2.split(left_img)
    center_b, center_g, center_r = cv2.split(center_img)
    right_b, right_g, right_r = cv2.split(right_img)

    ### READING HOMOLOGUE POINTS ###
    sourcePoints = rd.Reader.ReadSampleFile(r'center_image.json')
    targetPoints_left = rd.Reader.ReadSampleFile(r'left_image.json')
    targetPoints_right = rd.Reader.ReadSampleFile(r'right_image.json')

    ### COMPUTING TRANSFORMATION MATRIX ###
    R_left = computeTransformationMatrix(targetPoints_left, sourcePoints)
    R_right = computeTransformationMatrix(targetPoints_right, sourcePoints)

    #
    R_left, status = cv2.findHomography(sourcePoints, targetPoints_left)
    R_right, status = cv2.findHomography(sourcePoints, targetPoints_right)
    #

    ### COMPUTING NEW TRANSFORMATION MATRIX + PANORAMA DIMENSIONS ###
    left_dimension = computeImageCorners(center_b.shape, R_left)
    right_dimension = computeImageCorners(center_b.shape, R_right)

    dimensions = np.hstack((left_dimension, right_dimension))

    pano_cols = int(max(dimensions[0:2, :].reshape((-1))) - min(dimensions[0:2, :].reshape((-1))))
    pano_rows = int(max(dimensions[1:-1, :].reshape((-1))) - min((dimensions[1:-1, :].reshape((-1)))))

    panorama = np.zeros((pano_rows, pano_cols))

    ### SHIFT MATRIX ###
    T = np.eye(3)
    T[0, -1] = -min(dimensions[0:2, :].reshape((-1)))
    T[1, -1] = -min(dimensions[1:-1, :].reshape((-1)))
    T = T.astype(int)

    left_matrix_inv = la.inv(np.dot(T, R_left))
    right_matrix_inv = la.inv(np.dot(T, R_right))
    center_matrix_inv = la.inv(T)

    for i in range(panorama.shape[1]):
        for j in range(panorama.shape[0]):
            pix = np.array([[i], [j], [1]])
            target_pix = np.dot(left_matrix_inv, pix)
            target_pix = np.array([target_pix[0] / target_pix[2], target_pix[1] / target_pix[2]]).astype(int)
            if (target_pix[0] >= 0) and (target_pix[0] < left_b.shape[1]) and (target_pix[1] >= 0) and (
                    target_pix[1] < left_b.shape[0]):
                panorama[j, i] = left_b[target_pix[1], target_pix[0]]

    for i in range(panorama.shape[1]):
        for j in range(panorama.shape[0]):
            pix = np.array([[i], [j], [1]])
            target_pix = np.dot(right_matrix_inv, pix)
            target_pix = np.array([target_pix[0] / target_pix[2], target_pix[1] / target_pix[2]]).astype(int)
            if (target_pix[0] >= 0) and (target_pix[0] < right_b.shape[1]) and (target_pix[1] >= 0) and (
                    target_pix[1] < right_b.shape[0]):
                panorama[j, i] = right_b[target_pix[1], target_pix[0]]

    for i in range(panorama.shape[1]):
        for j in range(panorama.shape[0]):
            pix = np.array([[i], [j], [1]])
            target_pix = np.dot(center_matrix_inv, pix)
            target_pix = np.array([target_pix[0] / target_pix[2], target_pix[1] / target_pix[2]]).astype(int)
            if (target_pix[0] >= 0) and (target_pix[0] < center_b.shape[1]) and (target_pix[1] >= 0) and (
                    target_pix[1] < center_b.shape[0]):
                panorama[j, i] = center_b[target_pix[1], target_pix[0]]

    plt.imshow(panorama, cmap='gray')
    plt.show()

    print('hi')
