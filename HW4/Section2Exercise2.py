import numpy as np
import cv2
import Reader as rd
from scipy import linalg as la


### PART A - HOMOGRAPHY AND SPATIAL TRANSFORMATION ###

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

    ### READING HOMOLOGUE POINTS ###
    targetPoints_left = rd.Reader.ReadSampleFile(r'center_image_left.json')
    targetPoints_right = rd.Reader.ReadSampleFile(r'center_image_right.json')
    sourcePoints_left = rd.Reader.ReadSampleFile(r'left_image.json')
    sourcePoints_right = rd.Reader.ReadSampleFile(r'right_image.json')

    ### COMPUTING TRANSFORMATION MATRIX ###
    R_left = computeHomography(targetPoints_left, sourcePoints_left)
    R_right = computeHomography(targetPoints_right, sourcePoints_right)

    ### COMPUTING PANORAMA DIMENSIONS + SHIFT MATRIX ###
    left_dimension = computeImageCorners(center_img.shape[:-1], R_left)
    right_dimension = computeImageCorners(center_img.shape[:-1], R_right)

    dimensions = np.hstack((left_dimension, right_dimension))

    pano_cols = int(max(dimensions[0:2, :].reshape((-1))) - min(dimensions[0:2, :].reshape((-1))))
    pano_rows = int(max(dimensions[1:-1, :].reshape((-1))))

    panorama_left = np.zeros((pano_rows, pano_cols, 3), dtype='uint8')  # blank panorama size
    panorama_center = np.zeros((pano_rows, pano_cols, 3), dtype='uint8')
    panorama_right = np.zeros((pano_rows, pano_cols, 3), dtype='uint8')

    ### SHIFT MATRIX ###
    T = np.eye(3)
    T[0, -1] = -min(dimensions[0:2, :].reshape((-1)))
    T[1, -1] = -min(dimensions[2:4, :].reshape((-1)))
    T = T.astype(int)

    ### COMPUTING INVERSE TRANSFORMATION ###
    left_matrix_inv = la.inv(np.dot(T, R_left))
    right_matrix_inv = la.inv(np.dot(T, R_right))
    center_matrix_inv = la.inv(T)

    ### RESAMPLING WITH NEAREST NEIGHBOUR INTERPOLATION ###
    for i in range(panorama_left.shape[1]):
        for j in range(panorama_left.shape[0]):
            pix = np.array([[i], [j], [1]])
            target_pix = np.dot(left_matrix_inv, pix)
            target_pix = np.array([target_pix[0] / target_pix[2], target_pix[1] / target_pix[2]])
            # check if pixel is indeed in the image
            if (target_pix[0] >= 0) and (target_pix[0] + 1 < left_img.shape[1]) and (target_pix[1] >= 0) and (
                    target_pix[1] + 1 < left_img.shape[0]):
                target_pix = target_pix.astype(int)
                panorama_left[j, i] = left_img[target_pix[1], target_pix[0], :]

    for i in range(panorama_right.shape[1]):
        for j in range(panorama_right.shape[0]):
            pix = np.array([[i], [j], [1]])
            target_pix = np.dot(right_matrix_inv, pix)
            target_pix = np.array([target_pix[0] / target_pix[2], target_pix[1] / target_pix[2]])
            # check if pixel is indeed in the image
            if (target_pix[0] >= 0) and (target_pix[0] + 1 < right_img.shape[1]) and (target_pix[1] >= 0) and (
                    target_pix[1] + 1 < right_img.shape[0]):
                target_pix = target_pix.astype(int)
                panorama_right[j, i] = right_img[target_pix[1], target_pix[0], :]

    for i in range(panorama_center.shape[1]):
        for j in range(panorama_center.shape[0]):
            pix = np.array([[i], [j], [1]])
            target_pix = np.dot(center_matrix_inv, pix)
            target_pix = np.array([target_pix[0] / target_pix[2], target_pix[1] / target_pix[2]]).astype(int)
            # check if pixel is indeed in the image
            if (target_pix[0] >= 0) and (target_pix[0] + 1 < center_img.shape[1]) and (target_pix[1] >= 0) and (
                    target_pix[1] + 1 < center_img.shape[0]):
                target_pix = target_pix.astype(int)
                panorama_center[j, i] = center_img[target_pix[1], target_pix[0], :]

    ### SAVING IMAGES ###
    cv2.imwrite('left.jpg', panorama_left)
    cv2.imwrite('right.jpg', panorama_right)
    cv2.imwrite('center.jpg', panorama_center)

    """
    for i in range(panorama.shape[1]):
        for j in range(panorama.shape[0]):
            pix = np.array([[i], [j], [1]])
            target_pix = np.dot(left_matrix_inv, pix)
            target_pix = np.array([target_pix[0] / target_pix[2], target_pix[1] / target_pix[2]])
            # check if pixel is indeed in the image
            if (target_pix[0] >= 0) and (target_pix[0] + 1 < left_img.shape[1]) and (target_pix[1] >= 0) and (
                    target_pix[1] + 1 < left_img.shape[0]):
                a = float(target_pix[0] - int(target_pix[0]))
                b = float(target_pix[1] - int(target_pix[1]))
                target_pix = target_pix.astype(int)
                panorama[j, i, :] = int(np.dot(np.dot(np.array([[1 - a], [a]]).T, np.array(
                    [[left_img[target_pix[1], target_pix[0]], left_img[target_pix[1] + 1, target_pix[0]]],
                     [left_img[target_pix[1], target_pix[0] + 1], left_img[target_pix[1] + 1, target_pix[0] + 1]]])[:, :, 0, 0]), np.array([1 - b, b])))

    """
