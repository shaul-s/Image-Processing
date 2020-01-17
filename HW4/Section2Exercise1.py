import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import linalg as la
import glob


def dist(x, x0, y, y0):
    """
    :return: distance between two points
    """
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)


def computeImageCorners(imgShape, R):
    """
    :param imgShape:
    :param r:
    :return:
    """
    top_left = np.dot(R, np.array([[0], [0], [1]]))
    bottom_left = np.dot(R, np.array([[imgShape[0]], [0], [1]]))
    top_right = np.dot(R, np.array([[0], [imgShape[1]], [1]]))
    bottom_right = np.dot(R, np.array([[imgShape[1]], [imgShape[1]], [1]]))

    x_min = min(top_left[1], bottom_left[1], top_right[1], bottom_right[1])
    y_min = min(top_left[0], bottom_left[0], top_right[0], bottom_right[0])
    x_max = max(top_left[1], bottom_left[1], top_right[1], bottom_right[1])
    y_max = max(top_left[0], bottom_left[0], top_right[0], bottom_right[0])

    m = x_max - x_min
    n = y_max - y_min

    T = np.eye(3)
    T[0, -1] = -x_min
    T[1, -1] = -y_min

    return np.dot(T, R), np.array([n, m]).astype(int)


if __name__ == '__main__':
    img = cv2.imread(r'img1.jpg', 1)
    transformed_img1 = np.zeros(img.shape, dtype='uint8')
    transformed_img2 = np.zeros(img.shape, dtype='uint8')
    transformed_img3 = np.zeros(img.shape, dtype='uint8')
    transformed_img4 = np.zeros(img.shape, dtype='uint8')

    ### 1 ###
    a1 = 63  # a quarter of image length
    x1, y1 = 189, 126  # center of image

    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            dist1 = dist(i, x1, j, y1)
            if dist1 != 0:
                R = np.array([[np.cos(a1 / dist1), -np.sin(a1 / dist1), 0], [np.sin(a1 / dist1), np.cos(a1 / dist1), 0],
                              [0, 0, 1]])
            else:
                R = np.eye(3)
            pix = np.array([[i], [j], [1]])
            target_pix = np.dot(la.inv(R), pix).astype(int)  # computing the transformation
            if (target_pix[0] >= 0) and (target_pix[0] < img.shape[1]) and (target_pix[1] >= 0) and (
                    target_pix[1] < img.shape[0]):
                transformed_img1[j, i, :] = img[target_pix[1], target_pix[0]]

    downsized = img[::2, ::2]

    plt.imshow(downsized)
    # plt.imshow(img)

    # plt.imshow(transformed_img1)
    plt.show()
