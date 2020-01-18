import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import linalg as la


def dist(x, x0, y, y0):
    """
    :return: distance between two points
    """
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)


if __name__ == '__main__':
    img = cv2.imread(r'img1.jpg', 1)
    ### REDUCING IMG TO GET BETTER RUN TIMES ###
    reduced_img = cv2.GaussianBlur(img[::2, ::2], (5, 5), 1)
    reduced_img = cv2.GaussianBlur(reduced_img[::2, ::2], (5, 5), 1)

    transformed_img1 = np.zeros(reduced_img.shape, dtype='uint8')
    transformed_img2 = np.zeros(reduced_img.shape, dtype='uint8')
    transformed_img3 = np.zeros(reduced_img.shape, dtype='uint8')
    transformed_img4 = np.zeros(reduced_img.shape, dtype='uint8')

    ### 1 ###
    a1 = 24  # a quarter of image length
    x1, y1 = 47, 31  # center of image

    for i in range(reduced_img.shape[1]):
        for j in range(reduced_img.shape[0]):
            dist1 = dist(i, x1, j, y1)
            if dist1 != 0:
                R = np.array([[np.cos(a1 / dist1), -np.sin(a1 / dist1), 0], [np.sin(a1 / dist1), np.cos(a1 / dist1), 0],
                              [0, 0, 1]])
            else:
                R = np.eye(3)
            pix = np.array([[i], [j], [1]])
            target_pix = np.dot(la.inv(R), pix).astype(int)  # computing the transformation
            if (target_pix[0] >= 0) and (target_pix[0] < reduced_img.shape[1]) and (target_pix[1] >= 0) and (
                    target_pix[1] < reduced_img.shape[0]):
                transformed_img1[j, i, :] = reduced_img[target_pix[1], target_pix[0]]

    ### 2 ###
    a1 = 20
    x1, y1 = 90, 60

    for i in range(reduced_img.shape[1]):
        for j in range(reduced_img.shape[0]):
            dist1 = dist(i, x1, j, y1)
            if dist1 != 0:
                R = np.array([[np.cos(a1 / dist1), -np.sin(a1 / dist1), 0], [np.sin(a1 / dist1), np.cos(a1 / dist1), 0],
                              [0, 0, 1]])
            else:
                R = np.eye(3)
            pix = np.array([[i], [j], [1]])
            target_pix = np.dot(la.inv(R), pix).astype(int)  # computing the transformation
            if (target_pix[0] >= 0) and (target_pix[0] < reduced_img.shape[1]) and (target_pix[1] >= 0) and (
                    target_pix[1] < reduced_img.shape[0]):
                transformed_img2[j, i, :] = reduced_img[target_pix[1], target_pix[0]]

        ### 3 ###
    a1 = 10
    x1, y1 = 25, 15

    for i in range(reduced_img.shape[1]):
        for j in range(reduced_img.shape[0]):
            dist1 = dist(i, x1, j, y1)
            if dist1 != 0:
                R = np.array([[np.cos(a1 / dist1), -np.sin(a1 / dist1), 0], [np.sin(a1 / dist1), np.cos(a1 / dist1), 0],
                              [0, 0, 1]])
            else:
                R = np.eye(3)
            pix = np.array([[i], [j], [1]])
            target_pix = np.dot(la.inv(R), pix).astype(int)  # computing the transformation
            if (target_pix[0] >= 0) and (target_pix[0] < reduced_img.shape[1]) and (target_pix[1] >= 0) and (
                    target_pix[1] < reduced_img.shape[0]):
                transformed_img3[j, i, :] = reduced_img[target_pix[1], target_pix[0]]

        ### 4 ###
        a1 = 30
        x1, y1 = 20, 20

        for i in range(reduced_img.shape[1]):
            for j in range(reduced_img.shape[0]):
                dist1 = dist(i, x1, j, y1)
                if dist1 != 0:
                    R = np.array(
                        [[np.cos(a1 / dist1), -np.sin(a1 / dist1), 0], [np.sin(a1 / dist1), np.cos(a1 / dist1), 0],
                         [0, 0, 1]])
                else:
                    R = np.eye(3)
                pix = np.array([[i], [j], [1]])
                target_pix = np.dot(la.inv(R), pix).astype(int)  # computing the transformation
                if (target_pix[0] >= 0) and (target_pix[0] < reduced_img.shape[1]) and (target_pix[1] >= 0) and (
                        target_pix[1] < reduced_img.shape[0]):
                    transformed_img4[j, i, :] = reduced_img[target_pix[1], target_pix[0]]

    ### PLOTTING ###
    plt.subplot(2, 2, 1), plt.imshow(transformed_img1)
    plt.title('alpha = 24, [x0,y0]=[47, 31]'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(transformed_img2)
    plt.title('alpha = 20, [x0,y0]=[90, 60]'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(transformed_img3)
    plt.title('alpha = 10, [x0,y0]=[25, 15]'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(transformed_img4)
    plt.title('alpha = 30, [x0,y0]=[20, 20]'), plt.xticks([]), plt.yticks([])
    plt.show()
