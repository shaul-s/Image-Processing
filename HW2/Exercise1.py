import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
import glob

if __name__ == '__main__':
    # part 1 - applying shear filters
    img = np.zeros((300,300))
    cv2.circle(img, (100,100), 50, 100, thickness=1, lineType=8, shift=0)

    # sobel
    img_sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    img_sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel = img_sobelx + img_sobely

    # prewitt
    kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    img_prewittx = cv2.filter2D(img, -1, kernelx)
    img_prewitty = cv2.filter2D(img, -1, kernely)
    img_prewitt = img_prewittx + img_prewitty

    # binomial
    b2 = (1/16)*np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    b4 = np.dot(np.array([[1], [4], [6], [4], [1]]), np.array([[1, 4, 6, 4, 1]]))
    img_b2 = cv2.filter2D(img, -1, b2)
    img_b4 = cv2.filter2D(img, -1, b4)
    img_binomial = img_b2 + img_b4


    # plotting
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(img_sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(img_sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(img_sobel, cmap='gray')
    plt.title('Sobel X + Y'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(img_prewittx, cmap='gray')
    plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(img_prewitty, cmap='gray')
    plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(img_prewitt, cmap='gray')
    plt.title('Prewitt X + Y'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(img_b2, cmap='gray')
    plt.title('b2'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(img_b4, cmap='gray')
    plt.title('b4'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(img_binomial, cmap='gray')
    plt.title('b2 + b4'), plt.xticks([]), plt.yticks([])
    plt.show()

    # part 2 - add gaussian noise
    gaus2 = random_noise(img, mode='gaussian', var = 2)
    gaus10 = random_noise(img, mode='gaussian', var = 10)
    gaus20 = random_noise(img, mode='gaussian', var = 20)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(gaus2, cmap='gray')
    plt.title('var = 2'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(gaus10, cmap='gray')
    plt.title('var = 10'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(gaus20, cmap='gray')
    plt.title('var = 20'), plt.xticks([]), plt.yticks([])
    plt.show()

    img_sobelx = cv2.Sobel(gaus20, cv2.CV_64F, 1, 0, ksize=3)
    img_sobely = cv2.Sobel(gaus20, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel = img_sobelx + img_sobely

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(img_sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(img_sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(img_sobel, cmap='gray')
    plt.title('Sobel X + Y'), plt.xticks([]), plt.yticks([])
    plt.show()

    """
    # Find all the names of all the files with the format .jpg in the file Images\1
    imgNames = glob.glob(r'Images\1\*.jpg')
    # imgNames is a list with the file names with the format .jpg in file Images\1
    """