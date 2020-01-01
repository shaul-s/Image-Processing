import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
import glob


if __name__ == '__main__':
    # creating 5 random images
    a = np.eye(10)
    b = np.zeros((10, 10))
    b[1, :] = 1
    b[:, 1] = 1
    b[-2, :] = 1
    b[:, -2] = 1
    c = np.ones((10, 10)) * 255
    c[3:5, 3:5] = 0
    c[0, :] = 0
    c[-1, :] = 0
    d = np.zeros((10, 10))
    d[2:3, 2:3] = 255
    d[-2:-3, -2:-3] = 255
    e = (np.random.rand(10, 10) * 255).astype(int)


    imgs = np.array([a,b,c,d,e])
    # plotting the images before and after the inverse fft
    for img in imgs:
        img_in_time_domain = np.abs(np.fft.ifft2(img))
        magnitude_spectrum = np.log(1 + np.abs(img_in_time_domain))

        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Output Image'), plt.xticks([]), plt.yticks([])
        plt.show()

