import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
import glob


if __name__ == '__main__':
    img = cv2.imread('G0702695.JPG', 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(1 + np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    amplitude = np.abs(fshift)
    phase = np.angle(fshift)

