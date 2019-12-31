import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
import glob

def fun1(x):
    return -2*np.sin(2*np.pi*x)

if __name__ == '__main__':
    img = cv2.imread('pirate.tif', 0)
    kernel = np.array([[0,0.25,0],[0.25,0,0.25],[0,0.25,0]])
    filtered_img = cv2.filter2D(img, -1, kernel)

    #R = np.array([[np.cos(np.deg2rad(45)), -np.sin(np.deg2rad(45))],[np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45))]])

    plt.subplot(2, 2, 1)
    plt.title('original'); plt.xticks([]), plt.yticks([])
    plt.imshow(img, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('filtered'); plt.xticks([]), plt.yticks([])
    plt.imshow(filtered_img, cmap='gray')

    f_original = np.fft.fft2(filtered_img)
    fshift_original = np.fft.fftshift(f_original)
    magnitude_spectrum_original = 20 * np.log(np.abs(fshift_original))

    f_filtered = np.fft.fft2(filtered_img)
    fshift_filtered = np.fft.fftshift(f_filtered)
    magnitude_spectrum_filtered = 20 * np.log(np.abs(fshift_filtered))

    plt.subplot(2, 2, 3)
    plt.title('original freq'); plt.xticks([]), plt.yticks([])
    plt.imshow(magnitude_spectrum_original, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('filtered freq'); plt.xticks([]), plt.yticks([])
    plt.imshow(magnitude_spectrum_filtered, cmap='gray')

    plt.show()

    x = np.linspace(0,0.5*np.pi)
    y = fun1(x)
    plt.grid(color='r', linestyle='-', linewidth=0.1)
    plt.plot(x,y)

    plt.show()