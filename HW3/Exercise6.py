import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == '__main__':
    img = cv2.imread('mario.png', 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(1 + np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    #plt.show()

    amplitude = np.abs(fshift)
    phase = np.angle(fshift)

    plt.subplot(121), plt.imshow(amplitude, cmap='gray')
    plt.title('Amplitude'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(phase, cmap='gray')
    plt.title('Phase'), plt.xticks([]), plt.yticks([])
    #plt.show()

    amp_in_time_domain = np.abs(np.fft.ifft2(amplitude))
    phase_in_time_domain = np.abs(np.fft.ifft2(phase))

    magnitude_spectrum1 = 20 * np.log(np.abs(amplitude))
    magnitude_spectrum2 = 20 * np.log(np.abs(phase))

    plt.subplot(121), plt.imshow(amp_in_time_domain, cmap='gray')
    plt.title('Amplitude'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(phase_in_time_domain, cmap='gray')
    plt.title('Phase'), plt.xticks([]), plt.yticks([])
    plt.show()

