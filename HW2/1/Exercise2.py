import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob


def histogramEqualization(img):
    # original image
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.subplot(2, 2, 1)
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.subplot(2, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    # histogram equalized image

    height = img.shape[0]
    width = img.shape[1]
    cdf = (256*cdf/(height * width)).astype('uint8')
    he_img = cdf[img]
    # he_img = cv2.equalizeHist(img) - used to test
    hist2 = cv2.calcHist([he_img], [0], None, [256], [0, 256])
    cdf = hist2.cumsum()
    cdf_normalized = cdf * hist2.max() / cdf.max()
    plt.subplot(2, 2, 3)
    plt.plot(cdf_normalized, color='b')
    plt.hist(he_img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.subplot(2, 2, 4)
    plt.imshow(he_img, cmap='gray')
    plt.title('Histogram Equalized'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    # Find all the names of all the files with the format .jpg in the file Images\1
    imgNames1 = glob.glob(r'*.tif')
    imgNames2 = glob.glob(r'*.jpg')
    # imgNames is a list with the file names with the format .jpg in file Images\1

    for img in imgNames1:
        # part 1 - applying histogram equalization & plotting
        histogramEqualization(cv2.imread(img, 0))

    for img in imgNames2:
        # part 1 - applying histogram equalization & plotting
        histogramEqualization(cv2.imread(img, 0))











    print('')
    #for img in imgNames:



