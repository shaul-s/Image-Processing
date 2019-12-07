import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

def normalCDF(x, E, sigma):
    return (1/np.sqrt(2*np.pi*sigma))*np.exp((-(x-E)**2)/(2*sigma))

def histMatching(img, template_hist):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.subplot(1, 2, 1)
    plt.hist(img.flatten(), 256, [0, 256], color='r', label='histogram')
    plt.xlim([0, 256])
    plt.legend(loc='upper left')
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.show()

    k = 256
    new_values = np.zeros((k))
    cdf = hist.cumsum()/(img.shape[0]*img.shape[1])


    for a in np.arange(k):
        j = k - 1
        while True:
            new_values[a] = j
            j = j - 1
            if j < 0 or cdf[a] > template_hist[j]:
                break

    hm_img = img

    for i in np.arange(img.shape[0]) :
        for j in np.arange(img.shape[1]) :
            a = img.item(i, j)
            b = new_values[a]
            hm_img.itemset((i, j), b)

    plt.subplot(1, 2, 1)
    plt.plot(new_values, color='b', label='trans function')
    plt.xlim([0, 256])
    plt.legend(loc='upper left')
    plt.subplot(1, 2, 2)
    plt.hist(hm_img.flatten(), 256, [0, 256], color='r', label='matched histogram')
    plt.xlim([0, 256])
    plt.legend(loc='upper left')
    plt.show()
    plt.subplot(1, 1, 1)
    plt.imshow(hm_img, cmap='gray')
    plt.title('Matched Image'), plt.xticks([]), plt.yticks([])
    plt.show()



if __name__ == '__main__':
    # Find all the names of all the files with the format .jpg in the file Images\1
    imgNames = glob.glob(r'*.tif')
    # imgNames is a list with the file names with the format .jpg in file Images\1

    for img in imgNames:
        # part 1 - applying histogram equalization & plotting
        img = cv2.imread(img, 0)
        x = np.arange(0,256,1)
        template_hist = normalCDF(x, 100, 100)
        histMatching(img, template_hist.cumsum())




        print('')