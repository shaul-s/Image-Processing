import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

if __name__ == '__main__':
    # Find all the names of all the files with the format .jpg in the file Images\1
    imgNames = glob.glob(r'*.tif')
    # imgNames is a list with the file names with the format .jpg in file Images\1

    # reading image and template
    img = cv2.imread(imgNames[0], 0)
    temp = cv2.imread(imgNames[1], 0)
    w, h = temp.shape[::-1]

    # searching template within the image and putting a rectangle on it
    res = cv2.matchTemplate(img, temp, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 0, 2)

    # plotting images
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    #plt.show()

    # applying median filter to remove s&p noise
    img_fixed = cv2.medianBlur(img, 5)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_fixed, cmap='gray')
    plt.title('After median filter'), plt.xticks([]), plt.yticks([])

    res = cv2.matchTemplate(img_fixed, temp, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 0, 2)

    # plotting images
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_fixed, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    plt.show()




