import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.io as spIO
import matplotlib.animation as animation

if __name__ == '__main__':
    # loading the animation
    matF = spIO.loadmat(r'video1.mat')
    temp = cv2.imread(r'phone.tif', 0)
    w, h = temp.shape[: :-1]
    video = matF['video1']
    images = []
    fig, ax = plt.subplots()

    # template matching and appending frames to list
    for i in range(video.shape[2]):
        frame = video[:,:,i]
        res = cv2.matchTemplate(frame, temp, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        cv2.rectangle(frame, top_left, bottom_right, 0, 3)
        images.append([ax.imshow(frame, cmap='gray')])

    # plotting animation
    ani = animation.ArtistAnimation(fig, images, interval=200, blit=True, repeat_delay=1e9)
    plt.axis('off')
    plt.show()
