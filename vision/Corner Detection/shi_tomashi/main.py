import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_corners(name):
    img = cv2.imread(name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #you can test out various values as parameters for the below functions
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.20, 10)
    print("default", corners)
    corners = np.intp(corners)
    print("after processing", corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),6,255,2)        
    plt.imshow(img)
    plt.show()


find_corners('img.png')
find_corners('chess.png')