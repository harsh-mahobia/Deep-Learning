import cv2
import numpy as np

def get_range(color):
    
    c = color
    lowerLimit = c[0][0][0] - 10, 80, 100
    upperLimit = c[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit