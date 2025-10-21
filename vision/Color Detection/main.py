
import cv2
import numpy as np
from utils import get_range
from PIL import Image

# HSV - Hue, Saturation, Value for color detection
# BGR - Blue, Green, Red

cap = cv2.VideoCapture(0)
pink_bgr = np.uint8([[[203, 192, 255]]])  # BGR for pink
pink_hsv = cv2.cvtColor(pink_bgr, cv2.COLOR_BGR2HSV)

print(pink_hsv)

while True:
    ret, frame = cap.read()
    if not ret :
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lowerLimit, upperLimit = get_range(color=pink_hsv)
    
    mask = cv2.inRange(hsv, lowerLimit, upperLimit)
    

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    
    cv2.imshow('hsv image', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

    