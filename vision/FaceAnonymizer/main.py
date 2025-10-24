

# input image - any image/live video
# detection of face
# create bouding box/circle
# Blur the inner part 
# output

# packages : opencv, mediapipe

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp

#reading the image 

print(os.path.exists('Z:/Deep-Learning/vision/FaceAnonymizer/image.png'))

img = cv2.imread('Z:/Deep-Learning/vision/FaceAnonymizer/image.png')
H, W, _ = img.shape
detector = mp.solutions.face_detection

if img is not None :
    with detector.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detector :
        # we need RGB image, default is BGR
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = face_detector.process(image_rgb)

        for detection in out.detections:
            print(detection)

    cv2.imshow('Image Window', img)
    cv2.waitKey(0) # Waits indefinitely for a key press
    cv2.destroyAllWindows()
        
else:
    print("Error: Image not loaded.")  



def create_dot(x, y):
    center_coordinates = (x, y)
    dot_radius = 5 # Small radius for a dot
    dot_color = (0, 0, 255) # Red color (BGR)
    thickness = -1 # Fills the circle

    cv2.circle(img, center_coordinates, dot_radius, dot_color, thickness)