

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

print(os.path.exists('Z:/Deep-Learning/vision/Faces/FaceAnonymizer/input.png'))

img = cv2.imread('Z:/Deep-Learning/vision/Faces/FaceAnonymizer/input.png')
H, W, _ = img.shape
detector = mp.solutions.face_detection

if img is not None :
    with detector.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detector :
        # we need RGB image, default is BGR
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = face_detector.process(image_rgb)

        for detection in out.detections:
            locations = detection.location_data
            bbox = locations.relative_bounding_box
            x1 = int(bbox.xmin*W)
            y1 = int(bbox.ymin*H)
            x2 = int(bbox.width*W) 
            y2 = int(bbox.height*H)

            kernel_size = (50, 50) # intensity of blur

            img[y1:y1+y2, x1:x1+x2, :] = cv2.blur(img[y1:y1+y2, x1:x1+x2, :], kernel_size)


    cv2.imshow('Image Window', img)
    cv2.imwrite(os.path.join('Z:/Deep-Learning/vision/Faces/FaceAnonymizer', 'output.png'), img)
    cv2.waitKey(0) # Waits indefinitely for a key press
    cv2.destroyAllWindows()
        
else:
    print("Error: Image not loaded.")  
