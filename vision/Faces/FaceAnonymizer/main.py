import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp

detector = mp.solutions.face_detection

def image_processor(img, face_detector):
    H, W, _ = img.shape
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detector.process(image_rgb)
    if not out.detections:
        return img

    for detection in out.detections:
        locations = detection.location_data
        bbox = locations.relative_bounding_box
        x1 = int(bbox.xmin * W)
        y1 = int(bbox.ymin * H)
        x2 = int((bbox.xmin + bbox.width) * W)
        y2 = int((bbox.ymin + bbox.height) * H)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            roi = cv2.blur(roi, (50, 50))
            img[y1:y2, x1:x2] = roi

    return img 

cap = cv2.VideoCapture(0)
with detector.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detector:
    while True:
        ret, frame = cap.read()
        if ret is False:
            break

        frame = image_processor(frame, face_detector)
        cv2.imshow('Image Window', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
