
import cv2
import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
from head_pose_estimation import get_2d_points,draw_annotation_box,head_pose_points


cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
while True:
    ret, img = cap.read()
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
cap.release()