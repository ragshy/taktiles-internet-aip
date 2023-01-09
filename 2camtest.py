import cv2
from utils import *

import numpy as np
import cv2

# First (HELP) Camera
cap0 = cv2.VideoCapture(0)
focal_length = 1553.0
cam0 = np.array([[focal_length, 0, 986],
                  [0, focal_length, 499],
                  [0, 0, 1]])
dist0 = np.array([0.1525,-1.022,-0.00287,0.000317,1.172])
tracker0 = Tracker(camera_mat=cam0,distortion_mat=dist0)

# Second (MAIN) Camera
cap1 = cv2.VideoCapture(1)
cam1 = np.array([[focal_length, 0, 986],
                  [0, focal_length, 499],
                  [0, 0, 1]])
dist1 = np.array([0.1525,-1.022,-0.00287,0.000317,1.172])
tracker1 = Tracker(camera_mat=cam1,distortion_mat=dist1)

while True:
    # Capture frame-by-frame
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if not ret0 and not ret1:
      break
    
    # Display the resulting frame
    frame0,(_,_,_),nose0=tracker0.headpose(frame0)
    cv2.imshow('Cam 0', frame0)

    # Display the resulting frame
    frame1,(pitch,yaw,roll),nose1=tracker1.headpose(frame1)
    cv2.imshow('Cam 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
# When everything is done, release the capture
cap0.release()
cap1.release()