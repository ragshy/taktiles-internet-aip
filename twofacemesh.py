import cv2
import numpy as np
from utils import *
import math

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

tracker = Tracker()

# Calculate Euler Distance
def distanceCalculate(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

while True:
    # Capture frame-by-frame
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    size0 = frame0.shape
    size1 = frame1.shape
    frame0 = cv2.flip(frame0,1)
    frame1 = cv2.flip(frame1,1)
    frame0.flags.writeable = False
    frame1.flags.writeable = False

    if (ret0):
        # Display the resulting frame
        frame0,(pitch0,yaw0,roll0,nose0)=tracker.headpose(frame0)
        print('*'*34)
        print('Pitch0, Roll0, Yaw0, Nose0: ',pitch0,roll0,yaw0,nose0)
        cv2.imshow('Cam 0', frame0)
    if (ret1):
        # Display the resulting frame
        frame1,(pitch1,yaw1,roll1,nose1)=tracker.headpose(frame1)
        print('*'*34)
        print('Pitch1, Roll1, Yaw1, Nose1: ',pitch1,roll1,yaw1,nose1)
        cv2.imshow('Cam 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
# When everything is done, release the capture
cap0.release()
cap1.release()
