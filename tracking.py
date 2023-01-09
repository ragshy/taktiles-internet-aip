from utils import *

import cv2
import numpy as np
import math


cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape

tracker = Tracker()


while True:
  ret, img = cap.read()

  img = cv2.flip(img,1)
  img.flags.writeable = False

  img,(pitch,yaw,roll)=tracker.headpose(img)
  print('*'*34)
  print('Pitch, Roll, Yaw: ',pitch,roll,yaw)

  cv2.imshow('img', img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cv2.destroyAllWindows()
cap.release()