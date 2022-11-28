
import cv2
import numpy as np
import math


cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
while True:
    ret, img = cap.read()
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
cap.release()g