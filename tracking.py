from utils import *

import cv2
import numpy as np
import time
import csv

# open the file in write mode
f = open('data.csv', 'w',encoding='UTF8', newline='')
# create the csv writer
writer = csv.writer(f)
header = ['Time','Pitch','Roll','Yaw']
writer.writerow(header)

# create video capture
cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape

# create tracker object
tracker = Tracker()
start = time.time()

while True:
  ret, img = cap.read()

  img = cv2.flip(img,1)
  img.flags.writeable = False

  # estimate pose
  img,(pitch,yaw,roll),_=tracker.headpose(img)
  # current seconds elapsed
  timestamp = round(time.time() - start, 2)
  print('*'*34)
  print('Timestamp: ',timestamp)
  print('Pitch, Roll, Yaw: ',pitch,roll,yaw)
  # write a row to the csv file
  writer.writerow([timestamp,pitch,roll,yaw])

  cv2.imshow('img', img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# close video
cv2.destroyAllWindows()
cap.release()

# close file
f.close()