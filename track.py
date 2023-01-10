import cv2
import csv
import numpy as np

from utils import *

''' Data Tracking '''
# open the file in write mode
f = open('data.csv', 'w',encoding='UTF8', newline='')
# create the csv writer
writer = csv.writer(f)
header = ['Time','Pitch','Roll','Yaw','X-Pos','Y-Pos','Z-Pos']
writer.writerow(header)


'''Camera Setup'''
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

# Stereo Calibration
R = np.array([[ 0.99984487, -0.00711347 , 0.01611293],
 [ 0.00717558 , 0.99996704, -0.00379977],
 [-0.01608537 , 0.0039148,   0.99986296]])
T = np.array([[ 4.23830745],
 [ 0.62222669],
 [-1.61017813]])

'''Main Loop'''
#Start time
start = time.time()
while True:
    # Capture 2 frames
    ret0, frame0 = cap0.read()
    frame0 = cv2.flip(frame0,1)
    ret1, frame1 = cap1.read()
    frame1 = cv2.flip(frame1,1)
    if not ret0 and not ret1:
      break
    
    # Detect face and nose with helper camera image
    frame0,(_,_,_),nose0=tracker0.headpose(frame0)
    cv2.imshow('Cam 0', frame0)

    # Estimate pose and detect nose with main camera image
    frame1,(pitch,yaw,roll),nose1=tracker1.headpose(frame1)
    cv2.imshow('Cam 1', frame1)

    # Triangulate to find 3D coordinates of nose
    if nose0==0 or nose1==0:
      x,y,z = 0,0,0
    else:
      p3d = triangulate(nose0,nose1,cam0,cam1,R,T)
      print(p3d)
      x,y,z = p3d[0],p3d[1],p3d[2]
    # Save and log data
    # current seconds elapsed
    timestamp = round(time.time() - start, 2)
    print('Timestamp: ',timestamp)
    print('Pitch, Roll, Yaw: ',pitch,roll,yaw)
    print('X,Y,Z Coordinates: ',x,y,z)
    print('*'*34)
    # write a row to the csv file
    writer.writerow([timestamp,pitch,roll,yaw,x,y,z])

    # Press Q to stop loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

''' Close all '''
f.close()
cv2.destroyAllWindows()
cap0.release()
cap1.release()