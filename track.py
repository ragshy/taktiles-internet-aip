
import cv2
import csv
import argparse
import numpy as np
import datetime

from utils import *
from TCPConnection import Server,Client

''' Argument Parser'''
parser = argparse.ArgumentParser(description='A Person Tracking system')
parser.add_argument("-s", "--server", help = "Run script as Server or Client", action='store_true')
args = vars(parser.parse_args())

IP_ADDRESS = '172.20.10.2'
print('*'*34)
print(f'The Program is starting... The IP Address is: {IP_ADDRESS}')
print('*'*34)

'''Camera Setup'''
# First (HELP) Camera
cap0 = cv2.VideoCapture(1)
focal_length = 1553.0
cam0 = np.array([[focal_length, 0, 986],
                  [0, focal_length, 499],
                  [0, 0, 1]])
dist0 = np.array([0.1525,-1.022,-0.00287,0.000317,1.172])
tracker0 = Tracker(camera_mat=cam0,distortion_mat=dist0)

# Second (MAIN) Camera
cap1 = cv2.VideoCapture(2)
cam1 = np.array([[focal_length, 0, 986],
                  [0, focal_length, 499],
                  [0, 0, 1]])
dist1 = np.array([0.1525,-1.022,-0.00287,0.000317,1.172])
tracker1 = Tracker(camera_mat=cam1,distortion_mat=dist1)

# Stereo Calibration
R=  np.array([[ 9.99958646e-01, -6.68838720e-04,  9.06967305e-03],
 [ 1.23416009e-03,  9.98046116e-01, -6.24694146e-02],
 [-9.01017000e-03,  6.24780247e-02,  9.98005668e-01]])
T=np.array([[8.55364546],
 [0.25676638],
 [0.68013493]])
'''
R =  np.array([[ 0.999698,   -0.01712592,  0.01762426],
 [ 0.01583252,  0.99734504,  0.07107888],
 [-0.01879476, -0.07077837,  0.99731498]])
T =  np.array([[-8.11192645],
 [-0.15830232],
 [ 0.3807869 ]])
'''
print('*'*34)
print('Camera is setup')
print('*'*34)

''' TCP Connection '''
#print(args)
print('*'*34)
print(f'Is this the server: {args["server"]}')

"""
if args['server']:
  print('Waiting for client to connect...')
  connection = Server(IP_ADDRESS,PORT=9999)
else:
  connection = Client(IP_ADDRESS,PORT=9999)
print('*'*34)
"""

''' Data Tracking '''

'''Main Loop'''
#Start time
start = time.time()
#def update():
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
    #cv2.imshow('Cam 0', frame0)
    
    # Estimate pose and detect nose with main camera image
    frame1,(pitch,yaw,roll),nose1=tracker1.headpose(frame1)
    cv2.imshow('Cam 1', frame1)

    # Triangulate to find 3D coordinates of nose
    if nose0==0 or nose1==0:
      x,y,z = 0,0,0
    else:
      p3d = triangulate(nose0,nose1,cam0,cam1,R,T)
      #print(p3d)
      x,y,z = p3d[0],p3d[1],p3d[2]
      
    # Save and log data
    # current seconds elapsed
    timestamp = round(time.time() - start, 2)
    print('Timestamp: ',timestamp)
    print('Pitch, Roll, Yaw: ',pitch,roll,yaw)
    print('X,Y,Z Coordinates: ',x,y,z)
    print('*'*34)
    # write a row to the csv file
    array = np.rint(np.array([pitch,roll,yaw,x,y,z]))
    #connection.send(array.tobytes())
    #data_person2 = connection.receive()
    print('*'*34)

    # Render

    # Press Q to stop loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

''' Close all '''
cv2.destroyAllWindows()
cap0.release()
cap1.release()