
'''
import ursina as US
from ursina.prefabs.first_person_controller import FirstPersonController
'''

import cv2
import csv
import argparse
import numpy as np

from utils import *
from TCPConnection import Server,Client

''' Argument Parser'''
parser = argparse.ArgumentParser(description='A Person Tracking system')
parser.add_argument("-s", "--server", help = "Run script as Server or Client", action='store_true')
args = vars(parser.parse_args())

IP_ADDRESS = '10.181.162.16'
print('*'*34)
print(f'The Program is starting... The IP Address is: {IP_ADDRESS}')
print('*'*34)

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
cap1 = cv2.VideoCapture(0)
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

print('*'*34)
print('Camera is setup')
print('*'*34)

''' TCP Connection '''
#print(args)
print('*'*34)
print(f'Is this the server: {args["server"]}')


if args['server']:
  connection = Server(IP_ADDRESS,PORT=9999)
  print('Waiting for client to connect...')
else:
  connection = Client(IP_ADDRESS,PORT=9999)
print('*'*34)

''' Data Tracking '''
# open the file in write mode
f = open(f'data_{args["server"]}.csv', 'w',encoding='UTF8', newline='')
# create the csv writer
writer = csv.writer(f)
header = ['Time','Pitch','Roll','Yaw','X-Pos','Y-Pos','Z-Pos']
writer.writerow(header)



'''Rendering Setup

app = US.Ursina()

US.window.title = 'Room'
US.window.borderless = False
US.window.exit_button.visible = True
US.window.fps_counter.enabled = False

US.mouse.visible = False
US.mouse.locked = True

ground = US.Entity(model="plane", color = US.color.white, scale=(100, 1, 100), collider="box", position=(0, 0, 0))
cube = US.Entity(model='cube',position = (0,2,2), color = US.color.red)
player = FirstPersonController()
'''


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
    cv2.imshow('Cam 0', frame0)
    
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
    writer.writerow([timestamp,array])
    connection.send(array.tobytes())
    data_person2 = connection.receive()
    print('Person 2:')
    print(np.frombuffer(data_person2,dtype=np.float64))
    print('*'*34)

    # Render
    '''
    cube.rotation_y += time.dt * 10                 
    if US.held_keys['up arrow']:                           
      player.world_position += (0, 0, time.dt*10)           
    if US.held_keys['down arrow']:                            
      player.world_position -= (0, 0, time.dt*10) 
    if US.held_keys['left arrow']:
      player.world_rotation_y -=time.dt*50
    if US.held_keys['right arrow']:
      player.world_rotation_y +=time.dt*50
    '''
    # Press Q to stop loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

''' Close all '''
#app.run()
f.close()
cv2.destroyAllWindows()
cap0.release()
cap1.release()