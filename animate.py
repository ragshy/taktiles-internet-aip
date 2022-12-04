import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import mediapipe as mp

from reference_world import *


# Render
import pygame

pygame.init()
BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 0, 255, 0)
RED = ( 255, 0, 0)
width = 700
height = 700
size = (width, height)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Position Render")
clock = pygame.time.Clock()





mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1,circle_radius=1)




def process(cap):
  ret,img = cap.read()

  img = cv2.cvtColor(cv2.flip(img,1), cv2.COLOR_BGR2RGB)
  img.flags.writeable = False

  results = face_mesh.process(img)
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      # get only relevant landmarks
      head2d = ref2dHeadImagePoints(face_landmarks.landmark,w,h)
      distance = face_landmarks.landmark[1].z*w
      print('s'*50)
      print(distance)
      #print(head2d)
      # solve PnP
      success, rot_vec, trans_vec = cv2.solvePnP(ref3DHeadModel(),head2d,cam_matrix,distortion)
      
      # angles
      rmat, jac = cv2.Rodrigues(rot_vec)
      euler_angles, R, Q, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
      theta_angle = euler_angles[2]
      x = np.arctan2(Qx[2][1], Qx[2][2])*180/math.pi
      phi_angle = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))*180/math.pi
      theta_angle = np.arctan2(Qz[0][0], Qz[1][0])*180/math.pi
      print('+'*34)
      print(x,phi_angle,theta_angle)
      
      # project nose
      noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
      noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rot_vec, trans_vec, cam_matrix, distortion)

      # draw nose line 
      p1 = (int(head2d[0, 0]), int(head2d[0, 1]))
      p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
      cv2.line(img, p1, p2, (110, 220, 0),thickness=2, lineType=cv2.LINE_AA)
      # print tilt
      cv2.putText(img,str(phi_angle),org=(50,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2)
    
    mp_drawing.draw_landmarks(image=img,
                              landmark_list=face_landmarks,
                              connections = mp_face_mesh.FACEMESH_TESSELATION,
                              landmark_drawing_spec = drawing_spec,
                              connection_drawing_spec = drawing_spec)
  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
  return cv2.cvtColor(img,cv2.COLOR_BGR2RGB),phi_angle,theta_angle

# Video Capture
cap = cv2.VideoCapture(0)
ret, img = cap.read()
img =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
h,w,c = img.shape
focal_length = 1553.0
cam_matrix = np.array([[focal_length, 0, 986],
                       [0, focal_length, 499],
                       [0, 0, 1]])
distortion = np.zeros((4,1))
distortion = np.array([0.1525,-1.022,-0.00287,0.000317,1.172])

# Update data
def update(i,x,phi_angles,theta_angles):
  (img,phi_angle,theta_angle) = process(cap)
  im1.set_data(img)
  x.append(round(time.time() - start, 2))
  phi_angles.append(phi_angle)
  theta_angles.append(theta_angle)
  print(x[-1],phi_angles[-1],theta_angles[-1])
  x = x[-50:]
  phi_angles = phi_angles[-50:]
  theta_angles = theta_angles[-50:]
    # Draw x and y lists
  ax2.clear()
  ax2.plot(x,phi_angles)
  ax2.set_ylim([-75, 75])
  ax3.clear()
  ax3.plot(x,theta_angles)
  plt.yticks(np.arange(-75, 75+1, 7.5))

  # Render
  screen.fill(WHITE)
  pygame.draw.rect(screen, BLACK, [width/2-50/2, height-25, 50, 25],0)
  pygame.draw.circle(screen, BLACK, [width/2,height/2], 10)
  line_length = 100
  start_pos = [width/2,height/2]
  end_pos = [start_pos[0]+line_length*math.sin(math.radians(phi_angle)),start_pos[1]+line_length*math.cos(math.radians(phi_angle))]
  print('END'*23,end_pos)
  pygame.draw.line(screen,RED,start_pos=start_pos,end_pos=end_pos)
  pygame.display.update()
  clock.tick(60)


#create two subplots and set properties
fig, axs = plt.subplots(3,1,figsize=(16,9), gridspec_kw={'height_ratios': [3, 1,1]})
ax1 = axs[0]
ax1.axis('off')
ax2 = axs[1]
ax3 = axs[2]


#create two image plots
img, _,_ = process(cap)
im1 = ax1.imshow(img)

x = []
phi_angles = []
theta_angles = []
line, = ax2.plot(x, phi_angles)
line2, = ax3.plot(x, theta_angles)
    
# Animation
start = time.time()
ani = FuncAnimation(fig, update,fargs=(x,phi_angles,theta_angles), interval=1)

def close(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)
    for event in pygame.event.get(): # User did something
      if event.type == pygame.QUIT: # If user clicked close
        pygame.quit()
cid = plt.gcf().canvas.mpl_connect("key_press_event", close)
plt.show()
  

'''
Resolution:
angle range = +63 to -58 = 121
to 15 decimals
-57.99999999999999 to -57.999999999999986 -> ~0.000000000000007

-> #angles = 121/0.000000000000007 = 17_285_714_285_714_286 ~= 17 billarden
'''

print(np.nextafter(-57.999999999999986,63.0))
print(121/0.000000000000007)

pygame.quit()

# elevation degree