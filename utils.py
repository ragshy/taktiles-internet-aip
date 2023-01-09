import pygame
import math
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import mediapipe as mp

from reference_world import *



def distance_to_camera(irl_width,focal_length,image_width):
  return (irl_width*focal_length)/image_width

class Renderer():
  class BirdsEye():
    def __init__(self,width=700,height=700) -> None:
      pygame.init()
      self.BLACK = ( 0, 0, 0)
      self.WHITE = ( 255, 255, 255)
      self.GREEN = ( 0, 255, 0)
      self.RED = ( 255, 0, 0)
      self.width = width
      self.height = height
      self.size = (self.width, self.height)
      self.screen = pygame.display.set_mode(self.size)
      pygame.display.set_caption("Position Render")
      self.clock = pygame.time.Clock()

  class LiveHeadPosePlots():
    def __init__(self,start) -> None:
      #create subplots and set properties
      self.fig, axs = plt.subplots(3,1,figsize=(16,9), gridspec_kw={'height_ratios': [1,1,1]})
      self.ax1,self.ax2,self.ax3= axs[0],axs[1],axs[2]

      self.x,self.yaws,self.pitches,self.rolls = [],[],[],[]

      self.start = start

    def update_plots(self,i,x,yaw,pitch,roll):

      x.append(round(time.time() - self.start, 2))
      yaws.append(yaw)
      pitches.append(pitch)
      rolls.append(roll)

      x = x[-50:]
      yaws = yaws[-50:]
      pitches = pitches[-50:]
      rolls = rolls[-50:]

      # Draw lists
      self.ax1.clear()
      self.ax1.plot(x,yaws)
      self.ax1.set_ylim([-75, 75])
      self.ax2.clear()
      self.ax2.plot(x,pitches)
      plt.yticks(np.arange(-75, 75+1, 7.5))
      self.ax3.clear()
      self.ax3.plot(x,rolls)
      self.ax3.set_ylim([-75, 75])

    def start_plots(self):
      ani = FuncAnimation(self.fig, self.update_plots,fargs=(self.x,self.yaws,self.pitches,self.rolls), interval=1)


  def update_pygame(self,pos,yaw):
    # Render
    self.screen.fill(self.WHITE)
    pygame.draw.rect(self.screen, self.BLACK, [self.width/2-50/2, self.height-25, 50, 25],0) #camera
    pygame.draw.circle(self.screen, self.BLACK, [self.width/2,self.height/2], 10)
    line_length = 100
    start_pos = [self.width/2,self.height/2]
    end_pos = [start_pos[0]+line_length*math.sin(math.radians(yaw)),start_pos[1]+line_length*math.cos(math.radians(yaw))]
    pygame.draw.line(self.screen,self.RED,start_pos=start_pos,end_pos=end_pos)
    pygame.display.update()
    self.clock.tick(60)
  
  def close(self):
    for event in pygame.event.get(): # User did something
      if event.type == pygame.QUIT: # If user clicked close
        pygame.quit()
    if event.key == 'q':
        plt.close(event.canvas.figure)
        pygame.quit()

class Tracker():
  def __init__(self,camera_mat=None,distortion_mat=None) -> None:
    self.mp_face_mesh = mp.solutions.face_mesh
    self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    self.mp_drawing = mp.solutions.drawing_utils
    self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1,circle_radius=1)

    self.camera_mat = camera_mat
    self.distortion = distortion_mat

  def headpose(self,img):
    h,w,c = img.shape

    results = self.face_mesh.process(img)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # get only relevant landmarks
        head2d = ref2dHeadImagePoints(face_landmarks.landmark,w,h)
        #print(head2d)
        # solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(ref3DHeadModel(),head2d,self.camera_mat,self.distortion)
        
        # angles
        rotation_mat, jac = cv2.Rodrigues(rot_vec)
        proj_mat = np.hstack((rotation_mat, trans_vec))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_mat)[6] 
        pitch, yaw, roll = [math.radians(i) for i in eulerAngles]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        # project nose
        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rot_vec, trans_vec, self.camera_mat, self.distortion)

        # draw nose line 
        p1 = (int(head2d[0, 0]), int(head2d[0, 1])) # nose in img
        p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
        cv2.line(img, p1, p2, (110, 220, 0),thickness=2, lineType=cv2.LINE_AA)
      # tracked image
      self.mp_drawing.draw_landmarks(image=img,
                              landmark_list=face_landmarks,
                              connections = self.mp_face_mesh.FACEMESH_TESSELATION,
                              landmark_drawing_spec = self.drawing_spec,
                              connection_drawing_spec = self.drawing_spec)

      return img,(pitch,yaw,roll),p1
    else:
      return img,(0,0,0)

  def locate(self,p1,p2):
    x,y =0,0
    return (x,y)

