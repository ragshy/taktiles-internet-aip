import math
import time
import cv2
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mediapipe as mp

from reference_world import *



def distance_to_camera(irl_width,focal_length,image_width):
  return (irl_width*focal_length)/image_width

class Renderer():
  
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
      return img,(0,0,0),0

def DLT(P1, P2, point1, point2): 
  #direct linear transform(DLT).
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
    return Vh[3,0:3]/Vh[3,3]

def triangulate(uvs1,uvs2,cam1_mat,cam2_mat,R,T):
    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)

    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = cam1_mat @ RT1 #projection matrix for C1
 
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = cam2_mat @ RT2 #projection matrix for C2

    p3d = DLT(P1, P2, uvs1, uvs2)
    #print('Triangulated point: ', p3d)
    return p3d

def relative_position(p1,p2):
  return p1+p2