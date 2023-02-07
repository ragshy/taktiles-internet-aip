import math
import time
import cv2
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mediapipe as mp

# Reference head in 3D
def ref3DHeadModel():
    modelPoints = [[0.0, 0.0, 0.0],         # nose
                   [0.0, -330.0, -65.0],    # chin
                   [-225.0, 170.0, -135.0], # left eye left corner
                   [225.0, 170.0, -135.0],  # right eye right corner
                   [-150.0, -150.0, -125.0],# left mouth corner
                   [150.0, -150.0, -125.0]] # right mouth corner
    return np.array(modelPoints, dtype=np.float64)

# Get certain facial marks (same ones as reference head in 3d) and scale by image dimensions (face_landmarks are normalized)
def ref2dHeadImagePoints(face_landmarks,w,h):
    imagePoints = [[face_landmarks[1].x*w, face_landmarks[1].y*h],
                   [face_landmarks[199].x*w, face_landmarks[199].y*h],
                   [face_landmarks[33].x*w, face_landmarks[33].y*h],
                   [face_landmarks[263].x*w, face_landmarks[263].y*h],
                   [face_landmarks[61].x*w, face_landmarks[61].y*h],
                   [face_landmarks[291].x*w, face_landmarks[291].y*h]]
    return np.array(imagePoints, dtype=np.float64)

'''
def cameraMatrix(fl, center):
    mat = [[fl, 1, center[0]],
                    [0, fl, center[1]],
                    [0, 0, 1]]
    return np.array(mat, dtype=np.float)
    '''


def distance_to_camera(irl_width,focal_length,image_width):
  """ function that calculates the distance to the camera

  INPUT
  irl_width:      reference length in the real world
  focal_length:   focal length of the camera
  image_width:    length in the image

  OUTPUT
  distance to the camera
  """
  return (irl_width*focal_length)/image_width

class Renderer():
  """ class that holds classes that render plots
  """
  class LiveHeadPosePlots():
    """ class to render live headpose angles in a graph over time
    """
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
  """ class to track real-time headpose
  """
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
        
        # decompose to angles
        rotation_mat, jac = cv2.Rodrigues(rot_vec)
        proj_mat = np.hstack((rotation_mat, trans_vec))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_mat)[6] 
        pitch, yaw, roll = [math.radians(i) for i in eulerAngles]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        # project nose to draw line
        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rot_vec, trans_vec, self.camera_mat, self.distortion)

        # draw nose line 
        p1 = (int(head2d[0, 0]), int(head2d[0, 1])) # nose in img
        p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
        cv2.line(img, p1, p2, (110, 220, 0),thickness=2, lineType=cv2.LINE_AA)
        
      # tracked image, draw facial landmarks on top of image
      self.mp_drawing.draw_landmarks(image=img,
                              landmark_list=face_landmarks,
                              connections = self.mp_face_mesh.FACEMESH_TESSELATION,
                              landmark_drawing_spec = self.drawing_spec,
                              connection_drawing_spec = self.drawing_spec)

      return img,(pitch,yaw,roll),p1
    else:
      # if no face detected, return zeros
      return img,(0,0,0),0

def DLT(P1, P2, point1, point2): 
    """ function to solve direct linear transformation problem

    INPUT
    P1,P2:          projection matrices using the cameras
    point1,point2:  image point for which we search 3D coordinates

    OUTPUT
    3D world coordinates
    """
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
    """ function to solve triangulation problem
    INPUT
    uvs1,uvs2:          image points for which we search 3D coordinates
    cam1_mat,cam2_mat:  camera matrices using the cameras
    R,T:                stereo calibrated rotation and translation between             
                        the cameras
    
    OUTPUT
    3D coordinates of multiple points
    """
    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)

    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = cam1_mat @ RT1 #projection matrix for C1
 
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = cam2_mat @ RT2 #projection matrix for C2

    # solve direct linear transform
    p3d = DLT(P1, P2, uvs1, uvs2)
    #print('Triangulated point: ', p3d)
    return p3d

def relative_position(p1,p2):
  return p1+p2