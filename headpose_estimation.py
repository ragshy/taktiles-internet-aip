import mediapipe as mp
import cv2
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1,circle_radius=1)

cap = cv2.VideoCapture(0)

while True:
  ret, img = cap.read()

  img = cv2.cvtColor(cv2.flip(img,1), cv2.COLOR_BGR2RGB)
  img.flags.writeable = False

  results = face_mesh.process(img)

  img =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  h,w,c = img.shape
  # Add determined values from Camera Calibration 
  focal_length = 1 * w
  cam_matrix = np.array([[focal_length, 0, h/2],
                         [0, focal_length, w/2],
                         [0, 0, 1]])

  distortion = np.zeros((4,1))

  face_3d = []
  face_2d = []

  if results.multi_face_landmarks:
    for face in results.multi_face_landmarks:
      for i,lm in enumerate(face.landmark):
        if i==1 or i==33 or i==61 or i==199 or i==263 or i==291:
          if i==1:
            nose_2d = (lm.x * w, lm.y * h)
            nose_3d = (lm.x * w, lm.y * h,lm.z)
        x,y = int(lm.x * w),int(lm.y * h)

        face_2d.append([x,y])
        face_3d.append([x,y,lm.z])
      
      face_2d = np.array(face_2d,dtype=np.float64)
      face_3d = np.array(face_3d,dtype=np.float64)

        #solve PnP
      success, rot_vec, trans_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion)
      print('Rotation Vector :',rot_vec)
        # rot matrix
      Rot, jacobian = cv2.Rodrigues(rot_vec)

        # ANGLES
      euler_angles, R, Q, Qx, Qy, Qz = cv2.RQDecomp3x3(Rot)
        #in degrees
      x = euler_angles[0]*360
      y = euler_angles[1]*360
      z = euler_angles[2]*360
      print('Angles: ',x,y,z)
      if y < -2.5:
        text = "Left"
      elif y > 2.5:
        text = "Right"
      elif x < -2.5:
        text = "Down"
      elif x > 2.5:
        text = "Up"
      else:
        text = 'Straight'
        
      #Draw Direction
      nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec,trans_vec,cam_matrix,distortion)
      
      line_length = 15
      p1 = (int(nose_2d[0]),int(nose_2d[1]))
      p2 = (int(nose_3d_projection.flatten()[0] + y*line_length),int(nose_3d_projection.flatten()[1] - x*line_length))

      cv2.line(img,p1,p2,(255,0,0),3)

      # Draw text
      cv2.putText(img,text,org=(50,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2)

    mp_drawing.draw_landmarks(image=img,
                              landmark_list=face,
                              connections = mp_face_mesh.FACEMESH_TESSELATION,
                              landmark_drawing_spec = drawing_spec,
                              connection_drawing_spec = drawing_spec)
  

  cv2.imshow('img', img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cv2.destroyAllWindows()
cap.release()