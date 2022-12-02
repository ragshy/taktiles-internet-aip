import numpy as np

def ref3DHeadModel():
    modelPoints = [[0.0, 0.0, 0.0],         # nose
                   [0.0, -330.0, -65.0],    # chin
                   [-225.0, 170.0, -135.0], # left eye left corner
                   [225.0, 170.0, -135.0],  # right eye right corner
                   [-150.0, -150.0, -125.0],# left mouth corner
                   [150.0, -150.0, -125.0]] # right mouth corner
    return np.array(modelPoints, dtype=np.float64)

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
