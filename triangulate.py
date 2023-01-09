import numpy as np
from scipy import linalg
        
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
    print('Triangulated point: ', p3d)
    return p3d