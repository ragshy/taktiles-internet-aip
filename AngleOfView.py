import cv2
import numpy as np

GREEN = (0, 255, 0)
# Video Capture
cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)
while(True):
    ret, img = cam0.read()
    ret1, img1 = cam1.read()
    #img =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h,w,c = img.shape
    h_c = int(h/2)
    w_c = int(w/2)

    cv2.line(img, (0,h_c), (w,h_c) , GREEN, 3)
    cv2.line(img, (w_c,0), (w_c,h) , GREEN, 3)
    cv2.line(img1, (0,h_c), (w,h_c) , GREEN, 3)
    cv2.line(img1, (w_c,0), (w_c,h) , GREEN, 3)

    cv2.imshow('cam0', img)
    cv2.imshow('cam1', img1)

    # Check for user input to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the cameras
cam0.release()
cam1.release()
cv2.destroyAllWindows()

""" 
    # Try with cam matrix and distortion
    focal_length = 1553.0
    cam_matrix = np.array([[focal_length, 0, 986],
                        [0, focal_length, 499],
                        [0, 0, 1]])
    new_cam_matrix = np.array([[1497.2, 0, 986],
                        [0, 1503.7, 495.7],
                        [0, 0, 1]])
    distortion = np.zeros((4,1))
    distortion = np.array([0.1525,-1.022,-0.00287,0.000317,1.172])

    dst = cv2.undistort(img, cam_matrix, distortion, None, new_cam_matrix)
    cv2.imshow('undist_video', dst)
 """