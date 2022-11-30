import numpy as np
import cv2
import glob
import yaml
import os

# https://www.youtube.com/watch?v=nWOx_xXxB70&ab_channel=SyakilaMazmin
# https://drive.google.com/drive/folders/1hG6gykbPDwW3ZCjGo8DLmpNHMSS4FBQi 
#import pathlab

# Issue: For loop not looping

corner_x=10 # number of chessboard corner in x direction
corner_y=7 # number of chessboard corner in x direction

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((corner_y*corner_x,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x,0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from a ll the images.
objpoints = [] # 3d point in real world space
jpegpoints = [] # 2d points in image plane.

source_path = 'C:\\Users\\na86666\\Desktop\\taktiles-internet-aip\\callibration_data' 
#print('images found :',len(os.listdir(source_path))) # count number of images

#pic = glob.glob(r'C:\Users\na86666\Desktop\taktiles-internet-aip\callibration_data\*.jpeg')

# Problem found: Backslash as result through glob, Windows issue
# Liste mit Directories zu jeweiligem Bild
#images = ['r'+ source_path + '/' + f for f in glob.glob('*.jpeg')]  
# use this instead:
images = glob.glob(r'C:\Users\na86666\Desktop\taktiles-internet-aip\callibration_data\*.jpeg')
#print(images)
found = 0
pat_directory = r'C:\Users\na86666\Desktop\taktiles-internet-aip\Patterned_images'

for fname in images: # here, 10 can be changed to whatever number you like to choose
    jpeg = cv2.imread(fname) # capture frame by frame
    cv2.imshow('jpeg', jpeg)
    cv2.waitKey(50)
    #print(fname)
    gray = cv2.cvtColor(jpeg, cv2.COLOR_BGRA2GRAY)
    
    # find the chess noard corners
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
    # if found, ass object points, image points (after refining them)
    if ret == True:
        
        objpoints.append(objp) #Certainly, every loop objp is the same in 3D
        corners2 = cv2.cornerSubPix(gray,corners,(20,5),(-1,-5),criteria)
        jpegpoints.append(corners2)
        # Draw and display the corners
        jpeg = cv2.drawChessboardCorners(jpeg, (corner_x,corner_y), corners2, ret)
        found += 1
        cv2.imshow('chessboard', jpeg)
        cv2.waitKey(50)
        cv2.imwrite(str(pat_directory)+'\pat_img'+str(found)+'.jpeg', jpeg)
print("Number of images used for calibration: ", found)

 # when everything done, release the capture
#cap.release()
cv2.destroyAllWindows()

#calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, jpegpoints, gray.shape[::-1], None, None)

h, w = jpeg.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist , (w,h), 1, (w,h))
#undistort image
undis_directory = r'C:\Users\na86666\Desktop\taktiles-internet-aip\Undistorted_images'
undis_cropped_directory = r'C:\Users\na86666\Desktop\taktiles-internet-aip\Undistorted_cropped_images'
for fname in images: # here, 10 can be changed to whatever number you like to choose
     #print(fname)
     jpeg = cv2.imread(fname) # Capture frame-by-frame
     #cv2.imshow('jpeg', jpeg)
     #cv2.waitkey(500)
     found += 1
     #undistort
     dst = cv2.undistort(jpeg, mtx, dist, None, newcameramtx)
     cv2.imshow('undistorted', dst)
     cv2.waitKey(500)
     cv2.imwrite(str(undis_directory)+'\imdist_img'+str(found)+'.jpeg', jpeg)

     # crop the image
     x, y, w, h = roi
     dst = dst[y:y+h, x:x+w]
     #cv2.imshow('calibration.png',dst)
     cv2.imshow('undistort_cropped', dst)
     cv2.waitKey(500)
     cv2.imwrite(str(undis_cropped_directory)+'\imdist_cropped_img'+str(found)+'.jpeg', jpeg)
     
cv2.destroyAllWindows() 

# transforms the matrix distortion coefficients to writeable lists
data= {'camera_matrix': np.asarray(mtx).tolist(), 'new_camera_matrix': np.asarray(newcameramtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
print("Camera Matrix: ",mtx)
print("Distortion Coefficients: ", dist)
print("Optimized Camera Matrix: ", newcameramtx)
# and save it to a file
with open("calibration_matrix.yaml", "w")as f:
    yaml.dump(data, f)

