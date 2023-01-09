import cv2

cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)

cv2.namedWindow("cam0")
cv2.namedWindow("cam1")

img0_counter = 0
img1_counter = 1

while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if not ret0 and not ret1:
        print("failed to grab frame")
        break

    cv2.imshow("cam0", frame0)
    cv2.imshow("cam1", frame1)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        #img0_name = "cam0_frame_{}.png".format(img0_counter)
        #img1_name = "cam1_frame_{}.png".format(img1_counter)
        cv2.imwrite(str(img0_counter)+'.png', frame0)
        cv2.imwrite(str(img1_counter)+'.png', frame1)
        #print("{} written!".format(img0_name))
        #print("{} written!".format(img1_name))
        img0_counter += 2
        img1_counter += 2



cv2.destroyAllWindows()

cam0.release()
cam1.release()
