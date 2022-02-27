import numpy as np
import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

number = 0
while True:
    ret, frame = capture.read()
    #cv2.imshow("VideoFrame", frame)

    cv_image = cv2.flip(frame,-1)
    #cv2.imshow("cv_image", cv_image)

    x=280; y=240; w=40; h=40
    roi_img = cv_image[y:y+h, x:x+w]     
    cv2.imshow('roi_img', roi_img)

    # https://hoony-gunputer.tistory.com/entry/OpenCv-python-%EA%B8%B0%EC%B4%88-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%9D%BD%EA%B3%A0-%EC%A0%80%EC%9E%A5%ED%95%98%EA%B8%B0
    name = '/root/catkin_ws/src/inception_v3/dataset/9/' + str(number) + '.jpg'
    cv2.imwrite(name, roi_img)

    number = number + 1

    

    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        cv2.waitKey()
    else:
        print k # else print its value
    
    

capture.release()
cv2.destroyAllWindows()