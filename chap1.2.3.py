import numpy as np 
import cv2

img = cv2.imread('aaa.jpg')
#img = np.zeros((512,512,3), np.uint8)
img = cv2.line(img,(0,0),(736,736),(255,0,0),5)
img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
img = cv2.circle(img,(447,63), 63, (0,0,255), 10)
img = cv2.ellipse(img,(368,368),(100,50),0,0,180,255,-1) #180이 180도 전부 있다 기준 90하면 4분의 1만 있음
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 2,(255,255,255),12,cv2.LINE_AA)


cv2.imshow('frame',img)
k = cv2.waitKey(0)
if k == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()
