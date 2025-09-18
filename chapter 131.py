import cv2
import numpy as np 

img = cv2.imread('kkk.jpg')
cv2.namedWindow('image')

#print(img[10,10]) 
#for j in range(100):     
#    for i in range(100):         
#        img.itemset((100+j,100+i,2),255)

img[100:200, 100:200, 2] = 255

px = img[61, 104]       #그린 블루 레드
print (px)

while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)&0xFF
    if k== 27:
        break
cv2.destroyAllWindows()