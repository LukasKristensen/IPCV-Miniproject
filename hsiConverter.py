import os
import glob
import cv2
folder = 'Resized'
os.mkdir("HSI_images")
i = 0
for img in glob.glob(folder + "/*.jpg"):
    image = cv2.imread(img)
    imgHSI = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    cv2.imwrite("HSI_images/hsi%04i.jpg" %i,imgHSI)
    i +=1
