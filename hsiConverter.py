import os
import glob
import cv2


def hsiConverter(input_folder):
    folder_name = 'HSI_images'
    os.mkdir(folder_name)
    i = 0
    for img in glob.glob(input_folder + "/*.jpg"):
        image = cv2.imread(img)
        imgHSI = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        cv2.imwrite("HSI_images/hsi%04i.jpg" %i,imgHSI)
        i +=1

