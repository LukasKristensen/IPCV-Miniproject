import cv2
import os
import glob


# Resize all the images to (60,80)


def resizeImg(input_folder):
    os.mkdir("Resized")
    i = 0
    for img in glob.glob(input_folder + "/*.jpg"):
        image = cv2.imread(img)
        imgRe = cv2.resize(image,(60,80),interpolation = cv2.INTER_AREA)
        cv2.imwrite("Resized/image%04i.jpg" %i,imgRe)
        i +=1
