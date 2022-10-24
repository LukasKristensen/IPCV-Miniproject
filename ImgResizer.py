import cv2
import os
import glob


# Resize all the images to (50,60)
input_folder = 'images'
os.mkdir("Resized")
i = 0
for img in glob.glob(input_folder + "/*.jpg"):
    image = cv2.imread(img)
    imgRe = cv2.resize(image,(50,60),interpolation = cv2.INTER_AREA)
    cv2.imwrite("Resized/image%04i.jpg" %i,imgRe)
    i +=1