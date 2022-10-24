
import os
import cv2
import numpy as np
from SignalClassifier import ellipse_find,circle_template_match
from ImgResizer import resizeImg
from hsiConverter import hsiConverter


if __name__ == '__main__':
    # test_converted = cv2.imread('images/test_image_active_lights.png',0)
    # circle_template_match(test_converted)

    # input_image_load = cv2.imread('images/GOOD_TRACK_PU_1_AND_4---4_45_1.jpg')
    # ellipse_find(input_image_load)

    # input_image_load = cv2.imread('images/GOOD_TRACK_DWARF_1_AND_4---6_12803_0.jpg')
    # ellipse_find(input_image_load)
    
    # Original images
    input_folder = 'images'
    # Resized Images
    folder = 'Resized'

    if not os.path.isdir(folder):
        resizeImg(input_folder)
    if not os.path.isdir("HSI_images"):
        hsiConverter(folder)
    for dataset_image in os.listdir('HSI_images'):
        image_file = os.path.join('HSI_images', dataset_image)
        print("Showing:",image_file)

        if os.path.isfile(image_file):
            input_image_load = cv2.imread(image_file)
            input_hsv =input_image_load[:,:,2]
            ellipse_find(input_hsv)

# %%
