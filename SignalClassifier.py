import os
import random

import cv2
import numpy as np

# TODO:
#   - Circle extraction finding the lights
#       - Validate with circularity?
#   - Classification of which lights are active
#   - Evaluate performance with confusion matrix


def circle_template_match(input_image, original_image):
    circle_template = cv2.imread('templates/edge_light.png',0)

    # Documentation for template matching: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    template_result = cv2.matchTemplate(input_image,circle_template,cv2.TM_CCOEFF_NORMED)
    cv2.imshow("result", template_result)

    locations = np.where(template_result >= 0.4)
    cv2.imshow("Template result", template_result)

    highlighted_templates = original_image

    for point in zip(*locations[::-1]):
        print("Point:",point)
        cv2.rectangle(highlighted_templates, point, (point[0]+40,point[1]+20),(random.randint(0,255),random.randint(0,255),random.randint(0,255)),1)

    cv2.imshow("result threshold", highlighted_templates)


def ellipse_find(input_image):
    # Documentation: https://docs.opencv.org/3.4/de/d62/tutorial_bounding_rotated_ellipses.html
    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Original input", input_image)

    input_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    input_hsv = input_hsv[:,:,2]
    cv2.imshow("HSV input",input_hsv)

    canny = cv2.Canny(input_hsv, 100, 250)
    cv2.imshow("Canny output",canny)

    circle_template_match(canny, input_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    # test_converted = cv2.imread('images/test_image_active_lights.png',0)
    # circle_template_match(test_converted)

    # input_image_load = cv2.imread('images/GOOD_TRACK_PU_1_AND_4---4_45_1.jpg')
    # ellipse_find(input_image_load)

    # input_image_load = cv2.imread('images/GOOD_TRACK_DWARF_1_AND_4---6_12803_0.jpg')
    # ellipse_find(input_image_load)

    for dataset_image in os.listdir('Resized'):
        image_file = os.path.join('Resized', dataset_image)
        print("Showing:",image_file)

        if os.path.isfile(image_file):
            input_image_load = cv2.imread(image_file, 0)
            ellipse_find(input_image_load)