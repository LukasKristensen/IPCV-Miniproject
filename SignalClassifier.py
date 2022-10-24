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
    # circle_template = cv2.resize(circle_template, (20,20))

    print("Shape input:",input_image.shape,"Template shape:",circle_template.shape)

    # Documentation for template matching: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    template_result = cv2.matchTemplate(input_image,circle_template,cv2.TM_CCOEFF_NORMED)
    print(template_result)

    cv2.imshow("result", template_result)
    # cv2.waitKey(0)

    locations = np.where(template_result >= 0.4)
    # print("Locations:",locations)
    cv2.imshow("Template result", template_result)

    highlighted_templates = original_image

    for point in zip(*locations[::-1]):
        print("Point:",point)
        cv2.rectangle(highlighted_templates, point, (point[0]+40,point[1]+20),(random.randint(0,255),random.randint(0,255),random.randint(0,255)),1)

    cv2.imshow("result threshold", highlighted_templates)
    # cv2.waitKey(0)


def ellipse_find(input_image):
    # Documentation: https://docs.opencv.org/3.4/de/d62/tutorial_bounding_rotated_ellipses.html
    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Original input", input_image)
    # cv2.waitKey()

    input_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    input_hsv = input_hsv[:,:,2]
    cv2.imshow("HSV input",input_hsv)
    # cv2.waitKey(0)

    canny = cv2.Canny(input_hsv, 100, 250)
    cv2.imshow("Canny output",canny)

    # Documentation for contours and circularity: https://www.authentise.com/post/detecting-circular-shapes-using-contours
    contours, hierachy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("Contours:",contours[0].shape,contours[1].shape,contours[2].shape)

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
