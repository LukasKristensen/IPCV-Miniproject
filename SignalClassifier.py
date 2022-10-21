import cv2
import numpy as np

# TODO:
#   - Rectangle extraction (the signal itself)
#   - Circle extraction finding the lights
#       - Validate with circularity?
#   - Classification of which lights are active
#   - Evaluate performance with confusion matrix


def circle_template_match(input_image):
    circle_template = cv2.imread('templates/circle.png',0)
    circle_template = cv2.resize(circle_template, (20,20))

    # Documentation for template matching: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    template_result = cv2.matchTemplate(input_image,circle_template,cv2.TM_CCOEFF_NORMED)
    print(template_result)

    cv2.imshow("result", template_result)
    cv2.waitKey(0)

    locations = np.where(template_result >= 0.4)
    print("Locations:",locations)

    highlighted_templates = cv2.imread('images/test_image_active_lights.png')

    for point in zip(*locations[::-1]):
        print("Point:",point)
        cv2.rectangle(highlighted_templates, point, (point[0]+20,point[1]+20),(0,255,255),2)

    cv2.imshow("result threshold", highlighted_templates)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_converted = cv2.imread('images/test_image_active_lights.png',0)
    circle_template_match(test_converted)

