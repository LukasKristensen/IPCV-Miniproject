import os
import random
import cv2


def circle_template_match(canny_image, original_image, signal_template):
    # Documentation for template matching: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    template_result = cv2.matchTemplate(canny_image,signal_template,cv2.TM_CCORR_NORMED)
    minimum_value, maximum_value, minimum_location, maximum_location = cv2.minMaxLoc(template_result)

    cv2.imshow("result", template_result)
    cv2.imshow("Template result", template_result)

    highlighted_templates = original_image
    cv2.rectangle(highlighted_templates, minimum_location,
                  (minimum_location[1] + len(signal_template[0]), minimum_location[0] + len(signal_template)),
                  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)

    return maximum_value


matches = {}


def ellipse_find(input_image,path_input_image):
    # Documentation: https://docs.opencv.org/3.4/de/d62/tutorial_bounding_rotated_ellipses.html
    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Original input", input_image)

    input_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    input_hsv = input_hsv[:,:,2]
    cv2.imshow("HSV input",input_hsv)

    canny = cv2.Canny(input_hsv, 100, 250)
    cv2.imshow("Canny output",canny)

    signal_class_read = path_input_image.split("_AND_")
    signal_class = int(signal_class_read[0][-1]), int(signal_class_read[1][0])

    template_class = None

    best_template_value = 0
    best_template_location = None

    for template_image in os.listdir('templates'):
        template_location = os.path.join('templates', template_image)
        if os.path.isfile(template_location):
            template_file = cv2.imread(template_location,0)
            result_template = circle_template_match(canny, input_image, template_file)

            if result_template > best_template_value or best_template_value == 0:
                best_template_value = result_template
                best_template_location = template_location
                # cv2.waitKey()  # If manual review

    if best_template_value is not None:
        if "can_pass" in best_template_location:
            template_class = 1, 4
        elif "pass_prohibited" in best_template_location:
            template_class = 1, 2

        if not (signal_class, template_class) in matches:
            matches[(signal_class, template_class)] = 1
        else:
            matches[(signal_class, template_class)] += 1
        if signal_class != template_class:
            print("MISS:", signal_class, template_class)
            print("TEMPLATE:", best_template_location)


if __name__ == '__main__':
    for dataset_image in os.listdir('DATA_SET/test_imgs'):
        image_file = os.path.join('DATA_SET/test_imgs', dataset_image)
        print("Showing:",image_file)

        if os.path.isfile(image_file):
            input_image_load = cv2.imread(image_file, 0)
            ellipse_find(input_image_load,image_file)

    print("RESULT:",matches)

