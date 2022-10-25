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


def resizeImg(input_folder):
    # os.mkdir("Resized2")
    i = 0

    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)
        # image_read = glob.glob(image_path)
        image_resize = cv2.resize(cv2.imread(image_path), (60,80), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'Resized2/{image_file}', image_resize)
        i += 1
        print("Image file:",image_file)


if __name__ == '__main__':
    folder_path = input("Enter the input path for image folder")
    resizeImg(folder_path)



