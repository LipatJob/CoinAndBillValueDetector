import cv2
from value_calculator import get_values
import pprint
import numpy as np
from time import sleep


def main():
    VIDEO_LOCATION = "3fps_orig.mp4"

    cap = cv2.VideoCapture(VIDEO_LOCATION)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = apply_preprocess(frame)
            values = get_values(img)

            display_values(values, frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else: 
            break
        
    
    cap.release()
    cv2.destroyAllWindows()


def get_value_from_image(current_frame):
    img = current_frame
    original_img = img.copy()

    #display_image(img)

    img = apply_preprocess(img)
    values = get_values(img)

    #display_values(values, original_img)

    return calculate_total(values)


def apply_preprocess(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    return blurred


def calculate_total(values):
    total = 0
    for value in values:
        total += int(value["value"])
    return total


def display_image(image):
    factor = 700
    ratio = image.shape[0] / image.shape[1]
    width = int(factor)
    height = int(factor * ratio)

    image = cv2.resize(image, (width, height))

    cv2.imshow("", image)
    cv2.waitKey()


def display_values(values, image):
    for value in values:
        box = np.int0(value["location"])

        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        type_point = max(box, key=lambda point: point[1]).copy()
        value_point = type_point.copy()

        type_point[1] += 30
        value_point[1] += 70
        cv2.putText(image, value["type"], type_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(
            value["value"]) + " PHP", value_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    factor = 700
    ratio = image.shape[0] / image.shape[1]
    width = int(factor)
    height = int(factor * ratio)

    image = cv2.resize(image, (width, height))
    cv2.imshow("current_frame", image)


if __name__ == "__main__":
    main()
