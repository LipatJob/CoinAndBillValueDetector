import cv2
from value_calculator import get_values
import pprint
import numpy as np


def main():
    IMAGE_LOCATION = "dataset/rotated_bills.png"
    
    get_value_from_image(IMAGE_LOCATION)


def get_value_from_image(image_location):
    img = cv2.imread(image_location)
    original_img = img.copy()

    img = apply_preprocess(img)
    values = get_values(img)

    display_values(values, original_img)

    return calculate_total(values)
    

def display_values(values, image):
    for value in values:
        box = np.int0(value["location"])

        cv2.drawContours(image,[box],0,(0,0,255),2)
        type_point = max(box, key=lambda point:point[1]).copy()
        value_point = type_point.copy()

        type_point[1] += 30
        value_point[1] += 70
        cv2.putText(image, value["type"], type_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(value["value"]) + " PHP", value_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    factor = 700
    ratio = image.shape[0] / image.shape[1]
    width = int(factor)
    height = int(factor * ratio)

    image = cv2.resize(image, (width, height))

    cv2.imshow("", image)
    cv2.waitKey()

def apply_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return blurred

def calculate_total(values):
    total = 0
    for value in values:
        total += int(value["value"])
    return total



main()
