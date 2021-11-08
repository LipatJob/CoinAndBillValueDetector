import cv2
from debug_utils import draw_bounding_box
from value_calculator import ValueCalculator
import pprint
import numpy as np


def main():
    # Please see "test.py" for testing the dataset

    # Uncomment pictures to be processed
    IMAGE_LOCATION = [
        # "tests/dataset/examples/coins.jpg",
        # "tests/dataset/examples/bills.jpg",
        #"tests/dataset/examples/coins_bills.jpg",
        # "tests/dataset/examples/rotated_bills.png",
        # "tests/dataset/examples/other_bills.jpg",
        # "tests/dataset/pic.jpg"
        "tests/dataset/examples/1000 Pesos/1.jpg",
    ]

    for location in IMAGE_LOCATION:
        print("Value:", get_value_from_image(location, True))


def get_value_from_image(image_location, debug_mode = False):
    img = cv2.imread(image_location)
    original_img = img.copy()

    display_image(img)
    value_calculator = ValueCalculator()
    
    values = value_calculator.get_values(img, debug_mode)

    display_values(values, original_img)
    print("Value:", calculate_total(values))

    return calculate_total(values)


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
        names_point = type_point.copy()

        type_point[1] += 30
        value_point[1] += 70
        names_point[1] += 100
        cv2.putText(image, value["type"], type_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(
            value["value"]) + " PHP", value_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if len(value["names"]) > 0:
            names = ", ".join(value["names"])
            cv2.putText(image, names, names_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    factor = 700
    ratio = image.shape[0] / image.shape[1]
    width = int(factor)
    height = int(factor * ratio)

    image = cv2.resize(image, (width, height))

    cv2.imshow("", image)
    cv2.waitKey()


if __name__ == "__main__":
    main()