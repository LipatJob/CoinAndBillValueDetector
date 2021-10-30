import cv2
from value_calculator import get_values
import pprint


def main():
    img = cv2.imread("currencies/rotated.jpg")
    img = apply_preprocess(img)
    values = get_values(img)

    for value in values:
        pprint.pprint(value)


def apply_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return blurred


main()
