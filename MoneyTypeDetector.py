import imutils
import cv2
from imutils import contours
import numpy as np
from imutils import perspective


def detect_money_type(image):
    image = image.copy()

    coins = []
    bills = []

    image = apply_preprocess(image)
    contours = apply_contour(image)

    for contour in contours:
        box = get_bounding_box(contour)
        money_type = check_type(contour)
        if money_type == "bill":
            bills.append(box)
        elif money_type == "coin":
            coins.append(box)

    return (coins, bills)


def apply_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def apply_contour(image):
    thresh = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("", thresh)
    cv2.waitKey()
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)

    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 120]
    return cnts


def get_bounding_box(contour):
    box = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = perspective.order_points(box)
    return box


def check_type(c):
    shape = 'null'
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) == 4:
        shape = 'bill'
    else:
        shape = 'coin'

    return shape
