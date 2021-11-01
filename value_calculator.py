import cv2
import imutils
from imutils import contours
import numpy as np

from bill_value_calculator import get_bill_value
from coin_value_calculator import get_coin_value, get_pixel_per_metric


def get_values(image):
    cnts = get_contours(image)
    money_types = get_types_money(cnts)
    min_areas = get_min_area_rect(cnts)
    boxes = get_boxes(min_areas)
    items = get_rotated_objects(image, min_areas)

    pixel_per_metric = None

    values = []
    for money_type, item, box in zip(money_types, items, boxes):
        cv2.imshow("", item)
        cv2.waitKey()

        if money_type == "coin":
            if pixel_per_metric == None:
                pixel_per_metric = get_pixel_per_metric(box)
            value = get_coin_value(box, pixel_per_metric, item)
        elif money_type == "bill":
            value = get_bill_value(item)

        values.append({
            "type": money_type,
            "value": value,
            "location": tuple(map(tuple, box))
        })

    return values


def get_contours(image):
    image = image.copy()
    thresh = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)

    area = image.shape[0] * image.shape[1]

    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > area * .005]
    return cnts


def get_types_money(cnts):
    return [check_type(cnt) for cnt in cnts]


def get_min_area_rect(cnts):
    return [cv2.minAreaRect(contour) for contour in cnts]


def get_boxes(min_areas):
    return [cv2.boxPoints(min_area) for min_area in min_areas]


def get_rotated_objects(image, min_area_rects):
    rotated_images = []

    for rect in min_area_rects:
        # the order of the box points: bottom left, top left, top right, bottom right
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(image, M, (width, height))

        if height > width:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        rotated_images.append(warped)

    return rotated_images


def check_type(cnt):
    shape = 'null'
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    if len(approx) == 4:
        shape = 'bill'
    else:
        shape = 'coin'

    return shape
