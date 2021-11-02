import cv2
import imutils
from imutils import contours
import numpy as np

from bill_value_calculator import get_bill_value
from bills_encoder import get_encoded_bills
from coin_value_calculator import get_coin_value, get_pixel_per_metric


def get_values(image):
    cnts = get_contours(image)
    money_types = [check_type(cnt) for cnt in cnts]
    min_areas = [cv2.minAreaRect(contour) for contour in cnts]
    boxes = [cv2.boxPoints(min_area) for min_area in min_areas]
    items = get_rotated_objects(image, min_areas)

    pixel_per_metric = None
    pre_encoded_faces = None
    values = []
    for money_type, item, box in zip(money_types, items, boxes):
        cv2.imshow("", item)
        cv2.waitKey()

        # process coins
        if money_type == "coin":
            if pixel_per_metric == None: pixel_per_metric = get_pixel_per_metric(box)
            value = get_coin_value(box, pixel_per_metric, item)

        # process bills
        elif money_type == "bill":
            if pre_encoded_faces == None: pre_encoded_faces = get_encoded_bills()
            value = get_bill_value(item, pre_encoded_faces)

        # add value to list
        values.append({
            "type": money_type,
            "value": value,
            "location": tuple(map(tuple, box))
        })

    return values


def get_contours(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)[1]

    # get contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)

    # filter contours that are too small
    area = image.shape[0] * image.shape[1]
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > area * .005]
    
    return cnts


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
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    return "bill" if len(approx) == 4 else "coin"
