import cv2
import imutils
from imutils import contours
import numpy as np

from bill_value_calculator import get_bill_value
from bills_encoder import get_encoded_bills
from coin_value_calculator import get_coin_value, get_pixel_per_metric
from debug_utils import resize_image


class ValueCalculator:
    def __init__(self, cuda_available=False):
        if cuda_available:
            print("Using CUDA")
        self.pixel_per_metric = None
        self.pre_encoded_faces = None
        self.cuda_available = cuda_available

    def get_values(self, image, debug_mode=False):
        cnts = self.get_contours(image, debug_mode)
        money_types = [self.check_type(cnt) for cnt in cnts]
        min_areas = [cv2.minAreaRect(contour) for contour in cnts]
        boxes = [cv2.boxPoints(min_area) for min_area in min_areas]
        items = self.get_rotated_objects(image, min_areas)

        values = []
        for money_type, item, box in zip(money_types, items, boxes):
            if debug_mode:
                cv2.imshow("", resize_image(item, 300))
                cv2.waitKey()
            bill_boxes = []
            bill_names = []

            # process coins
            if money_type == "coin":
                if self.pixel_per_metric == None:
                    self.pixel_per_metric = get_pixel_per_metric(box)
                value = get_coin_value(
                    box, self.pixel_per_metric, debug_mode, item)

            # process bills
            if money_type == "bill":
                if self.pre_encoded_faces == None:
                    self.pre_encoded_faces = get_encoded_bills(
                        cuda_available=self.cuda_available)
                value, bill_box, bill_name = get_bill_value(
                    item, self.pre_encoded_faces, self.cuda_available, debug_mode)
                bill_boxes.append(bill_box)
                bill_names.append(bill_name)

            # add value to list
            values.append({
                "type": money_type,
                "value": value,
                "location": tuple(map(tuple, box)),
                "faces": [{
                    "name": bill_name,
                    "location": bill_box
                } for bill_box, bill_name in zip(bill_boxes, bill_names)]
            })

        return values

    def apply_preprocess(self, image):
        image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        image = cv2.medianBlur(image, 7)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    def get_contours(self, image, debug_mode=False):
        image = image.copy()
        image = self.apply_preprocess(image)
        thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        if debug_mode:
            cv2.imshow("", thresh)
            cv2.waitKey()

        # get contours
        cnts = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)

        # filter contours that are too small or too big
        area = image.shape[0] * image.shape[1]
        min_area = area * .001
        max_area = area * 1
        cnts = [cnt for cnt in cnts if min_area <
                cv2.contourArea(cnt) < max_area]

        return cnts

    def get_rotated_objects(self, image, min_area_rects):
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

    def check_type(self, cnt):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        return "bill" if len(approx) == 4 else "coin"
