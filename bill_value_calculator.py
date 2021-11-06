import cv2
import face_recognition
import numpy as np
from debug_utils import draw_bounding_box, resize_image


def get_bill_value(bill_image, pre_encoded_faces, cuda_available=False, debug_mode=False):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(bill_image, (0, 0), fx=0.5, fy=0.5)
    small_frame = cv2.normalize(small_frame, small_frame, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    boxes, names = match_face_names(
        rgb, pre_encoded_faces, cuda_available)
    value = get_value_from_names(names)

    if debug_mode:
        show_bill(small_frame, boxes, names, value)

    return value, boxes, names


def match_face_names(bill, pre_encoded_faces, cuda_available):
    bill = bill.copy()
    model = "cnn" if cuda_available else "hog"
    boxes = face_recognition.face_locations(bill, model=model)
    unknown_encodings = face_recognition.face_encodings(bill, boxes)

    # initialize the list of names for each face detected
    names = []
    for encoding in unknown_encodings:
        # attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(
            pre_encoded_faces["encodings"], encoding, tolerance=0.4)

        # get only the names of those matched faces
        matched_names = [name for name, matched in zip(
            pre_encoded_faces["names"], matches) if matched]

        # count how many times the the name was matched
        counts = {matched_name: matched_names.count(
            matched_name) for matched_name in set(matched_names)}

        # get the name that was matched the most
        name = max(counts, key=counts.get) if len(counts) > 0 else "Unknown"

        # add the name to the list of matched names
        names.append(name)

    return boxes, names


def get_value_from_names(names):
    values = {
        "Manuel L. Quezon": 20,
        "Sergio Osmena": 50,
        "Manuel Roxas": 100,
        "Diosdado P. Macapagal": 200,
        "Corazon Aquino": 500,
        "Benigno Aquino Jr": 500,
        "Jose Abad Santos": 1000,
        "Vicente Lim": 1000,
        "Josefa Llanes Escoda": 1000
    }

    for name in names:
        if name in values:
            return values[name]

    return -1


def get_int_value(value):
    int_values = {
        "20 Pesos": 20,
        "50 Pesos": 50,
        "100 Pesos": 100,
        "500 Pesos": 500,
        "1000 Pesos": 1000,
        "null": 0
    }

    return int_values.get(value, None)


def get_value_of_names(names):
    values = {
        "Manuel L. Quezon": 20,
        "Sergio Osmeña": 50,
        "Manuel A. Roxas": 100,
        "Diosdado P. Macapagal": 200,
        "Corazon C. Aquino": 500,
        "Benigno S. Aquino Jr": 500,
        "José Abad Santos": 1000,
        "Vicente Lim": 1000,
        "Josefa Llanes Escoda": 1000
    }

    value = None
    for name in names:
        if name not in values:
            return None
        current_value = values[name]
        if value != None and current_value != value:
            return None
        value = current_value
    return value


def show_bill(image, boxes, names, value):
    # helper function for displaying the bill after processing
    image = image.copy()

    for box, name in zip(boxes, names):
        top, right, bottom, left = box
        draw_bounding_box(image, name, left, top, right,
                          bottom, (255, 0, 0), (255, 0, 0))

    cv2.putText(image, f"Value: {value}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.namedWindow("show bill", cv2.WINDOW_NORMAL)
    cv2.imshow("show bill", image)
    cv2.waitKey()
