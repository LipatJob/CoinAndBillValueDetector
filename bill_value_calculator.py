import cv2
import face_recognition
import numpy as np
import pytesseract


def get_bill_value(bill_image, pre_encoded_faces):
    bill_image = apply_preprocess(bill_image)
    boxes, value = match_face_and_value(bill_image, pre_encoded_faces)

    value = get_int_value(value)

    #show_bill(bill_image, boxes, value)

    return value


def apply_preprocess(bill):
    # Apply Gaussian Blur and Median Blur
    gauss_img = cv2.GaussianBlur(bill, (11, 11), 0)
    median_img = cv2.medianBlur(gauss_img, 7)

    return median_img


def match_face_and_value(bill, pre_encoded_faces):
    bill = bill.copy()
    boxes = face_recognition.face_locations(bill)
    encodings = face_recognition.face_encodings(bill, boxes)
    
    # initialize the list of names for each face detected
    names = []

    for encoding in encodings:
        # attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(
            pre_encoded_faces["encodings"], encoding)

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

    name_count = {name:names.count(name) for name in set(names)}

    if len(name_count) == 0:
        return boxes, 'null'

    max_name_count = max(name_count, key=name_count.get)
    return boxes, max_name_count


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


def show_bill(image, boxes, value):
    # helper function for displaying the bill after processing
    image = image.copy()

    factor = 150
    ratio = image.shape[1] / image.shape[0]
    width = int(factor)
    height = int(factor * ratio)

    image = cv2.resize(image, (height, width))
    cv2.putText(image, f"Value: {value}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("", image)
    cv2.waitKey()