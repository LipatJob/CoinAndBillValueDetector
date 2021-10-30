import cv2
import numpy as np
import pytesseract


def get_bill_value(bill_image):
    area_of_interest = get_area_of_interest(bill_image)
    area_of_interest = apply_preprocess(area_of_interest)
    return process_ocr(area_of_interest)


def apply_preprocess(bill):

    # Apply Gaussian Blur and Median Blur
    gauss_img = cv2.GaussianBlur(bill, (11, 11), 0)
    median_img = cv2.medianBlur(gauss_img, 7)

    # Apply binary thresholding
    thresh = cv2.adaptiveThreshold(
        median_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 6)

    # Dilate the image
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    cv2.imshow("", img_dilation)
    cv2.waitKey()

    return img_dilation


def get_area_of_interest(bill):
    height, width = bill.shape

    bill = cv2.normalize(bill, bill, 0, 215, cv2.NORM_MINMAX)

    h_start = int(height * .80)
    h_end = int(height * .97)
    w_start = int(width * .80)
    w_end = int(width * .97)
    crop_img = bill[h_start:h_end, w_start:w_end].copy()

    # resize image for clearer recognition
    scale_percent = 400
    width = int(crop_img.shape[1] * scale_percent / 100)
    height = int(crop_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

    return resized


def process_ocr(img):
    # if tesseract is not in PATH variable, use this
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    custom_config = f'--psm 13 outputbase digits -l eng'
    data = pytesseract.image_to_string(img, config=custom_config)
    string = ''.join([char for char in data if char.isdigit()])

    return string
