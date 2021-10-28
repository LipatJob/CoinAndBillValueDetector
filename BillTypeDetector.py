import cv2
import numpy as np
import pytesseract


def count_bill_denominator(img, bills):
    img = img.copy()
    bills_values = []

    for bill_area in bills:
        bill_image = get_bill_image(bill_area, img)
        bill_image = get_area_of_interest(bill_image)
        bill_image = apply_preprocess(bill_image)

        cv2.imshow("", bill_image)
        cv2.waitKey()

        output = process_ocr(bill_image)
        bills_values.append(output)

    return {bill: bills_values.count(bill) for bill in bills_values}


def get_bill_image(bill_area, image):
    tl, tr, br, bl = bill_area
    h_start = int(tl[1])
    h_end = int(bl[1])
    w_start = int(tl[0])
    w_end = int(tr[0])

    crop_img = image[h_start:h_end, w_start:w_end].copy()
    return crop_img


def get_area_of_interest(bill):
    height, width, _ = bill.shape

    bill = cv2.normalize(bill, bill, 0, 215, cv2.NORM_MINMAX)
    bill = cv2.cvtColor(bill, cv2.COLOR_BGR2GRAY)

    h_start = int(height * .78)
    h_end = int(height * .98)
    w_start = int(width * .87)
    w_end = int(width * .98)
    crop_img = bill[h_start:h_end, w_start:w_end].copy()

    # resize image for clearer recognition
    scale_percent = 400
    width = int(crop_img.shape[1] * scale_percent / 100)
    height = int(crop_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

    return resized

# if tesseract is not in PATH variable, use this
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def apply_preprocess(bill):
    # Apply Gaussian Blur and Median Blur
    gauss_img = cv2.GaussianBlur(bill, (11, 11), 0)
    median_img = cv2.medianBlur(gauss_img, 7)

    # Apply binary thresholding
    thresh = cv2.adaptiveThreshold(median_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 6)

    # Dilate the image
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=2)

    return img_dilation


def process_ocr(img):
    custom_config = f'--psm 13 outputbase digits -l eng'
    data = pytesseract.image_to_string(img, config=custom_config, lang='eng')
    string = ''.join([char for char in data[:-2 or None]])
    return string
