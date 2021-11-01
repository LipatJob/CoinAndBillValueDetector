import cv2
import numpy as np
import pytesseract


def get_bill_value(bill_image):
    area_of_interest = get_area_of_interest(bill_image)
    area_of_interest = apply_preprocess(area_of_interest)
    value  = read_bill_value(area_of_interest)

    show_bill(area_of_interest, value)

    return value


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


    return img_dilation


def get_area_of_interest(bill):
    height, width = bill.shape

    bill = cv2.normalize(bill, bill, 0, 215, cv2.NORM_MINMAX)

    # get the lower right portion of the image
    h_start = int(height * .76)
    h_end = int(height * .97)
    w_start = int(width * .80)
    w_end = int(width * .97)
    crop_img = bill[h_start:h_end, w_start:w_end].copy()

    # resize image for clearer recognition
    scale = 200
    ratio = crop_img.shape[0]/crop_img.shape[1]
    width = int(scale)
    height = int(scale * ratio)
    dim = (width, height)
    resized = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

    return resized


def read_bill_value(img):
    # if tesseract is not in PATH variable, use this
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    custom_config = f'--psm 13 outputbase digits -l eng'
    data = pytesseract.image_to_string(img, config=custom_config)
    string = ''.join([char for char in data if char.isdigit()])
    return int(string) if string != "" else 0

def show_bill(image, value):
    # helper function for displaying the bill after processing
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    factor = 150
    ratio = image.shape[1] / image.shape[0]
    width = int(factor)
    height = int(factor * ratio)

    image = cv2.resize(image, (height, width))
    cv2.putText(image, f"Value: {value}", (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("", image)
    cv2.waitKey()