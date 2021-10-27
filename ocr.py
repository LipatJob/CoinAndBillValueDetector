import cv2
import numpy as np
import pytesseract
import os

# if tesseract is not in PATH variable, use this
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def img_preprocess(file):
    img = cv2.imread(file)

    # normalize image then convert to grayscale
    cv2.normalize(img, img, 0, 215, cv2.NORM_MINMAX)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # crop image to get specific text
    h_start=158
    h_end=475
    w_start=400
    w_end=475
    crop_img = gray[h_start:h_end, w_start:w_end].copy()

    # resize image for clearer recognition
    scale_percent = 400 
    width = int(crop_img.shape[1] * scale_percent / 100)
    height = int(crop_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)

    # Apply Gaussian Blur and Median Blur
    gauss_img = cv2.GaussianBlur(resized,(11,11), 0)
    median_img = cv2.medianBlur(gauss_img, 7)

    # Apply binary thresholding
    ret, thresh = cv2.threshold(median_img, 99, 255, cv2.THRESH_BINARY)

    # Dilate the image
    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=2)

    return img_dilation

def process_ocr(img):
    custom_config = f'--psm 13 outputbase digits -l eng'
    data = pytesseract.image_to_string(img, config=custom_config, lang='eng')
    string = ''.join([char for char in data[:-2 or None]])
    return string

if __name__ == '__main__':
    main_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(main_dir,r'bills\100php.png')
    image = img_preprocess(file)
    result = process_ocr(image)

    img = cv2.imread(file)
    cv2.imshow("Current Bill", img)
    print("Current Bill: ", result)
    cv2.waitKey(0)
