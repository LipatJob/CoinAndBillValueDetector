import cv2
import numpy as np
import pytesseract
import os

# if tesseract is not in PATH variable, use this
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 

main_dir = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(main_dir,r'bills\100php.png')

print(file)

img = cv2.imread(file)

cv2.normalize(img, img, 0, 215, cv2.NORM_MINMAX)

cv2.imshow('normalize img', img)
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray img', gray)
cv2.waitKey(0)

print(gray.shape)

h_start=158
h_end=475

w_start=400
w_end=475

crop_img = gray[h_start:h_end, w_start:w_end].copy()
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

scale_percent = 400 # percent of original size
width = int(crop_img.shape[1] * scale_percent / 100)
height = int(crop_img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)

gauss_img = cv2.GaussianBlur(resized,(11,11), 0)
cv2.imshow("gauss image", gauss_img)
cv2.waitKey(0)

median_img = cv2.medianBlur(gauss_img, 7)
cv2.imshow("median", median_img)
cv2.waitKey(0)

ret, thresh = cv2.threshold(median_img, 99, 255, cv2.THRESH_BINARY)

cv2.imshow('Binary image', thresh)
cv2.waitKey(0)

kernel = np.ones((3,3), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=2)

cv2.imshow('dilate image', img_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(14):
    try:
        custom_config = f'--psm {i} outputbase digits -l eng'
        string = pytesseract.image_to_string(img_dilation, config=custom_config, lang='eng')
        print(list(string), f"psm{i}")
        d = pytesseract.image_to_data(img_dilation, output_type=pytesseract.Output.DICT)

        n_boxes = len(d['level'])
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(img_dilation, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('img', img_dilation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)

        continue
