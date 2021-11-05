import cv2
import imutils
from imutils import contours
import numpy as np

img = cv2.imread("pic.jpg")

cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(image, (3, 3), 0)
thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("thresh", thresh)
cv2.waitKey()
cv2.destroyAllWindows()
