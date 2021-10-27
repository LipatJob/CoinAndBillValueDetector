from scipy.spatial import distance as dist
from currDetector import CurrDetector
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import glob

def main():
    # Change the directory 
    path = glob.glob(r'C:\Users\Mark Anthony Mamauag\Desktop\School Files\Programming\1st Term (2021-2022)\CS124\images\*.jpg')
    detectCurrency(path)

def detectCurrency(path_var):
    pixelsPerMetric = None
    for file in path_var:
        image = cv2.imread(file)

        # FOR CURRENCY DETECTION
        resized = imutils.resize(image, width=300)
        ratio = image.shape[0] / float(resized.shape[0])

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        sd = CurrDetector()

        # FOR COIN TYPE DETECTION
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(gray, (7, 7), 0)

        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        (cnts, _) = contours.sort_contours(cnts)
        
        for c in cnts:
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            shape = sd.detect(c)
                
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

            # FOR COIN TYPE DETECTION
            if cv2.contourArea(c) < 100:
                continue
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if pixelsPerMetric is None:
                pixelsPerMetric = dB / 0.996
                
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            finalDim = (dimA + dimB) / 2

            # print(dimA)
            # print(dimB)
            # print(finalDim)

            if finalDim >= 0.90 and finalDim <= 1.005:
                print('Coin Type: 1-Peso Coin')
            elif finalDim >= 1.13 and finalDim <= 1.17:
                print('Coin Type: 5-Peso Coin')
            elif finalDim >= 1.25 and finalDim <= 1.30:
                print('Coin Type: 10-Peso Coin')
            else:
                print('Invalid Type: Bill Detected')
            
            cv2.imshow("Image", image)
            cv2.waitKey(0)

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    
main()
