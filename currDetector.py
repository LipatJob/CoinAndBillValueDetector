import cv2

class CurrDetector:
    def __init__(self):
        pass

    def detect(self, c):
        shape = 'null'
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 4:
            shape = 'bill'
        else:
            shape = 'coin'

        return shape