import cv2
from value_calculator import get_values
import pprint
import numpy as np

cap = cv2.VideoCapture('3fps_orig.mp4')
   
# Read until video is completed
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imwrite('pic.jpg', frame)
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    else: 
        break

   
cap.release()
cv2.destroyAllWindows()