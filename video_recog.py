import cv2
from main import display_values
from value_calculator import ValueCalculator
import pprint
import numpy as np
from time import sleep
import dlib

dlib.DLIB_USE_CUDA

def main():
    VIDEO_LOCATION = "tests/dataset/videos/test2.mp4"

    cap = cv2.VideoCapture(VIDEO_LOCATION)

    print("CUDA enabled:", dlib.DLIB_USE_CUDA)

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("videos/output.mp4", fourcc, frame_rate, frame_size, True)

    value_calculator = ValueCalculator(cuda_available=True)

    current_frame = 0
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if current_frame % frame_rate == 0:
                values = value_calculator.get_values(frame)
                print("Current Value:", calculate_total(values))

            display_values(values, frame)

            if writer is not None:
                writer.write(frame)
            

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else: 
            break

        current_frame += 1
    
    cap.release()
    cv2.destroyAllWindows()

def calculate_total(values):
    total = 0
    for value in values:
        total += int(value["value"])
    return total




if __name__ == "__main__":
    main()
