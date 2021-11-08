This is a short guide on how our project is structured. 

To detect the value of money in an image, see main.py. You may uncomment items in the IMAGE_LOCATION array to test out different images.

To detect the value of money in a video, see video_recog.py. You may change the value of INPUT_VIDEO_LOCATION to select a different video. You can change where the output video is located by changing OUTPUT_VIDEO_LOCATION

There are also bills_test.py, main_test.py, coins_test.py which could be used to test the image recognition. It was originally located in the tests folder but to make it easy to use without test runners, we have placed it in the root directory.

Speaking of the tests folder, it contains the test images which we use for the examples and test cases.

There is also the dataset folder which contains the data that we use for encoding.

The video folder also contains the input videos and the resulting videos after applying face recognition.

The model folder just contains the encoded faces and labels.

value_calculator.py contains the logic for calculating the total value of money in an image. It also determines whether it is a coin or a bill and it also does some preprocessing.

coin_value_calculator.py contains the logic for getting the value of a coin. It calculates the diameter of the contour of the coin to do this.

bill_value_calculator.py contains the logic for getting the value of a bill. It uses face recognition to do this.
