import cv2
import numpy as np

from MoneyTypeDetector import apply_preprocess, get_contour


def main():
    img = cv2.imread("currencies/rotated.jpg")
    # points for test.jpg
    img = apply_preprocess(img)

    for cnt in get_contour(img):
        rect = cv2.minAreaRect(cnt)

        # the order of the box points: bottom left, top left, top right, bottom right
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(img, M, (width, height))

        if height > width:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        cv2.imshow("crop_img.jpg", warped)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
