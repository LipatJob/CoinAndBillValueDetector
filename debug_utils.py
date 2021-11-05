import cv2

def resize_image(image, factor):
    image = image.copy()
    ratio = image.shape[1] / image.shape[0]
    width = int(factor)
    height = int(factor * ratio)

    image = cv2.resize(image, (height, width))
    return image