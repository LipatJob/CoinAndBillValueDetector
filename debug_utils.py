import cv2

def resize_image(image, factor):
    image = image.copy()
    ratio = image.shape[1] / image.shape[0]
    width = int(factor)
    height = int(factor * ratio)

    image = cv2.resize(image, (height, width))
    return image
    

def draw_bounding_box(image, label, startX, startY, endX, endY, boxcolor = (0, 0, 255), labelcolor = (0, 0, 255)):
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.rectangle(image, (startX, startY), (endX, endY), boxcolor, 2)
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, labelcolor, 2)