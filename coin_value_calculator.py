from scipy.spatial import distance as dist
import cv2


def get_coin_value(box, pixel_per_metric, image):
    diameter = get_diameter(box, pixel_per_metric)

    coin_value = "-1"
    dimensions = [("1", 23), ("5", 25), ("10", 27)]
    epsilon = 1.5
    for current_type, size in dimensions:
        if size - epsilon < diameter < size + epsilon:
            coin_value = current_type
            break

    show_coin(image, coin_value, diameter)
    return coin_value


def get_diameter(cnt, pixel_per_metric):
    (tl, tr, br, bl) = cnt

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    dimA = dA / pixel_per_metric
    dimB = dB / pixel_per_metric
    return (dimA + dimB) / 2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def get_pixel_per_metric(box):
    (tl, tr, br, bl) = box
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    return dB / 23


def show_coin(image, amount, diameter):
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    factor = 300
    ratio = image.shape[1] / image.shape[0]
    width = int(factor)
    height = int(factor * ratio)

    image = cv2.resize(image, (height, width))
    cv2.putText(image, f"Diameter: {diameter:.2f}", (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, f"Amount: {amount}", (0, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("", image)
    cv2.waitKey()
