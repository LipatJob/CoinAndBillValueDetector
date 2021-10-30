from scipy.spatial import distance as dist


def get_coin_value(box, pixel_per_metric):
    diameter = get_diameter(box, pixel_per_metric)

    coin_type = "-1"
    dimensions = [("1", 23), ("5", 25), ("10", 27)]
    for current_type, size in dimensions:
        if size - 1.9 < diameter < size + 1.9:
            coin_type = current_type
            break

    return coin_type


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
