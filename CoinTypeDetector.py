from scipy.spatial import distance as dist


def count_coin_denominator(image, coins):
    if len(coins) == 0: return {}
    
    pixel_per_metric = get_pixel_per_metric(coins[0]) 
    coin_types = [check_coin_type(coin, pixel_per_metric) for coin in coins]
    return {coin: coin_types.count(coin) for coin in set(coin_types)}


def check_coin_type(coin, pixel_per_metric):
    (tl, tr, br, bl) = coin

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    dimA = dA / pixel_per_metric
    dimB = dB / pixel_per_metric
    final_dim = (dimA + dimB) / 2

    coin_type = "-1"
    dimensions = [("1", 23), ("5", 25), ("10", 27)]
    for current_type, size in dimensions:
        if size - 1.9 < final_dim < size + 1.9:
            coin_type = current_type
            break

    return coin_type


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def get_pixel_per_metric(coin):
    (tl, tr, br, bl) = coin
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    return dB / 23
