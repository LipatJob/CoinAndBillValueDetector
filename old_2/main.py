from CoinTypeDetector import (count_coin_denominator, get_pixel_per_metric)
from MoneyTypeDetector import detect_money_type
from BillTypeDetector import count_bill_denominator
import cv2


def main():
    test = [
        ("currencies/coins_and_bills.jpg", 86),
        ("bills/1kphp.png", 1000),
        ("bills/20php.png", 20),
        ("bills/50php.png", 50),
        ("bills/100php.png", 100),
        ("bills/200php.png", 200),
        ("bills/500php.png", 500),
    ]

    for image_location, amount in test:
        image = retrieve_image_from_disk(image_location)
        value = CalculateValue(image)
        status = "Passed" if value == amount else "Failed"
        print(status, amount, value)


def GetImage():
    image = retrieve_image_from_disk("")
    # image = preprocess_image(image)
    return image


def CalculateValue(image):
    coins, bills = detect_money_type(image)

    coinCount = count_coin_denominator(image, coins)
    billCount = count_bill_denominator(image, bills)

    value = calculate_value(coinCount, billCount)

    return value


def retrieve_image_from_disk(file_location):
    return cv2.imread(file_location)


def calculate_value(coinCount, billCount):
    sum = 0
    values = {
        "1": 1,
        "5": 5,
        "10": 10,
        "20": 20,
        "50": 50,
        "100": 100,
        "200": 200,
        "500": 500,
        "1000": 1000
    }

    for denomination, count in coinCount.items():
        sum += values.get(denomination, 0) * count

    for denomination, count in billCount.items():
        sum += values.get(denomination, 0) * count

    return sum


main()
