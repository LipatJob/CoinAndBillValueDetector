from CoinTypeDetector import (count_coin_denominator, get_pixel_per_metric)
from MoneyTypeDetector import detect_money_type
from BillTypeDetector import count_bill_denominator
import cv2

def main():
    image = GetImage()
    value = CalculateValue(image)
    print(value)


def GetImage():
    image = retrieve_image_from_disk("currencies/coins_and_bills.jpg")
    # image = preprocess_image(image)
    return image

def CalculateValue(image):
    coins, bills = detect_money_type(image)

    pixel_per_metric = get_pixel_per_metric(coins[0]) 
    coinCount = count_coin_denominator(image, coins, pixel_per_metric)
    billCount = count_bill_denominator(image, bills)

    value = calculate_value(coinCount, billCount)

    return value

def retrieve_image_from_disk(file_location):
    return cv2.imread(file_location)

def calculate_value(coinCount, billCount):
    sum = 0
    values = {
        "1":1,
        "5":5,
        "10":10,
        "20":20,
        "50":50,
        "100":100
    }

    for denomination, count in coinCount.items():
        sum += values.get(denomination, 0) * count
    
    for denomination, count in billCount.items():
        sum += values.get(denomination, 0) * count
    
    return sum

main()