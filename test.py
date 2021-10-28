from CoinTypeDetector import (count_coin_denominator, get_pixel_per_metric)
from MoneyTypeDetector import detect_money_type
from BillTypeDetector import count_bill_denominator
import cv2
image = cv2.imread("currencies/coins_and_bills.jpg")

coins, bills = detect_money_type(image)
pixel_per_metric = get_pixel_per_metric(coins[0])

coin_count = count_bill_denominator(image, bills)

print(coin_count)