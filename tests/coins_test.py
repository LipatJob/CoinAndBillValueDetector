import unittest

from main import get_value_from_image


class CoinsTest(unittest.TestCase):

    def __init__(self):
        self.debug_mode = False

    def test_coins(self):
        self.assertEqual(get_value_from_image(
            "examples/dataset/coins.jpg", self.debug_mode), 22)


if __name__ == '__main__':
    unittest.main()
