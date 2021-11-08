import unittest

from main import get_value_from_image


class CoinsTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CoinsTest, self).__init__(*args, **kwargs)
        self.debug_mode = False

    def test_coins(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/coins.jpg", self.debug_mode), 22)


if __name__ == '__main__':
    unittest.main()
