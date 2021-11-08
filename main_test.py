import unittest

from main import get_value_from_image


class MainTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MainTest, self).__init__(*args, **kwargs)
        self.debug_mode = True

    def test_coins(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/coins.jpg", self.debug_mode), 22)

    def test_bills(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/bills.jpg", self.debug_mode), 70)

    def test_both(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/coins_bills.jpg", self.debug_mode), 92)

    def test_rotated(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/rotated_bills.png", self.debug_mode), 90)

    def test_other_bills(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/other_bills.jpg", self.debug_mode), 1100)

    def test_other_image(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/pic.jpg", self.debug_mode), 187)

if __name__ == '__main__':
    unittest.main()
