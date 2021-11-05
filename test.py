import unittest

from main import get_value_from_image


class TestMain(unittest.TestCase):

    def test_coins(self):
        self.assertEqual(get_value_from_image("examples/coins.jpg"), 22)

    def test_bills(self):
        self.assertEqual(get_value_from_image("examples/bills.jpg"), 70)

    def test_both(self):
        self.assertEqual(get_value_from_image("examples/coins_bills.jpg"), 92)

    def test_rotated(self):
        self.assertEqual(get_value_from_image("examples/rotated_bills.png"), 90)

    def test_other_bills(self):
        self.assertEqual(get_value_from_image("examples/other_bills.jpg"), 1100)

    def test_other_image(self):
        self.assertEqual(get_value_from_image("examples/pic.jpg"), 187)


if __name__ == '__main__':
    unittest.main()