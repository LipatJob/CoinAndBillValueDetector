import unittest

from main import get_value_from_image

class BillsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(BillsTest, self).__init__(*args, **kwargs)
        self.debug_mode = True

    def test_20(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/20 Pesos/1.jpg", self.debug_mode), 20)

    def test_50(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/50 Pesos/1.jpg", self.debug_mode), 50)

    def test_100(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/100 Pesos/1.jpg", self.debug_mode), 100)

    def test_500(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/500 Pesos/1.jpg", self.debug_mode), 500)

    def test_1000(self):
        self.assertEqual(get_value_from_image(
            "tests/dataset/examples/1000 Pesos/1.jpg", self.debug_mode), 1000)

if __name__ == '__main__':
    unittest.main()