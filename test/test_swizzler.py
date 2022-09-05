import unittest

from spatial3d import CoordinateAxes, Vector3


class TestSwizzler(unittest.TestCase):
    def setUp(self) -> None:
        self.v1 = Vector3(-1, 2, -3)

    def test__getattr__returns_the_values_of_the_composed_parameters(self) -> None:
        values = self.v1.xxyyzz
        for index, axis in enumerate(CoordinateAxes):
            with self.subTest(case=axis):
                self.assertEqual(values[2 * index], self.v1[axis])
                self.assertEqual(values[2 * index + 1], self.v1[axis])

    def test__getattr__raises_for_an_unknown_parameter(self) -> None:
        with self.assertRaises(AttributeError):
            values = self.v1.xyzrxyz
