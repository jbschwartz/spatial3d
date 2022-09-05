import unittest

from spatial3d import Dual, Quaternion, dual, quaternion
from spatial3d.vector3 import Vector3


class TestDual(unittest.TestCase):
    def setUp(self):
        self.r1 = Quaternion(1, 2, 3, 4)
        self.d1 = Quaternion(0, 1, 2, 3)
        self.r2 = Quaternion(0, -1, 0, 0)
        self.d2 = Quaternion(1, 3, -4, 5)

        self.dq1 = Dual(self.r1, self.d1)
        self.dq2 = Dual(self.r2, self.d2)

        self.s = 2.5

    def test__abs__returns_the_absolute_values(self) -> None:
        self.assertEqual(abs(self.dq1), Dual(abs(self.r1), abs(self.d1)))

    def test__add__adds_two_duals(self) -> None:
        self.assertEqual(self.dq1 + self.dq2, Dual(self.r1 + self.r2, self.d1 + self.d2))

    def test__add__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.dq1.__add__(2) == NotImplemented)

    def test__eq__returns_true_for_the_zero_dual_and_zero(self) -> None:
        self.assertTrue(Dual(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 0)) == 0)
        self.assertTrue(Dual(0, 0) == 0)

    def test__eq__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.dq1.__eq__(2) == NotImplemented)
        self.assertTrue(self.dq1.__eq__("string") == NotImplemented)

    def test__mul__returns_the_dual_product(self) -> None:
        expected = Dual(self.r1 * self.r2, self.r1 * self.d2 + self.d1 * self.r2)
        self.assertEqual(self.dq1 * self.dq2, expected)

        expected = Dual(self.r2 * self.r1, self.r2 * self.d1 + self.d2 * self.r1)
        self.assertEqual(self.dq2 * self.dq1, expected)

    def test__mul__returns_the_left_and_right_scalar_product(self) -> None:
        expected = Dual(self.s * self.r1, self.s * self.d1)
        self.assertEqual(self.s * self.dq1, expected)
        self.assertEqual(self.dq1 * self.s, expected)

    def test__mul__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.dq1.__mul__("string") == NotImplemented)

    def test__round__returns_the_components_rounded(self) -> None:
        self.assertEqual(round(Dual(1.1, -2.4)), Dual(1, -2))
        self.assertEqual(round(Dual(1.12, -2.44), 1), Dual(1.1, -2.4))

    def test__str__contains_the_components_of_the_vector(self) -> None:
        self.assertTrue(str(self.r1) in str(self.dq1))
        self.assertTrue(str(self.d1) in str(self.dq1))

    def test__sub__subtracts_two_duals(self) -> None:
        self.assertEqual(self.dq1 - self.dq2, Dual(self.r1 - self.r2, self.d1 - self.d2))

    def test__sub__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.dq1.__sub__(2) == NotImplemented)

    def test__truediv__returns_the_scalar_quotient(self) -> None:
        self.assertEqual(self.dq1 / self.s, Dual(self.r1 / self.s, self.d1 / self.s))

    def test__truediv__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.dq1.__truediv__("string") == NotImplemented)

    def test_conjugate_conjugates_the_dual(self):
        expected = Dual(quaternion.conjugate(self.r1), -quaternion.conjugate(self.d1))
        self.dq1.conjugate()
        self.assertEqual(self.dq1, expected)

        d = Dual(1, 2)
        d.conjugate()
        self.assertEqual(d, Dual(1, -2))

    def test_conjugate_raises_for_incompatible_types(self) -> None:
        d = Dual(Vector3(), Vector3())
        with self.assertRaises(NotImplementedError):
            d.conjugate()

    def test_dual_conjugate_returns_the_conjugate(self) -> None:
        expected = dual.conjugate(Dual(self.r1, self.d1))
        self.dq1.conjugate()
        self.assertEqual(expected, self.dq1)

    def test_dual_conjugate_raises_for_incompatible_types(self) -> None:
        with self.assertRaises(NotImplementedError):
            dual.conjugate(Dual(Vector3(), Vector3()))
