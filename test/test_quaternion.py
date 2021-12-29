import math, unittest

from spatial import Quaternion, Vector3, quaternion
from spatial.euler import Axes, Order


class TestQuaternion(unittest.TestCase):
    def setUp(self) -> None:
        self.I = Quaternion()
        self.q = Quaternion(1, 2, 3, 4)
        self.r = Quaternion(4, -3, 2, -1)
        self.axis = Vector3(1, 2, 3)

    def test__init__defaults_to_identity_quaternion(self) -> None:
        self.assertEqual(self.I, Quaternion(1, 0, 0, 0))

    def test__init__accepts_component_values(self) -> None:
        self.assertEqual(Quaternion(1, *self.axis), Quaternion(1, 1, 2, 3))

    def test_from_axis_angle_constructs_a_quaternion_from_axis_angle(self) -> None:
        angle = math.radians(30)
        c = math.cos(angle / 2)
        s = math.sin(angle / 2)
        expected = Quaternion(c, s * self.axis.x, s * self.axis.y, s * self.axis.z)
        self.assertEqual(Quaternion.from_axis_angle(self.axis, angle), expected)

    def test_from_euler_constructs_a_quaternion_from_euler_angles(self) -> None:
        angle = math.radians(45)
        y = Quaternion.from_axis_angle(Vector3(0, 1, 0), angle)
        z = Quaternion.from_axis_angle(Vector3(0, 0, 1), angle)

        actual = Quaternion.from_euler([angle, angle, 0], Axes.ZYZ, Order.EXTRINSIC)
        self.assertAlmostEqual(actual, y * z)

        actual = Quaternion.from_euler([angle, angle, 0], Axes.ZYZ, Order.INTRINSIC)
        self.assertAlmostEqual(actual, z * y)

    def test__abs__returns_the_absolute_values(self) -> None:
        self.assertEqual(abs(self.r), Quaternion(4, 3, 2, 1))

    def test__add__adds_two_quaternions(self) -> None:
        expected = Quaternion(5, -1, 5, 3)
        self.assertEqual(self.q + self.r, expected)
        self.assertEqual(self.r + self.q, expected)

    def test__add__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.q.__add__(2) == NotImplemented)

    def test__eq__returns_true_for_the_zero_quaternion_and_zero(self) -> None:
        self.assertTrue(Quaternion(0, 0, 0, 0) == 0)

    def test__eq__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.q.__eq__(2) == NotImplemented)
        self.assertTrue(self.q.__eq__("string") == NotImplemented)

    def test__getitem__returns_the_component_at_the_provided_index(self) -> None:
        expecteds = [self.q.r, self.q.x, self.q.y, self.q.z]
        for index, expected in zip(range(4), expecteds):
            self.assertEqual(self.q[index], expected)

    def test__mul__returns_the_quaternion_product(self) -> None:
        self.assertEqual(self.I * self.q, self.q)
        self.assertEqual(self.q * self.I, self.q)

        expected = Quaternion(8, -6, 4, 28)
        self.assertEqual(self.q * self.r, expected)

        expected = Quaternion(8, 16, 24, 2)
        self.assertEqual(self.r * self.q, expected)

    def test__mul__returns_the_left_and_right_scalar_product(self) -> None:
        s = 2.5
        expected = Quaternion(s * self.q.r, s * self.q.x, s * self.q.y, s * self.q.z)
        self.assertAlmostEqual(self.q * s, expected)
        self.assertAlmostEqual(s * self.q, expected)

    def test__mul__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.q.__mul__("string") == NotImplemented)

    def test__neg__returns_the_negation(self) -> None:
        self.assertEqual(-self.r, Quaternion(-4, 3, -2, 1))

    def test__round__returns_the_components_rounded(self) -> None:
        self.assertEqual(round(Quaternion(1.1, -2.4, 2.9, -4.4)), Quaternion(1, -2, 3, -4))
        self.assertEqual(
            round(Quaternion(1.12, -2.44, 2.92, -4.43), 1), Quaternion(1.1, -2.4, 2.9, -4.4)
        )

    def test__setitem__sets_the_component_at_the_provided_index(self) -> None:
        self.q[0] = self.r.r
        self.q[1] = self.r.x
        self.q[2] = self.r.y
        self.q[3] = self.r.z

        self.assertEqual(self.q, self.r)

    def test__str__contains_the_components_of_the_vector(self) -> None:
        self.assertTrue(str(self.q.r) in str(self.q))
        self.assertTrue(str(self.q.x) in str(self.q))
        self.assertTrue(str(self.q.y) in str(self.q))
        self.assertTrue(str(self.q.z) in str(self.q))

    def test__sub__subracts_two_quaternions(self) -> None:
        self.assertEqual(self.q - self.r, Quaternion(-3, 5, 1, 5))

    def test__sub__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.q.__sub__(2) == NotImplemented)

    def test__truediv__returns_the_scalar_quotient(self) -> None:
        s = 2
        expected = Quaternion(self.q.r / s, self.q.x / s, self.q.y / s, self.q.z / s)
        self.assertAlmostEqual(self.q / s, expected)

    def test__truediv__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.q.__truediv__("string") == NotImplemented)

    def test_conjugate_conjugates_the_quaternion(self) -> None:
        expected = Quaternion(self.q.r, -self.q.x, -self.q.y, -self.q.z)
        self.q.conjugate()
        self.assertEqual(self.q, expected)

    def test_norm_returns_the_norm_of_the_quaternion(self) -> None:
        self.assertAlmostEqual(self.q.norm(), math.sqrt(30))

    def test_normalize_normalizes_the_quaternion_to_unit_length(self) -> None:
        norm = self.q.norm()
        expected = self.q / norm
        self.q.normalize()

        self.assertAlmostEqual(self.q, expected)
        self.assertAlmostEqual(self.q.norm(), 1)

    def test_quaternion_conjugate_returns_the_conjugate(self) -> None:
        result = quaternion.conjugate(self.q)

        self.q.conjugate()
        expected = self.q

        self.assertEqual(result, expected)
