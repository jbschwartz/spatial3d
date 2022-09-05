import math
import unittest

from spatial3d.vector3 import (
    Vector3,
    almost_equal,
    angle_between,
    cross,
    is_orthonormal_basis,
    normalize,
)


class TestVector3(unittest.TestCase):
    def setUp(self) -> None:
        self.v1 = Vector3(-1, 2, -3)
        self.v2 = Vector3(2, -5, 3)
        self.s = 2.5

    def test__init__defaults_to_zero_vector(self) -> None:
        v = Vector3()
        self.assertEqual(v, Vector3(0, 0, 0))

    def test_X_Y_Z_construct_standard_basis_vectors(self) -> None:
        self.assertEqual(Vector3.X(), Vector3(1, 0, 0))
        self.assertEqual(Vector3.Y(), Vector3(0, 1, 0))
        self.assertEqual(Vector3.Z(), Vector3(0, 0, 1))

    def test__abs__returns_the_absolute_values(self) -> None:
        self.assertEqual(abs(self.v1), Vector3(1, 2, 3))

    def test__add__adds_two_vectors(self) -> None:
        expected = Vector3(1, -3, 0)
        self.assertEqual(self.v1 + self.v2, expected)
        self.assertEqual(self.v2 + self.v1, expected)

    def test__add__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.v1.__add__(2) == NotImplemented)

    def test__eq__returns_true_for_the_zero_vector_and_zero(self) -> None:
        self.assertTrue(Vector3(0, 0, 0) == 0)

    def test__eq__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.v1.__eq__(2) == NotImplemented)
        self.assertTrue(self.v1.__eq__("string") == NotImplemented)

    def test__getitem__returns_the_component_at_the_provided_index(self) -> None:
        expecteds = [self.v1.x, self.v1.y, self.v1.z]
        for index, expected in zip(range(len(self.v1)), expecteds):
            self.assertEqual(self.v1[index], expected)

    def test__len__returns_the_number_of_dimensions(self) -> None:
        self.assertEqual(len(self.v1), 3)

    def test__mod__returns_the_cross_product(self) -> None:
        expected = Vector3(-9, -3, 1)
        self.assertAlmostEqual(self.v1 % self.v2, expected)
        self.assertAlmostEqual(self.v2 % self.v1, -expected)

    def test__mul__returns_the_dot_product(self) -> None:
        expected = -21
        self.assertEqual(self.v1 * self.v2, expected)
        self.assertEqual(self.v2 * self.v1, expected)

    def test__mul__returns_the_left_and_right_scalar_product(self) -> None:
        expected = Vector3(self.s * self.v1.x, self.s * self.v1.y, self.s * self.v1.z)
        self.assertAlmostEqual(self.v1 * self.s, expected)
        self.assertAlmostEqual(self.s * self.v1, expected)

    def test__mul__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.v1.__mul__("string") == NotImplemented)

    def test__neg__returns_the_negation(self) -> None:
        self.assertEqual(-self.v1, Vector3(1, -2, 3))

    def test__repr__contains_the_components_of_the_vector(self) -> None:
        self.assertTrue("Vector3" in repr(self.v1))
        self.assertTrue(str(self.v1.x) in repr(self.v1))
        self.assertTrue(str(self.v1.y) in repr(self.v1))
        self.assertTrue(str(self.v1.z) in repr(self.v1))

    def test__round__returns_the_components_rounded(self) -> None:
        self.assertEqual(round(Vector3(1.1, -2.4, 2.9)), Vector3(1, -2, 3))
        self.assertEqual(round(Vector3(1.12, -2.44, 2.92), 1), Vector3(1.1, -2.4, 2.9))

    def test__setitem__sets_the_component_at_the_provided_index(self) -> None:
        self.v1[0] = self.v2.x
        self.v1[1] = self.v2.y
        self.v1[2] = self.v2.z

        self.assertEqual(self.v1, self.v2)

    def test__str__contains_the_components_of_the_vector(self) -> None:
        self.assertTrue(str(self.v1.x) in str(self.v1))
        self.assertTrue(str(self.v1.y) in str(self.v1))
        self.assertTrue(str(self.v1.z) in str(self.v1))

    def test__sub__subtracts_two_vectors(self) -> None:
        self.assertEqual(self.v1 - self.v2, Vector3(-3, 7, -6))

    def test__sub__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.v1.__sub__(2) == NotImplemented)

    def test__truediv__returns_the_scalar_quotient(self) -> None:
        expected = Vector3(self.v1.x / self.s, self.v1.y / self.s, self.v1.z / self.s)
        self.assertAlmostEqual(self.v1 / self.s, expected)

    def test__truediv__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.v1.__truediv__("string") == NotImplemented)

    def test_is_perpendicular_to_returns_true_for_perpendicular_vectors(self) -> None:
        self.assertFalse(self.v1.is_perpendicular_to(self.v2))
        self.assertTrue(Vector3.Z().is_perpendicular_to(Vector3.X()))

    def test_is_unit_returns_true_for_vectors_of_length_one(self) -> None:
        unit_vector = normalize(self.v1)
        self.assertFalse(self.v1.is_unit())
        self.assertTrue(unit_vector.is_unit())

    def test_length_returns_the_euclidean_length_of_the_vector(self) -> None:
        self.assertAlmostEqual(self.v1.length(), math.sqrt(14))

    def test_length_sq_returns_the_squared_euclidean_length_of_the_vector(self) -> None:
        self.assertAlmostEqual(self.v1.length_sq(), 14)

    def test_normalize_normalizes_the_vector_to_unit_length(self) -> None:
        length = self.v1.length()
        expected = self.v1 / length
        self.v1.normalize()

        self.assertAlmostEqual(self.v1, expected)
        self.assertAlmostEqual(self.v1.length(), 1)

    def test_vector3_almost_equal_returns_true_for_vectors_that_are_almost_equal(self) -> None:
        other = self.v1 + Vector3(0.01, 0.01, 0.01)
        self.assertFalse(almost_equal(self.v1, other))
        self.assertTrue(almost_equal(self.v1, other, 0.001))

    def test_vector3_angle_between_returns_the_angle_between_two_vectors_in_radians(self) -> None:
        self.assertAlmostEqual(angle_between(self.v1, 5 * self.v1), 0)

        x, y = Vector3.X(), Vector3.Y()
        expected = math.radians(90)
        self.assertAlmostEqual(angle_between(x, y), expected)
        self.assertAlmostEqual(angle_between(y, x), expected)

        p = Vector3(45, 45, 0)
        expected = math.radians(45)
        self.assertAlmostEqual(angle_between(x, p), expected)
        self.assertAlmostEqual(angle_between(p, x), expected)

    def test_vector3_cross_returns_the_cross_product_of_two_vectors(self) -> None:
        expected = Vector3(-9, -3, 1)
        self.assertAlmostEqual(cross(self.v1, self.v2), expected)
        self.assertAlmostEqual(cross(self.v2, self.v1), -expected)

    def test_vector3_is_orthonormal_basis_returns_true_for_an_orthonormal_basis(self) -> None:
        basis = [Vector3.X(), Vector3.Y(), Vector3.Z()]
        self.assertTrue(is_orthonormal_basis(*basis))

        basis = [2 * Vector3.X(), Vector3.Y(), Vector3.Z()]
        self.assertFalse(is_orthonormal_basis(*basis))

        basis = [Vector3(1, 1, 0).normalize(), Vector3.Y(), Vector3.Z()]
        self.assertFalse(is_orthonormal_basis(*basis))

    def test_vector3_normalize_returns_the_normalized_vector(self) -> None:
        result = normalize(self.v1)

        self.v1.normalize()
        expected = self.v1

        self.assertEqual(result, expected)
