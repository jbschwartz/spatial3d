import math
import unittest

from spatial import Dual, Matrix4, Quaternion, Transform, Vector3


class TestMatrix4(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = Transform.from_axis_angle_translation(
            axis=Vector3(0, 1, 0), angle=math.radians(90), translation=Vector3(1, 2, 3)
        )
        self.expected = [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 2, 3, 1]

    def test__init__defaults_to_identity_matrix(self) -> None:
        identity = Matrix4()
        for index in [0, 5, 10, 15]:
            self.assertEqual(identity[index], 1)

    def test__init__accepts_16_elements(self) -> None:
        m = Matrix4(range(16))
        for index, value in enumerate(m):
            self.assertEqual(value, index)

    def test__init__raises_if_not_given_16_elements(self) -> None:
        with self.assertRaises(TypeError):
            Matrix4([0] * 15)

    def test_from_basis_constructs_a_matrix_from_three_basis_vectors_and_an_origin(self) -> None:
        vectors = [
            Vector3(1, 2, 3).normalize(),
            Vector3(-4, -7, 6).normalize(),
            Vector3(),
            Vector3(-1, -2, -3),
        ]
        vectors[2] = (vectors[0] % vectors[1]).normalize()

        matrix = Matrix4.from_basis(*vectors)

        self.assertEqual(Vector3(*matrix.elements[0:3]), vectors[0])
        self.assertEqual(Vector3(*matrix.elements[4:7]), vectors[1])
        self.assertEqual(Vector3(*matrix.elements[8:11]), vectors[2])
        self.assertEqual(Vector3(*matrix.elements[12:15]), vectors[3])

    def test_from_basis_raises_for_non_orthonormal_vectors(self) -> None:
        vectors = [Vector3(1, 2, 3), Vector3(4, 6, 7), Vector3(8, 9, 0), Vector3(-1, -2, -3)]
        with self.assertRaises(AssertionError):
            _ = Matrix4.from_basis(*vectors)

        vectors = [
            2 * Vector3.X(),
            2 * Vector3.Y(),
            2 * Vector3.Z(),
            Vector3(-1, -2, -3),
        ]
        with self.assertRaises(AssertionError):
            _ = Matrix4.from_basis(*vectors)

    def test_from_transform_constructs_a_matrix_from_a_dual_quaternion(self) -> None:
        for value, expected in zip(Matrix4.from_transform(self.transform).elements, self.expected):
            self.assertAlmostEqual(value, expected)

    def test_from_dual_constructs_a_matrix_from_a_dual_quaternion(self) -> None:
        for value, expected in zip(Matrix4.from_dual(self.transform.dual).elements, self.expected):
            self.assertAlmostEqual(value, expected)

    def test__str__contains_the_components_of_the_matrix(self) -> None:
        m = Matrix4([e / 10 for e in range(1, 161, 10)])
        for value in m:
            with self.subTest(case=value):
                self.assertTrue(str(value) in str(m))
