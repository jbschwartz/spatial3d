import math
import unittest

from spatial3d import Facet, Ray, Transform, Vector3, vector3


class TestRay(unittest.TestCase):
    def setUp(self):
        self.direction = Vector3(-1, 4, 5)
        self.ray = Ray(Vector3(1, 2, 3), self.direction)

    def test__init__normalized_the_direction(self):
        self.assertAlmostEqual(self.ray.direction, vector3.normalize(self.direction))

    def test__init__raises_if_direction_is_zero_vector(self):
        with self.assertRaises(ValueError):
            Ray(Vector3(), Vector3())

    def test__str__contains_the_components_of_the_ray(self) -> None:
        self.assertTrue(str(self.ray.origin) in str(self.ray))

        normalized = vector3.normalize(self.ray.direction)
        self.assertTrue(str(normalized) in str(self.ray))

    def test_closest_intersection_returns_the_closest_intersection(self) -> None:
        facets = [
            Vector3(),  # Ignored by the algorithm.
            Facet([Vector3(), Vector3.X(), Vector3.Y()]),
            Facet([Vector3(0, 0, 1), Vector3(1, 0, 1), Vector3(0, 1, 1)]),
        ]

        r = Ray(Vector3(0, 0, 2), Vector3.Z())
        actual = r.closest_intersection(facets)

        self.assertIsNone(actual.t)
        self.assertIsNone(actual.obj)

        r = Ray(Vector3(0, 0, 2), -Vector3.Z())
        actual = r.closest_intersection(facets)

        self.assertEqual(actual.t, 1)
        self.assertEqual(actual.obj, facets[2])

    def test_evaluate_returns_the_location_along_the_ray(self) -> None:
        self.assertEqual(self.ray.evaluate(0), self.ray.origin)
        self.assertEqual(
            self.ray.evaluate(2), self.ray.origin + 2 * vector3.normalize(self.ray.direction)
        )

    def test_transform_transforms_the_rays_origin_and_direction(self) -> None:
        r = Ray(Vector3(1, 1, 0), Vector3(1, 1, 0))
        t = Transform.from_axis_angle_translation(
            axis=Vector3.Z(), angle=math.radians(-45), translation=Vector3(-math.sqrt(2), 0, 0)
        )

        expected = Ray(Vector3(), Vector3.X())
        actual = r.transform(t)

        self.assertAlmostEqual(actual.origin, expected.origin)
        self.assertAlmostEqual(actual.direction, expected.direction)
