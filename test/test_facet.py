import math
import unittest
from typing import Optional

from spatial import AABB, Edge, Facet, Intersection, Ray, Vector3
from spatial.exceptions import DegenerateTriangleError
from spatial.transform import Transform


class TestFacet(unittest.TestCase):
    def setUp(self):
        self.vertices = [Vector3.X(), Vector3.Y(), -Vector3.X()]
        self.facet = Facet(self.vertices)

        self.origins = [Vector3(0, 0, 0), Vector3(0, 0, 1), Vector3(5, 5, 5)]

    def test__init__defaults_to_computed_normal(self) -> None:
        self.assertEqual(self.facet.normal, self.facet.computed_normal)

    def test_aabb_returns_the_bounding_box_of_the_facet(self) -> None:
        expected = AABB(self.vertices)
        self.assertEqual(self.facet.aabb.min, expected.min)
        self.assertEqual(self.facet.aabb.max, expected.max)

    def test_computed_normal_returns_the_normal_of_the_facet(self) -> None:
        self.assertEqual(self.facet.computed_normal, Vector3(0, 0, 1))

    def test_computed_normal_raises_for_a_degenerate_triangle(self) -> None:
        vertices = [Vector3(), Vector3(), Vector3()]
        bad_facet = Facet(vertices, Vector3())
        with self.assertRaises(DegenerateTriangleError):
            bad_facet.computed_normal

    def test_edges_return_all_edges_in_the_facet(self) -> None:
        self.assertEqual(len(self.facet.edges), 3)
        self.assertTrue(all(isinstance(edge, Edge) for edge in self.facet.edges))

    def test_is_triangle_returns_true_for_three_vertices(self) -> None:
        self.assertTrue(self.facet.is_triangle)
        self.facet.vertices.append(Vector3())
        self.assertFalse(self.facet.is_triangle)

    def test_intersect_returns_the_intersection_if_one_exists(self):
        def compute_t(ray: Ray, intersection: Vector3) -> Optional[Vector3]:
            """Backwards compute parameter t given a ray and intersection."""
            if intersection is None:
                return None

            return (ray.origin - intersection).length()

        test_cases = [
            (Ray(self.origins[0], Vector3.Z()), None),
            (Ray(self.origins[0], -Vector3.Z()), self.origins[0]),
            (Ray(self.origins[1], Vector3.Y()), None),
            (Ray(self.origins[1], -Vector3.Z()), self.origins[0]),
            (Ray(self.origins[1], Vector3.Z()), None),
            (Ray(self.origins[1], Vector3(0.5, 0.25, -1)), Vector3(0.5, 0.25, 0)),
            (Ray(self.origins[2], -Vector3.Z()), None),
            (Ray(self.origins[2], Vector3(-1, -1, -1)), Vector3()),
        ]

        for ray, intersection in test_cases:
            with self.subTest(f"Check {ray}"):
                result = self.facet.intersect(ray)

                expected = Intersection(compute_t(ray, intersection), None)

                if expected.hit:
                    self.assertAlmostEqual(result.t, expected.t)
                else:
                    self.assertIsNone(result.t)
                    self.assertIsNone(result.obj)

    def test_scale_returns_a_scaled_facet(self) -> None:
        scaled_facet = self.facet.scale(2)
        for actual, vertex in zip(scaled_facet.vertices, self.vertices):
            self.assertEqual(actual, 2 * vertex)

    def test_transform_returns_a_transformed_facet(self) -> None:
        t = Transform.from_axis_angle_translation(axis=Vector3.Z(), angle=math.radians(90))
        transformed_facet = self.facet.transform(t)

        for actual, vertex in zip(transformed_facet.vertices, self.vertices):
            self.assertEqual(actual, t(vertex))
