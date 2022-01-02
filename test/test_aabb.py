import math
import unittest

from spatial import AABB, CoordinateAxes, Ray, Vector3


class TestAABB(unittest.TestCase):
    def setUp(self) -> None:
        self.v1 = Vector3(1, 2, 3)
        self.v2 = Vector3(-1, -2, -3)
        self.v3 = Vector3(0, 1, 2)
        self.v4 = Vector3(4, 4, 4)

        self.aabb = AABB([self.v1, self.v2, self.v3])

    def test__str__returns_a_string_with_the_min_and_max_points(self) -> None:
        self.assertTrue(str(self.aabb.min) in str(self.aabb))
        self.assertTrue(str(self.aabb.max) in str(self.aabb))

    def test_center_returns_the_center_of_the_bounding_box(self) -> None:
        expected = (self.v1 - self.v2) / 2 + self.v2

        self.assertAlmostEqual(self.aabb.center, expected)

    def test_center_returns_the_origin_when_empty(self) -> None:
        empty = AABB()

        self.assertAlmostEqual(empty.center, Vector3(0, 0, 0))

    def test_corners_returns_all_eight_corners_of_the_bounding_box(self) -> None:
        aabb = AABB([Vector3(-1, -1, -1), Vector3(1, 1, 1)])

        corners = [
            Vector3(-1, -1, -1),
            Vector3(-1, +1, -1),
            Vector3(+1, +1, -1),
            Vector3(+1, -1, -1),
            Vector3(-1, -1, +1),
            Vector3(-1, +1, +1),
            Vector3(+1, +1, +1),
            Vector3(+1, -1, +1),
        ]

        actual = aabb.corners

        self.assertEqual(len(actual), len(corners))

        for corner in corners:
            with self.subTest(case=corner):
                self.assertTrue(corner in actual)

    def test_size_returns_the_component_wise_size_of_the_bounding_box(self) -> None:
        expected = self.v1 - self.v2

        self.assertAlmostEqual(self.aabb.size, expected)

    def test_size_returns_infinite_for_empty_bounding_box(self) -> None:
        aabb = AABB()
        self.assertEqual(aabb.size, Vector3(math.inf, math.inf, math.inf))

    def test_axis_extents_returns_the_minimum_and_maximum_of_an_axis(self) -> None:
        for axis in CoordinateAxes:
            with self.subTest(case=axis):
                self.assertEqual(self.aabb.axis_extents(axis), (self.v2[axis], self.v1[axis]))

    def test_contains_returns_true_if_the_point_is_in_the_bounding_box(self) -> None:
        self.assertTrue(self.aabb.contains(self.v3))
        self.assertFalse(self.aabb.contains(self.v4))

    def test_expand_expands_to_fit_passed_points_or_bounding_boxes(self) -> None:
        other_aabb = AABB([self.v3, self.v4])

        self.aabb.expand(other_aabb)

        self.assertAlmostEqual(self.v2, self.aabb.min)
        self.assertAlmostEqual(self.v4, self.aabb.max)

    def test_expand_raises_for_incompatible_type(self) -> None:
        with self.assertRaises(TypeError):
            self.aabb.expand("String")

    def test_expand_invalidates_cached_properties(self) -> None:
        old_corners = self.aabb.corners
        old_center = self.aabb.center

        self.aabb.expand(self.v4)

        self.assertNotEqual(self.aabb.corners, old_corners)
        self.assertNotEqual(self.aabb.center, old_center)

    def test_intersect(self) -> None:
        origin = Vector3(2, 3, 1)

        cases = [
            (Ray(origin, Vector3(1, 0, 0)), False),
            (Ray(origin, Vector3(-1, 0, 0)), False),
            (Ray(origin, Vector3(0, 1, 0)), False),
            (Ray(origin, Vector3(0, -1, 0)), False),
            (Ray(origin, Vector3(0, 0, 1)), False),
            (Ray(origin, Vector3(0, 0, -1)), False),
            (Ray(origin, Vector3(-2, 1, 1)), False),
            (Ray(origin, Vector3(-1, 0, 1)), False),
            (Ray(origin, Vector3(-1, -1, 1)), True),
            (Ray(origin, Vector3(-1, -1, 0)), True),
            (Ray(origin, Vector3(-2, -1, 0)), True),
        ]

        for index, (ray, expected) in enumerate(cases):
            with self.subTest(msg=f"Test #{index}, Ray {ray}"):
                self.assertEqual(self.aabb.intersect(ray), expected)

    def test_sphere_radius_returns_the_radius_of_a_bounding_sphere(self) -> None:
        expected = self.aabb.min.length()
        self.assertAlmostEqual(self.aabb.sphere_radius(), expected)

    def test_split_returns_a_left_and_right_bounding_box(self) -> None:
        value = -0.5

        for axis in CoordinateAxes:
            with self.subTest(f"Split on axis {axis.name}"):
                left, right = self.aabb.split(axis, value)

                left_max = Vector3(*self.v1)
                left_max[axis] = value

                right_min = Vector3(*self.v2)
                right_min[axis] = value

                self.assertAlmostEqual(left.min, self.aabb.min, msg="Left split incorrect min")
                self.assertAlmostEqual(left.max, left_max, msg="Left split incorrect max")

                self.assertAlmostEqual(right.min, right_min, msg="Right split incorrect min")
                self.assertAlmostEqual(right.max, self.aabb.max, msg="Right split incorrect max")

    def test_split_raises_if_the_value_is_outside_the_bounding_box(self) -> None:
        value = 2 * self.aabb.max[CoordinateAxes.X]

        with self.assertRaises(ValueError):
            self.aabb.split(CoordinateAxes.X, value)
