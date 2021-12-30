import unittest

from spatial import Intersection


class TestIntersection(unittest.TestCase):
    def setUp(self) -> None:
        self.miss = Intersection.Miss()
        self.near = Intersection(0.5, "a")
        self.far = Intersection(1.0, "b")

    def test__new__raises_for_negative_t_values(self) -> None:
        with self.assertRaises(ValueError):
            Intersection(-1.0, None)

    def test__new__parameters_t_and_obj_are_optional(self) -> None:
        x = Intersection(None, None)

        self.assertIsNone(x.t)
        self.assertIsNone(x.obj)

    def test_miss_creates_an_intersection_which_is_not_a_hit(self) -> None:
        self.assertFalse(self.miss.hit)
        self.assertIsNone(self.miss.t)
        self.assertIsNone(self.miss.obj)

    def test_hit_returns_true_for_positive_t(self) -> None:
        cases = [Intersection(0, None), Intersection(1.0, None), Intersection(1.0, "Something")]

        for case in cases:
            with self.subTest(case=case):
                self.assertTrue(case.hit)

    def test_closer_than_returns_true_for_found_intersections(self) -> None:
        self.assertTrue(self.near.closer_than(self.far))
        self.assertFalse(self.far.closer_than(self.near))

    def test_closer_than_returns_true_compared_to_misses(self) -> None:
        self.assertTrue(self.near.closer_than(self.miss))

    def test_closer_than_returns_false_for_a_miss(self) -> None:
        for case in [self.miss, self.near, self.far]:
            with self.subTest(case=case):
                self.assertFalse(self.miss.closer_than(case))

    def test_closer_than_returns_false_for_identical_intersections(self) -> None:
        for case in [self.miss, self.near, self.far]:
            with self.subTest(case=case):
                self.assertFalse(case.closer_than(case))

    def test_intersection_attributes_t_and_obj_are_immutable(self) -> None:
        with self.assertRaises(AttributeError):
            self.near.t = None

        with self.assertRaises(AttributeError):
            self.near.obj = None
