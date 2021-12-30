import math
import unittest

from spatial import Quaternion, Transform, Vector3


class TestTransform(unittest.TestCase):
    def setUp(self) -> None:
        self.pureTranslate = Transform.from_axis_angle_translation(translation=Vector3(4, 2, 6))
        self.pureRotate = Transform.from_axis_angle_translation(
            axis=Vector3.X(), angle=math.radians(180)
        )
        self.both = Transform.from_axis_angle_translation(
            axis=Vector3.X(), angle=math.radians(180), translation=Vector3(4, 2, 6)
        )
        self.point = Vector3(3, 4, 5)

    def test__init__defaults_to_identity_transformation(self) -> None:
        t = Transform()
        self.assertEqual(t.dual.r, Quaternion(1, 0, 0, 0))
        self.assertEqual(t.dual.d, Quaternion(0, 0, 0, 0))
        self.assertEqual(t(self.point, as_type="point"), self.point)

    def test_Identity_returns_the_identity_transformation(self) -> None:
        t = Transform.Identity()
        self.assertEqual(t.dual.r, Quaternion(1, 0, 0, 0))
        self.assertEqual(t.dual.d, Quaternion(0, 0, 0, 0))
        self.assertEqual(t(self.point, as_type="point"), self.point)

    def test_from_axis_angle_translation_constructs_a_transform_given_components(self) -> None:
        self.assertEqual(self.pureTranslate.transform(Vector3()), Vector3(4, 2, 6))
        self.assertAlmostEqual(self.pureRotate.transform(Vector3.Z()), -Vector3.Z())

    def test_from_orientation_translation_constructs_a_transform_given_components(self) -> None:
        q = Quaternion.from_axis_angle(axis=Vector3.X(), angle=math.radians(180))
        t = Transform.from_orientation_translation(q, translation=Vector3(4, 2, 6))

        self.assertAlmostEqual(t.transform(self.point), Vector3(7, -2, 1))

    def test__call__applies_the_transformation_to_the_passed_object(self) -> None:
        self.assertEqual(self.pureTranslate(self.point), Vector3(7, 6, 11))
        self.assertEqual(self.pureRotate(self.point), Vector3(3, -4, -5))

        self.assertAlmostEqual(self.both(self.point), Vector3(7, -2, 1))

    def test__call__applies_the_transformation_to_the_passed_objects(self) -> None:
        points = [self.point, Vector3(7, -2, 1)]
        results = self.both(points)

        self.assertAlmostEqual(results[0], Vector3(7, -2, 1))
        self.assertAlmostEqual(results[1], Vector3(11, 4, 5))

    def test__call__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.both("string") == NotImplemented)

    def test__mul__composes_two_transformations(self) -> None:
        # Rotate then translate
        combined = self.pureTranslate * self.pureRotate
        self.assertAlmostEqual(
            combined(self.point), self.pureTranslate(self.pureRotate(self.point))
        )

        # Translate then rotate
        combined = self.pureRotate * self.pureTranslate
        self.assertAlmostEqual(
            combined(self.point), self.pureRotate(self.pureTranslate(self.point))
        )

    def test__mul__returns_notimplemented_for_incompatible_types(self) -> None:
        self.assertTrue(self.both.__mul__("string") == NotImplemented)

    def test_rotation_returns_the_rotation_component_of_the_transform(self) -> None:
        self.assertEqual(self.pureRotate.rotation, self.pureRotate.dual.r)

    def test_translation_returns_the_translation_component_of_the_transform(self) -> None:
        self.assertEqual(self.pureTranslate.translation, Vector3(4, 2, 6))

    def test_inverse_returns_the_inverse_of_the_transform(self) -> None:
        inverse = self.both.inverse()
        self.assertAlmostEqual(inverse(self.both(self.point)), self.point)

    def test_transform_applies_the_transformation_to_the_passed_object(self) -> None:
        self.assertEqual(self.pureTranslate.transform(self.point), Vector3(7, 6, 11))
        self.assertEqual(self.pureRotate.transform(self.point), Vector3(3, -4, -5))

        self.assertAlmostEqual(self.both.transform(self.point), Vector3(7, -2, 1))

    def test_transform_raises_for_an_unknown_as_type(self) -> None:
        with self.assertRaises(KeyError):
            self.pureTranslate.transform(self.point, as_type="Unknown")
