import math
import unittest
from operator import itemgetter

from spatial import Quaternion, Vector3
from spatial.euler import Axes, Order, angles


class TestEuler(unittest.TestCase):
    def setUp(self) -> None:
        # Frame is constructed by rotating around Z 45 degrees, rotation around new Y 135 degrees
        self.q = Quaternion(0.353553, -0.353553, 0.853553, 0.146447)

        q1 = Quaternion.from_axis_angle(axis=Vector3.Z(), angle=math.radians(90))
        q2 = Quaternion.from_axis_angle(axis=Vector3.X(), angle=math.radians(90))

        self.extrinsic = q1 * q2

    def checkSolutions(self, results, solutions):
        results.sort(key=itemgetter(1), reverse=True)
        solutions.sort(key=itemgetter(1), reverse=True)

        for index, (result, expecteds) in enumerate(zip(results, solutions)):
            for angleIndex, (angle, expected) in enumerate(zip(result, expecteds)):
                with self.subTest(f"Solution #{index+1}, Angle #{angleIndex+1}"):
                    self.assertAlmostEqual(angle, expected, places=5)

    def test_angles_returns_zyx_intrinsic_euler_angles(self) -> None:
        results = angles(self.q, Axes.ZYX, Order.INTRINSIC)
        solutions = [
            [math.radians(-135), math.radians(45), math.radians(180)],
            [math.radians(45), math.radians(225), math.radians(0)],
        ]

        self.checkSolutions(results, solutions)

    def test_angles_returns_zyz_intrinsic_euler_angles(self) -> None:
        results = angles(self.q, Axes.ZYZ, Order.INTRINSIC)
        solutions = [
            [math.radians(45), math.radians(135), math.radians(0)],
            [math.radians(-135), math.radians(-135), math.radians(-180)],
        ]

        self.checkSolutions(results, solutions)

    def test_angles_returns_zyz_extrinsic_euler_angles(self) -> None:
        results = angles(self.extrinsic, Axes.ZYZ, Order.EXTRINSIC)
        solutions = [
            [math.radians(90), math.radians(90), math.radians(0)],
            [math.radians(-90), math.radians(-90), math.radians(-180)],
        ]

        self.checkSolutions(results, solutions)

    def test_angles_returns_zeros_for_singular_configurations(self) -> None:
        singular_q = Quaternion.from_axis_angle(axis=Vector3.Z(), angle=math.radians(90))
        results = angles(singular_q, Axes.ZYZ, Order.INTRINSIC)

        self.assertTrue(len(results), 0)
        self.assertEqual(results[0], [0, 0, 0])

    def test_angles_raises_for_unknown_types(self) -> None:
        with self.assertRaises(TypeError):
            results = angles(self.q, "ZYZ", Order.INTRINSIC)

        with self.assertRaises(TypeError):
            results = angles(self.q, Axes.ZYZ, "Intrinsic")

    def test_angles_raises_for_conversion_that_do_not_have_implementations(self) -> None:
        with self.assertRaises(NotImplementedError):
            results = angles(self.q, Axes.XYZ, Order.INTRINSIC)
