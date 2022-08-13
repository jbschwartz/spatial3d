import math
import unittest
from operator import itemgetter
from typing import List

from spatial import Quaternion, Vector3
from spatial.euler import Axes, Order, angles


def fix_angle_range(angle: float) -> float:
    """Return the provided angle in the range [-math.pi, math.pi]."""

    tau = 2 * math.pi
    return ((angle + math.pi) % tau) - math.pi


def build_solutions(angles: List[float], is_tait_bryan: bool):
    second_solution = [
        fix_angle_range(angles[0] - math.pi),
        -angles[1],
        fix_angle_range(angles[2] - math.pi),
    ]

    if is_tait_bryan:
        second_solution[1] = math.pi - angles[1]

    return [angles, second_solution]


class TestEuler(unittest.TestCase):
    def setUp(self) -> None:
        self.target_angles = [math.radians(135), math.radians(-45), math.radians(-30)]

        q1 = Quaternion.from_axis_angle(axis=Vector3.Z(), angle=math.radians(90))
        q2 = Quaternion.from_axis_angle(axis=Vector3.X(), angle=math.radians(90))

        self.extrinsic = q1 * q2

    def checkSolutions(self, results, solutions, axes):
        results.sort(key=itemgetter(1), reverse=True)
        solutions.sort(key=itemgetter(1), reverse=True)

        for index, (result, expecteds) in enumerate(zip(results, solutions), 1):
            for angleIndex, (angle, expected) in enumerate(zip(result, expecteds), 1):
                with self.subTest(f"{axes.name}: Solution #{index}, Angle #{angleIndex}"):
                    self.assertAlmostEqual(angle, expected, places=5)

    def test_angles_returns_intrinsic_euler_angles(self) -> None:
        for euler_axes in Axes:
            with self.subTest(f"Axes: {euler_axes}"):
                q = Quaternion.from_euler(self.target_angles, euler_axes, Order.INTRINSIC)
                results = angles(q, euler_axes, Order.INTRINSIC)
                solutions = build_solutions(self.target_angles, euler_axes.is_tait_bryan)

                self.checkSolutions(results, solutions, euler_axes)

    def test_angles_returns_zyz_intrinsic_euler_angles(self) -> None:
        q = Quaternion.from_euler(self.target_angles, Axes.ZYZ, Order.INTRINSIC)
        intrinsic = angles(q, Axes.ZYZ, Order.INTRINSIC)
        intrinsic[0].reverse()
        intrinsic[1].reverse()

        results = angles(q, Axes.ZYZ, Order.EXTRINSIC)
        solutions = [intrinsic[0], intrinsic[1]]

        self.checkSolutions(results, solutions, Axes.ZYZ)

    def test_angles_returns_zyz_extrinsic_euler_angles(self) -> None:
        results = angles(self.extrinsic, Axes.ZYZ, Order.EXTRINSIC)
        solutions = [
            [math.radians(90), math.radians(90), math.radians(0)],
            [math.radians(-90), math.radians(-90), math.radians(-180)],
        ]

        self.checkSolutions(results, solutions, Axes.ZYZ)

    def test_angles_returns_zeros_for_singular_configurations(self) -> None:
        singular_q = Quaternion.from_axis_angle(axis=Vector3.Z(), angle=math.radians(90))
        results = angles(singular_q, Axes.ZYZ, Order.INTRINSIC)

        self.assertTrue(len(results), 0)
        self.assertEqual(results[0], [0, 0, 0])

    def test_angles_raises_for_unknown_types(self) -> None:
        with self.assertRaises(TypeError):
            _ = angles(Quaternion(), "ZYZ", Order.INTRINSIC)

        with self.assertRaises(TypeError):
            _ = angles(Quaternion(), Axes.ZYZ, "Intrinsic")
