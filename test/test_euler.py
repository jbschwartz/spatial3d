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
        q = Quaternion.from_euler(self.target_angles, Axes.ZXY, Order.INTRINSIC)
        results = angles(q, Axes.YXZ, Order.EXTRINSIC)

        self.target_angles.reverse()
        solutions = build_solutions(self.target_angles, True)

        self.checkSolutions(results, solutions, Axes.ZYZ)

    def test_angles_returns_one_solution_for_proper_singular_configurations(self) -> None:
        values = [math.radians(90), 0, math.radians(-30)]
        expected_values = [math.radians(60), 0, 0]
        for axes in (axes for axes in Axes if not axes.is_tait_bryan):
            with self.subTest(case=axes):
                q = Quaternion.from_euler(values, axes, Order.INTRINSIC)
                results = angles(q, axes, Order.INTRINSIC)

                self.assertEqual(len(results), 1)
                for actual, expected in zip(results[0], expected_values):
                    self.assertAlmostEqual(actual, expected)

    def test_angles_returns_one_solution_for_tait_bryan_singular_configurations(self) -> None:
        values = [0, math.radians(90), math.radians(-30)]
        for axes in (axes for axes in Axes if axes.is_tait_bryan):
            with self.subTest(case=axes):
                q = Quaternion.from_euler(values, axes, Order.INTRINSIC)
                results = angles(q, axes, Order.INTRINSIC)

                self.assertEqual(len(results), 1)
                for actual, expected in zip(results[0], values):
                    self.assertAlmostEqual(actual, expected)

    def test_angles_raises_for_unknown_types(self) -> None:
        with self.assertRaises(TypeError):
            _ = angles(Quaternion(), "ZYZ", Order.INTRINSIC)

        with self.assertRaises(TypeError):
            _ = angles(Quaternion(), Axes.ZYZ, "Intrinsic")
