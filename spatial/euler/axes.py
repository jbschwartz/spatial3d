# Letter variable names in this file match common mathematical notation.
# Local variables maintain more concise mathematical equations.
# pylint: disable=invalid-name,too-many-locals

import enum
import math
from typing import List

from spatial.vector3 import Vector3


class Axes(enum.Enum):
    """All the different types of Euler angles."""

    XYX = enum.auto()
    XZX = enum.auto()
    YXY = enum.auto()
    YZY = enum.auto()
    ZXZ = enum.auto()
    ZYZ = enum.auto()
    XYZ = enum.auto()
    YZX = enum.auto()
    ZXY = enum.auto()
    XZY = enum.auto()
    ZYX = enum.auto()
    YXZ = enum.auto()

    @classmethod
    def basis_vector(cls, axis: str) -> Vector3:
        """Return a basis vector corresponding to the provided axis letter."""
        v = Vector3()

        # For now, allow exceptions for unrecognized `axis` to go uncaught
        setattr(v, axis, 1)
        return v

    def convert(self, quaternion: "Quaternion") -> List[List[float]]:
        """Return the Euler angles from the provided quaternion."""
        if self.is_tait_bryan:
            return self._tait_bryan(quaternion)

        return self._proper(quaternion)

    @property
    def is_tait_bryan(self) -> bool:
        """Return True if these angles are Tait-Bryan angles."""
        return all(letter in self.name for letter in ["X", "Y", "Z"])

    def reverse(self) -> "Axes":
        """Return the reversed order Euler angles."""
        reversed_name = self.name[::-1]
        return Axes[reversed_name]

    @property
    def vectors(self) -> List[Vector3]:
        """Return the basis vectors corresponding to the Euler angles axes."""
        return [Axes.basis_vector(axis.lower()) for axis in self.name]

    def _proper(self, quaternion: "Quaternion") -> List[List[float]]:
        """Convert the provided unit Quaternion to intrinsic, proper Euler angles.

        There are two solutions (unless the representation is singular, i.e., gimbal lock).

        The function works by equating the quaternion's matrix representation to the Euler angle's
        matrix representation and then solving for the Euler angles. Only a few of the matrix
        elements are necessary.

        The function was created by solving each different Euler angle and then looking at the
        pattern although there is probably a nice mathematical basis for this.
        """
        assert math.isclose(quaternion.norm(), 1), "The quaternion must be normalized"

        # The indices of the axes used (e.g., YZY => 2, 3).
        first_letter: int = ord(self.name[0]) - ord("X") + 1
        second_letter: int = ord(self.name[1]) - ord("X") + 1
        # The number of axes between first and second letter. Take the modulus so only the forward
        # direction is considered (e.g., ZX => 1).
        distance: int = (second_letter - first_letter) % 3

        # The axis not used (e.g., YZY => X).
        missing_letter: int = sum(range(4)) - (first_letter + second_letter)

        a = quaternion[first_letter] * quaternion[second_letter]
        b = quaternion[0] * quaternion[missing_letter]
        c = quaternion[0] * quaternion[second_letter]
        d = quaternion[first_letter] * quaternion[missing_letter]

        # Use atan instead of acos as atan performs better for very small angle values.
        cosine = 1 - 2 * (quaternion[second_letter] ** 2 + quaternion[missing_letter] ** 2)
        beta = math.atan2(math.sqrt(1 - cosine * cosine), cosine)

        if math.isclose(beta, 0):
            # There is no rotation around the second axis so just compute the rotation around the
            # first axis.
            return [[2 * math.acos(quaternion[0]), 0, 0]]

        results = []
        for second in [beta, -beta]:
            sign = 1 if math.sin(second) > 0 else -1

            first = math.atan2((a - b) * sign, (c + d) * sign)
            third = math.atan2((a + b) * sign, (c - d) * sign)

            # If the rotational axes are adjacent, swap the first and third rotation.
            if distance % 2 == 1:
                first, third = third, first

            results.append([first, second, third])

        return results

    def _tait_bryan(self, quaternion: "Quaternion") -> List[List[float]]:
        """Convert the provided Quaternion to intrinsic, Tait-Bryant Euler angles.

        See the notes in the `Axes._proper` function.
        """
        assert math.isclose(quaternion.norm(), 1), "The quaternion must be normalized"

        # The indices of the axes used (e.g., YZX => 2, 3, 1).
        first_letter: int = ord(self.name[0]) - ord("X") + 1
        second_letter: int = ord(self.name[1]) - ord("X") + 1
        third_letter: int = ord(self.name[2]) - ord("X") + 1

        letters = [0, first_letter, second_letter, third_letter]

        m = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for row_index, i in enumerate(letters):
            for column_index, j in enumerate(letters):
                m[row_index][column_index] = 2 * quaternion[i] * quaternion[j]

        invert = 1 if self.name in ["XYZ", "YZX", "ZXY"] else -1
        beta = math.asin(m[0][2] + (invert * m[1][3]))

        results = []
        for second in [beta, math.pi - beta]:
            sign = 1 if math.cos(second) > 0 else -1

            first = math.atan2(
                (m[0][1] - (invert * m[2][3])) * sign, (1 - (m[1][1] + m[2][2])) * sign
            )
            third = math.atan2(
                (m[0][3] - (invert * m[1][2])) * sign, (1 - (m[2][2] + m[3][3])) * sign
            )

            results.append([first, second, third])

        return results
