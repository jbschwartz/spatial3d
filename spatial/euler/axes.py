# Letter variable names in this file match common mathematical notation.
# Local variables maintain more concise mathematical equations.
# pylint: disable=invalid-name,too-many-locals

import enum
import functools
import math

from spatial.vector3 import Vector3

# All of these functions convert quaternion representations to intrinsic euler angles.
# The quaternion representation is first converted to a partial matrix representation.
# The partial matrix representation is then converted to an euler angle representation.
# There are two solutions in general (unless the representation is singular, i.e. gimbal lock).


def _zyz(r: float, x: float, y: float, z: float) -> list[list[float]]:
    """Return ZYZ Euler angles from quaternion components."""
    xz = 2 * x * z
    ry = 2 * r * y
    yz = 2 * y * z
    rx = 2 * r * x

    beta = math.acos(1 - 2 * (x ** 2 + y ** 2))
    if math.isclose(beta, 0):
        # Y is zero so there are multiple solutions (infinitely many?). Pick [0, 0, 0]
        # TODO: Confirm that this is a valid way to handle the singular configuration.
        #   That is, is there ever a case where choosing [0, 0, 0] is wrong?
        return [[0, 0, 0]]

    results = []
    for yp in [beta, -beta]:
        sign = 1 if math.sin(yp) > 0 else -1

        zpp = math.atan2((yz + rx) * sign, (-xz + ry) * sign)
        z = math.atan2((yz - rx) * sign, (xz + ry) * sign)
        results.append([z, yp, zpp])

    return results


def _zyx(r: float, x: float, y: float, z: float) -> list[list[float]]:
    """Return ZYX Euler angles from quaternion components."""
    xy = 2 * x * y
    xz = 2 * x * z
    ry = 2 * r * y
    yz = 2 * y * z
    rx = 2 * r * x
    rz = 2 * r * z

    xSq = x ** 2
    ySq = y ** 2
    zSq = z ** 2

    beta = math.asin(-xz + ry)

    results = []
    for yp in [beta, math.pi + beta]:
        sign = 1 if math.cos(yp) > 0 else -1

        xpp = math.atan2((yz + rx) * sign, (1 - 2 * (xSq + ySq)) * sign)
        z = math.atan2((xy + rz) * sign, (1 - 2 * (ySq + zSq)) * sign)

        results.append([z, yp, xpp])

    return results


def _not_implemented(r: float, x: float, y: float, z: float) -> None:
    """Raise for conversion functions which are not implemented."""
    raise NotImplementedError


class Axes(enum.Enum):
    """All the different types of Euler angles."""

    ZXZ = functools.partial(_not_implemented)
    XYX = functools.partial(_not_implemented)
    YZY = functools.partial(_not_implemented)
    ZYZ = functools.partial(_zyz)
    XZX = functools.partial(_not_implemented)
    YXY = functools.partial(_not_implemented)
    XYZ = functools.partial(_not_implemented)
    YZX = functools.partial(_not_implemented)
    ZXY = functools.partial(_not_implemented)
    XZY = functools.partial(_not_implemented)
    ZYX = functools.partial(_zyx)
    YXZ = functools.partial(_not_implemented)

    @classmethod
    def basis_vector(cls, axis: str) -> Vector3:
        """Return a basis vector corresponding to the provided axis letter."""
        v = Vector3()

        # For now, allow exceptions for unrecognized `axis` to go uncaught
        setattr(v, axis, 1)
        return v

    def convert(self, quaternion: "Quaternion") -> list[list[float]]:
        """Return the Euler angles from the provided quaternion."""
        return self.value(*quaternion)

    def reverse(self) -> "Axes":
        """Return the reversed order Euler angles."""
        reversed_name = self.name[::-1]
        return Axes[reversed_name]

    @property
    def vectors(self) -> list[Vector3]:
        """Return the basis vectors corresponding to the Euler angles axes."""
        return [Axes.basis_vector(axis.lower()) for axis in self.name]
