# Letter variable names in this file match common mathematical notation.
# Local variables maintain more concise mathematical equations.
# pylint: disable=invalid-name,too-many-locals

import enum
import functools
import math
from typing import List

from spatial.vector3 import Vector3

# All of these functions convert quaternion representations to intrinsic euler angles.
# The quaternion representation is first converted to a partial matrix representation.
# The partial matrix representation is then converted to an euler angle representation.
# There are two solutions in general (unless the representation is singular, i.e. gimbal lock).


def _xyx(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return XYX Euler angles from quaternion components."""
    xz = x * z
    ry = r * y
    xy = x * y
    rz = r * z

    beta = math.acos(1 - 2 * (y**2 + z**2))
    if math.isclose(beta, 0):
        # Y is zero so there are multiple solutions (infinitely many?). Pick [0, 0, 0]
        # TODO: Confirm that this is a valid way to handle the singular configuration.
        #   That is, is there ever a case where choosing [0, 0, 0] is wrong?
        return [[0, 0, 0]]

    results = []
    for zp in [beta, -beta]:
        sign = 1 if math.sin(zp) > 0 else -1

        xpp = math.atan2((xy - rz) * sign, (ry + xz) * sign)
        x = math.atan2((xy + rz) * sign, (ry - xz) * sign)
        results.append([x, zp, xpp])

    return results


def _xzx(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return XZX Euler angles from quaternion components."""
    xz = x * z
    ry = r * y
    rz = r * z
    xy = x * y

    beta = math.acos(1 - 2 * (y**2 + z**2))
    if math.isclose(beta, 0):
        # Y is zero so there are multiple solutions (infinitely many?). Pick [0, 0, 0]
        # TODO: Confirm that this is a valid way to handle the singular configuration.
        #   That is, is there ever a case where choosing [0, 0, 0] is wrong?
        return [[0, 0, 0]]

    results = []
    for zp in [beta, -beta]:
        sign = 1 if math.sin(zp) > 0 else -1

        xpp = math.atan2((xz + ry) * sign, (rz - xy) * sign)
        x = math.atan2((xz - ry) * sign, (rz + xy) * sign)
        results.append([x, zp, xpp])

    return results


def _yxy(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return YXY Euler angles from quaternion components."""
    xy = x * y
    rz = r * z
    yz = y * z
    rx = r * x

    beta = math.acos(1 - 2 * (x**2 + z**2))
    if math.isclose(beta, 0):
        # Y is zero so there are multiple solutions (infinitely many?). Pick [0, 0, 0]
        # TODO: Confirm that this is a valid way to handle the singular configuration.
        #   That is, is there ever a case where choosing [0, 0, 0] is wrong?
        return [[0, 0, 0]]

    results = []
    for zp in [beta, -beta]:
        sign = 1 if math.sin(zp) > 0 else -1

        xpp = math.atan2((xy + rz) * sign, (rx - yz) * sign)
        x = math.atan2((xy - rz) * sign, (rx + yz) * sign)
        results.append([x, zp, xpp])

    return results


def _yzy(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return YZY Euler angles from quaternion components."""
    yz = y * z
    rx = r * x
    xy = x * y
    rz = r * z

    beta = math.acos(1 - 2 * (x**2 + z**2))
    if math.isclose(beta, 0):
        # Y is zero so there are multiple solutions (infinitely many?). Pick [0, 0, 0]
        # TODO: Confirm that this is a valid way to handle the singular configuration.
        #   That is, is there ever a case where choosing [0, 0, 0] is wrong?
        return [[0, 0, 0]]

    results = []
    for zp in [beta, -beta]:
        sign = 1 if math.sin(zp) > 0 else -1

        xpp = math.atan2((yz - rx) * sign, (rz + xy) * sign)
        x = math.atan2((yz + rx) * sign, (rz - xy) * sign)
        results.append([x, zp, xpp])

    return results


def _zxz(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return ZXZ Euler angles from quaternion components."""
    xz = x * z
    ry = r * y
    yz = y * z
    rx = r * x

    beta = math.acos(1 - 2 * (x**2 + y**2))
    if math.isclose(beta, 0):
        # Y is zero so there are multiple solutions (infinitely many?). Pick [0, 0, 0]
        # TODO: Confirm that this is a valid way to handle the singular configuration.
        #   That is, is there ever a case where choosing [0, 0, 0] is wrong?
        return [[0, 0, 0]]

    results = []
    for yp in [beta, -beta]:
        sign = 1 if math.sin(yp) > 0 else -1

        zpp = math.atan2((xz - ry) * sign, (rx + yz) * sign)
        z = math.atan2((xz + ry) * sign, (rx - yz) * sign)
        results.append([z, yp, zpp])

    return results


def _zyz(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return ZYZ Euler angles from quaternion components."""
    xz = x * z
    ry = r * y
    yz = y * z
    rx = r * x

    beta = math.acos(1 - 2 * (x**2 + y**2))
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


def _xyz(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return XYZ Euler angles from quaternion components."""
    xy = 2 * x * y
    xz = 2 * x * z
    yz = 2 * y * z
    rx = 2 * r * x
    ry = 2 * r * y
    rz = 2 * r * z

    xSq = x**2
    ySq = y**2
    zSq = z**2

    beta = math.asin(ry + xz)

    results = []
    for second in [beta, math.pi - beta]:
        sign = 1 if math.cos(second) > 0 else -1

        first = math.atan2((rx - yz) * sign, (1 - 2 * (xSq + ySq)) * sign)
        third = math.atan2((rz - xy) * sign, (1 - 2 * (ySq + zSq)) * sign)

        results.append([first, second, third])

    return results


def _xzy(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return XZY Euler angles from quaternion components."""
    xy = 2 * x * y
    xz = 2 * x * z
    yz = 2 * y * z
    rx = 2 * r * x
    ry = 2 * r * y
    rz = 2 * r * z

    xSq = x**2
    ySq = y**2
    zSq = z**2

    beta = math.asin(rz - xy)

    results = []
    for second in [beta, math.pi - beta]:
        sign = 1 if math.cos(second) > 0 else -1

        first = math.atan2((rx + yz) * sign, (1 - 2 * (xSq + zSq)) * sign)
        third = math.atan2((ry + xz) * sign, (1 - 2 * (ySq + zSq)) * sign)

        results.append([first, second, third])

    return results


def _yxz(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return YZX Euler angles from quaternion components."""
    xy = 2 * x * y
    xz = 2 * x * z
    yz = 2 * y * z
    rx = 2 * r * x
    ry = 2 * r * y
    rz = 2 * r * z

    xSq = x**2
    ySq = y**2
    zSq = z**2

    beta = math.asin(rx - yz)

    results = []
    for second in [beta, math.pi - beta]:
        sign = 1 if math.cos(second) > 0 else -1

        first = math.atan2((ry + xz) * sign, (1 - 2 * (xSq + ySq)) * sign)
        third = math.atan2((rz + xy) * sign, (1 - 2 * (xSq + zSq)) * sign)

        results.append([first, second, third])

    return results


def _yzx(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return YZX Euler angles from quaternion components."""
    xy = 2 * x * y
    xz = 2 * x * z
    yz = 2 * y * z
    rx = 2 * r * x
    ry = 2 * r * y
    rz = 2 * r * z

    xSq = x**2
    ySq = y**2
    zSq = z**2

    beta = math.asin(rz + xy)

    results = []
    for second in [beta, math.pi - beta]:
        sign = 1 if math.cos(second) > 0 else -1

        first = math.atan2((ry - xz) * sign, (1 - 2 * (ySq + zSq)) * sign)
        third = math.atan2((rx - yz) * sign, (1 - 2 * (xSq + zSq)) * sign)

        results.append([first, second, third])

    return results


def _zxy(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return ZXY Euler angles from quaternion components."""
    xy = 2 * x * y
    xz = 2 * x * z
    yz = 2 * y * z
    rx = 2 * r * x
    ry = 2 * r * y
    rz = 2 * r * z

    xSq = x**2
    ySq = y**2
    zSq = z**2

    beta = math.asin(rx + yz)

    results = []
    for second in [beta, math.pi - beta]:
        sign = 1 if math.cos(second) > 0 else -1

        first = math.atan2((rz - xy) * sign, (1 - 2 * (xSq + zSq)) * sign)
        third = math.atan2((ry - xz) * sign, (1 - 2 * (xSq + ySq)) * sign)

        results.append([first, second, third])

    return results


def _zyx(r: float, x: float, y: float, z: float) -> List[List[float]]:
    """Return ZYX Euler angles from quaternion components."""
    xy = 2 * x * y
    xz = 2 * x * z
    yz = 2 * y * z
    rx = 2 * r * x
    ry = 2 * r * y
    rz = 2 * r * z

    xSq = x**2
    ySq = y**2
    zSq = z**2

    beta = math.asin(ry - xz)

    results = []
    for yp in [beta, math.pi - beta]:
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

    XYX = functools.partial(_xyx)
    XZX = functools.partial(_xzx)
    YXY = functools.partial(_yxy)
    YZY = functools.partial(_yzy)
    ZXZ = functools.partial(_zxz)
    ZYZ = functools.partial(_zyz)
    XYZ = functools.partial(_xyz)
    YZX = functools.partial(_yzx)
    ZXY = functools.partial(_zxy)
    XZY = functools.partial(_xzy)
    ZYX = functools.partial(_zyx)
    YXZ = functools.partial(_yxz)

    @classmethod
    def basis_vector(cls, axis: str) -> Vector3:
        """Return a basis vector corresponding to the provided axis letter."""
        v = Vector3()

        # For now, allow exceptions for unrecognized `axis` to go uncaught
        setattr(v, axis, 1)
        return v

    def convert(self, quaternion: "Quaternion") -> List[List[float]]:
        """Return the Euler angles from the provided quaternion."""
        return self.value(*quaternion)

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
