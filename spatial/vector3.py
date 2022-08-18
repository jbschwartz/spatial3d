import math
from typing import Union, overload

from .swizzler import Swizzler


class Vector3(Swizzler):
    """A 3D Vector."""

    __slots__ = ["x", "y", "z"]

    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def X(cls) -> "Vector3":
        """Construct the standard X unit basis vector."""
        return cls(1, 0, 0)

    @classmethod
    def Y(cls) -> "Vector3":
        """Construct the standard Y unit basis vector."""
        return cls(0, 1, 0)

    @classmethod
    def Z(cls) -> "Vector3":
        """Construct the standard Z unit basis vector."""
        return cls(0, 0, 1)

    def __abs__(self) -> "Vector3":
        """Return a vector with the component-wise absolute values of this vector."""
        return Vector3(abs(self.x), abs(self.y), abs(self.z))

    def __add__(self, other: "Vector3") -> "Vector3":
        """Return a vector with the component-wise sum of this vector and the other."""
        if isinstance(other, Vector3):
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """Return True if this vector is equal to the other."""
        if isinstance(other, Vector3):
            return self.x == other.x and self.y == other.y and self.z == other.z

        if isinstance(other, int):
            if other == 0:
                # Used to check Vector3 == 0 (i.e. check if Vector3 is the zero vector)
                # This is useful for unittest.assertAlmostEqual
                return self.x == 0 and self.y == 0 and self.z == 0
            return NotImplemented

        return NotImplemented

    def __format__(self, spec: str = "") -> str:
        """Return the string representation following the format specification component-wise."""
        return f"({self.x:{spec}}, {self.y:{spec}}, {self.z:{spec}})"

    def __getitem__(self, index: int) -> float:
        """Return the value of the component at the provided index."""
        components = ["x", "y", "z"]
        return getattr(self, components[index])

    def __len__(self) -> int:
        """Return the number of (coordinate) dimensions in this vector."""
        return 3

    def __mod__(self, other: "Vector3") -> "Vector3":
        """Return the cross product of this vector with the other."""
        return cross(self, other)

    @overload
    def __mul__(self, other: "Vector3") -> float:  # noqa: D105
        ...

    @overload
    def __mul__(self, other: Union[float, int]) -> "Vector3":  # noqa: D105
        ...

    def __mul__(self, other: Union["Vector3", float, int]) -> Union["Vector3", float]:
        """Return the dot product of this vector with the other.

        If `other` is a scalar, return a vector with the component-wise multiple of this vector.
        """
        if isinstance(other, (float, int)):
            return Vector3(other * self.x, other * self.y, other * self.z)

        if isinstance(other, Vector3):
            # Dot product
            return self.x * other.x + self.y * other.y + self.z * other.z

        return NotImplemented

    def __neg__(self) -> "Vector3":
        """Return a vector with the component-wise negation of this vector."""
        return Vector3(-self.x, -self.y, -self.z)

    def __repr__(self) -> str:
        """Return a string representation of this vector's construction."""
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"

    __rmul__ = __mul__

    def __round__(self, places: int = 0) -> "Vector3":
        """Return a vector with the component-wise rounded values of this vector."""
        return Vector3(round(self.x, places), round(self.y, places), round(self.z, places))

    def __setitem__(self, index: int, value: float) -> None:
        """Set the value of the component at the provided index."""
        components = ["x", "y", "z"]
        setattr(self, components[index], value)

    def __str__(self) -> str:
        """Return the string representation of this vector."""
        return format(self, "")

    def __sub__(self, other: "Vector3") -> "Vector3":
        """Return a vector with the component-wise difference of this vector and the other."""
        if isinstance(other, Vector3):
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

        return NotImplemented

    # pylint: disable=duplicate-code
    def __truediv__(self, other: Union[float, int]) -> "Vector3":
        """Return a vector with the component-wise scalar quotient of this vector."""
        if isinstance(other, (float, int)):
            reciprocal = 1 / other
            return reciprocal * self

        return NotImplemented

    def is_perpendicular_to(self, other: "Vector3", tolerance: float = 0.00001) -> bool:
        """Return True if the vector is perpendicular to the other vector."""
        return math.isclose(self * other, 0, abs_tol=tolerance)

    def is_unit(self, tolerance: float = 0.00001) -> bool:
        """Return True if the vector has unit length (within a given tolerance)."""
        return math.isclose(self.length_sq(), 1.0, abs_tol=tolerance)

    def length(self) -> float:
        """Return the length of the vector."""
        return math.sqrt(self.length_sq())

    def length_sq(self) -> float:
        """Return the squared length of the vector."""
        return self.x**2 + self.y**2 + self.z**2

    def normalize(self) -> "Vector3":
        """Normalize this vector to unit length.

        This function will raise an exception if the vector has zero length.
        """
        length = self.length()
        self.x /= length
        self.y /= length
        self.z /= length
        return self


def almost_equal(v1: Vector3, v2: Vector3, tol=0.00001) -> bool:
    """Return True if two vectors are equal within a given tolerance."""
    difference = v1 - v2

    return math.isclose(difference.length_sq(), 0.0, abs_tol=tol)


def angle_between(v1: Vector3, v2: Vector3) -> float:
    """Return the angle between two vectors in radians."""
    dot = v1 * v2
    lengths = v1.length() * v2.length()

    return math.acos(dot / lengths)


def cross(v1: Vector3, v2: Vector3) -> Vector3:
    """Return the cross product of two vectors."""
    return Vector3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x)


def normalize(v: Vector3) -> Vector3:
    """Return the vector normalized to unit length.

    This function will raise an exception if the vector has zero length.
    """
    length = v.length()
    return Vector3(v.x / length, v.y / length, v.z / length)
