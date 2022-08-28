import math
from typing import List, Union

from .euler import Axes, Order
from .swizzler import Swizzler
from .vector3 import Vector3


class Quaternion(Swizzler):
    """A quaternion of the form r + xi + yj + zk."""

    __slots__ = ["r", "x", "y", "z"]

    def __init__(self, r: float = 1, x: float = 0, y: float = 0, z: float = 0):
        self.r = r
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_axis_angle(cls, axis: Vector3, angle: float) -> "Quaternion":
        """Construct a quaternion from an axis and angle (in radians)."""
        half_angle = angle / 2
        try:
            axis = math.sin(half_angle) * axis.normalize()
        except ZeroDivisionError:
            axis = Vector3()

        return cls(math.cos(half_angle), axis.x, axis.y, axis.z)

    @classmethod
    def from_basis(cls, x: Vector3, y: Vector3, z: Vector3):
        """Construct a quaternion from three mutually perpendicular basis vectors."""
        assert all(
            [x.is_perpendicular_to(y), y.is_perpendicular_to(z), z.is_perpendicular_to(x)]
        ), "All basis vectors must be mutually perpendicular"

        # Implementing the algorithm described in "Converting a Rotation Matrix to a Quaternion" by
        # Mike Day, Insomniac Games
        if z.z < 0:
            if x.x > y.y:
                t = 1 + x.x - y.y - z.z
                quaternion = cls(y.z - z.y, t, x.y + y.x, z.x + x.z)
            else:
                t = 1 - x.x + y.y - z.z
                quaternion = cls(z.x - x.z, x.y + y.x, t, y.z + z.y)
        else:
            if x.x < -y.y:
                t = 1 - x.x - y.y + z.z
                quaternion = cls(x.y - y.x, z.x + x.z, y.z + z.y, t)
            else:
                t = 1 + x.x + y.y + z.z
                quaternion = cls(t, y.z - z.y, z.x - x.z, x.y - y.x)

        return (0.5 / math.sqrt(t)) * quaternion

    @classmethod
    def from_euler(cls, angles: List[float], axes: Axes, order: Order) -> "Quaternion":
        """Construct a quaternion from euler angles (in radians)."""
        quaternion = cls()

        if order == Order.EXTRINSIC:
            angles.reverse()
            axes = axes.reverse()

        for axis, angle in zip(axes.vectors, angles):
            quaternion *= cls.from_axis_angle(axis=axis, angle=angle)

        return quaternion

    @classmethod
    def from_vector(cls, vector: Vector3) -> "Quaternion":
        """Construct a quaternion from a vector."""
        return cls(0, *vector)

    def __abs__(self) -> "Quaternion":
        """Return a quaternion with the component-wise absolute values of this quaternion."""
        return Quaternion(abs(self.r), abs(self.x), abs(self.y), abs(self.z))

    def __add__(self, other: "Quaternion") -> "Quaternion":
        """Return a quaternion with the component-wise sum of this quaternion and the other."""
        if isinstance(other, Quaternion):
            return Quaternion(
                self.r + other.r, self.x + other.x, self.y + other.y, self.z + other.z
            )

        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """Return True if this quaternion is equal to the other."""
        if isinstance(other, Quaternion):
            return (
                self.r == other.r and self.x == other.x and self.y == other.y and self.z == other.z
            )

        if isinstance(other, int):
            if other == 0:
                # Used to check Quaternion == 0 (i.e. check if Quaternion is the zero Quaternion)
                # This is useful for unittest.assertAlmostEqual
                return self.r == 0 and self.x == 0 and self.y == 0 and self.z == 0

        return NotImplemented

    def __format__(self, spec: str = "") -> str:
        """Return the string representation following the format specification component-wise."""
        return f"({self.r:{spec}}, {self.x:{spec}}, {self.y:{spec}}, {self.z:{spec}})"

    def __getitem__(self, index: int) -> float:
        """Return the value of the component at the provided index."""
        components = ["r", "x", "y", "z"]
        return getattr(self, components[index])

    def __mul__(self, other: Union[float, int, "Quaternion"]) -> "Quaternion":
        """Quaternion multiplication.

        If passed a number, scalar multiplication is performed.
        If passed another quaternion, quaternion multiplication is performed.
        """
        if isinstance(other, (float, int)):
            return Quaternion(self.r * other, self.x * other, self.y * other, self.z * other)

        if isinstance(other, Quaternion):
            r = self.r * other.r - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.r * other.x + self.x * other.r + self.y * other.z - self.z * other.y
            y = self.r * other.y - self.x * other.z + self.y * other.r + self.z * other.x
            z = self.r * other.z + self.x * other.y - self.y * other.x + self.z * other.r

            return Quaternion(r, x, y, z)

        return NotImplemented

    def __neg__(self) -> "Quaternion":
        """Return a quaternion with the component-wise negation of this quaternion."""
        return Quaternion(-self.r, -self.x, -self.y, -self.z)

    __rmul__ = __mul__

    def __round__(self, places: int = 0) -> "Quaternion":
        """Return a quaternion with the component-wise rounded values of this quaternion."""
        return Quaternion(
            round(self.r, places),
            round(self.x, places),
            round(self.y, places),
            round(self.z, places),
        )

    def __setitem__(self, index: int, value: float) -> None:
        """Set the value of the component at the provided index."""
        components = ["r", "x", "y", "z"]
        setattr(self, components[index], value)

    def __str__(self) -> str:
        """Return the string representation of this quaternion."""
        return format(self, "")

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        """Return a quaternion with the component-wise difference of this and the other."""
        if isinstance(other, Quaternion):
            return Quaternion(
                self.r - other.r, self.x - other.x, self.y - other.y, self.z - other.z
            )

        return NotImplemented

    # pylint: disable=duplicate-code
    def __truediv__(self, other: Union[float, int]) -> "Quaternion":
        """Return a quaternion with the component-wise scalar quotient of this quaternion."""
        if isinstance(other, (float, int)):
            reciprocal = 1 / other
            return reciprocal * self

        return NotImplemented

    @property
    def vector(self) -> Vector3:
        """Return the vector component of this quaternion."""
        return Vector3(self.x, self.y, self.z)

    def conjugate(self) -> None:
        """Conjugates the quaternion instance."""
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z

    def dot(self, other: "Quaternion") -> float:
        """Return the dot product of this quaternion and the other."""
        return self.r * other.r + self.x * other.x + self.y * other.y + self.z * other.z

    # TODO: I need to make the naming consistent across all objects (i.e., I use length in Vector3)
    def norm(self) -> float:
        """Return the length of the quaternion."""
        return math.sqrt(self.r**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Quaternion":
        """Normalize the quaternion instance (i.e. norm of one)."""
        norm = self.norm()
        self.r /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm
        return self

    def rotate(self, vector: Vector3) -> Vector3:
        """Return the provided vector rotated by this quaternion."""
        result = self * Quaternion.from_vector(vector) * conjugate(self)
        return Vector3(*result.xyz)


def conjugate(q: Quaternion) -> Quaternion:
    """Return the conjugate of the provided quaternion."""
    return Quaternion(q.r, -q.x, -q.y, -q.z)


def slerp(q1: Quaternion, q2: Quaternion, alpha: float, shortest_path: bool = True) -> Quaternion:
    """Return the spherical linear interpolation between q1 and q2.

    By default, the function follows the shortest path.
    """
    assert 0 <= alpha <= 1, "Interpolation value must be between 0 and 1"
    assert math.isclose(q1.norm(), 1) and math.isclose(
        q2.norm(), 1
    ), "Both quaternions must be unit length"

    dot = q1.dot(q2)

    # Ensure the shortest path is used by negating the second quaternion when the dot product is
    # negative.
    if dot < 0 and shortest_path:
        q2 = -q2
        dot = q1.dot(q2)

    # Clamp dot between [1, -1] to avoid domain errors in the subsequent arccosine.
    dot = min(1, max(-1, dot))

    theta = math.acos(dot) * alpha

    q = (q2 - (q1 * dot)).normalize()

    return q1 * math.cos(theta) + q * math.sin(theta)
