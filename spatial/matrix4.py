import math
from typing import Iterable

from .dual import Dual
from .quaternion import Quaternion, conjugate
from .transform import Transform
from .vector3 import Vector3


class Matrix4:
    """A column-major 4 x 4 Matrix.

    Elements in groups of four form columns (e.g. self.elements[0:4] is the first column).
    """

    def __init__(self, elements: Iterable[float] = None) -> None:
        if elements:
            if len(elements) != 16:
                raise TypeError(
                    f"Matrix4 requires 16 floating point elements. Received {len(elements)}"
                )

            self.elements = elements
        else:
            self.elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    @classmethod
    def from_basis(cls, x: Vector3, y: Vector3, z: Vector3, origin: Vector3 = None):
        """Construct a quaternion from three mutually perpendicular basis vectors."""
        assert all(
            [x.is_perpendicular_to(y), y.is_perpendicular_to(z), z.is_perpendicular_to(x)]
        ), "All basis vectors must be mutually perpendicular"

        assert all(
            [math.isclose(x.length(), 1), math.isclose(y.length(), 1), math.isclose(z.length(), 1)]
        ), "All basis vectors must be unit length"

        o = origin or Vector3()
        return cls([x.x, x.y, x.z, 0, y.x, y.y, y.z, 0, z.x, z.y, z.z, 0, o.x, o.y, o.z, 1])

    @classmethod
    def from_transform(cls, transform: Transform) -> "Matrix4":
        """Construct a matrix from a transform."""
        return cls.from_dual(transform.dual)

    @classmethod
    def from_dual(cls, dual: Dual[Quaternion]) -> "Matrix4":
        """Construct a matrix from a dual quaternion."""
        assert isinstance(dual.r, Quaternion) and isinstance(dual.d, Quaternion)

        elements = []

        for basis in [Vector3.X(), Vector3.Y(), Vector3.Z()]:
            elements.extend([*dual.r.rotate(basis), 0])

        translation = 2 * dual.d * conjugate(dual.r)
        elements.extend([*translation.xyz, 1])

        return cls(elements)

    def __getitem__(self, index: int) -> float:
        """Return the value of the matrix at the provided index."""
        return self.elements[index]

    def __str__(self) -> str:
        """Return the string representation of this matrix."""
        # Get the width of "widest" floating point number
        longest = max(map(len, map(lambda e: f"{e:.4f}", self.elements)))
        # Pad the left of each element to the widest number found
        padded = list(map(lambda elem: f"{elem:>{longest}.4f}", self.elements))

        # Since the values are stored column-major, we need to "transpose"
        columns = [padded[i : i + 4] for i in range(0, 15, 4)]

        # Join columns by commas and rows by new lines
        return "\n".join(map(", ".join, zip(*columns)))
