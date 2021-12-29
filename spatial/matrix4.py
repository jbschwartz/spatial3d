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
    def from_transform(cls, transform: Transform) -> "Matrix4":
        """Construct a matrix from a transform."""
        return cls.from_dual(transform.dual)

    @classmethod
    def from_dual(cls, dual: Dual[Quaternion]) -> "Matrix4":
        """Construct a matrix from a dual quaternion."""
        assert isinstance(dual.r, Quaternion) and isinstance(dual.d, Quaternion)

        elements = []

        r_star = conjugate(dual.r)
        for basis in [Vector3.X(), Vector3.Y(), Vector3.Z()]:
            transformed_basis = dual.r * Quaternion(0, *basis) * r_star
            elements.extend([*transformed_basis.xyz, 0])

        translation = 2 * dual.d * r_star
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
