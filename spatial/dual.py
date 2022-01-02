from typing import Generic, TypeVar, Union

from .quaternion import Quaternion
from .quaternion import conjugate as quaternion_conjugate

T = TypeVar("T", float, int, Quaternion)


class Dual(Generic[T]):
    """Class for representing dual (numbers) of the form r + dε."""

    def __init__(self, r: T, d: T) -> None:
        self.r: T = r
        self.d: T = d

    def __abs__(self) -> "Dual":
        """Return a dual with the component-wise absolute values of this dual."""
        return Dual(abs(self.r), abs(self.d))

    def __add__(self, other: "Dual") -> "Dual":
        """Return a dual with the component-wise sum of this dual and the other."""
        if isinstance(other, Dual):
            return Dual(self.r + other.r, self.d + other.d)

        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """Return True if this dual is equal to the other."""
        if isinstance(other, Dual):
            return self.r == other.r and self.d == other.d

        if isinstance(other, int):
            if other == 0:
                # Used to check Dual == 0 (i.e. check if Dual is the zero Dual)
                # This is useful for unittest.assertAlmostEqual
                return self.r == 0 and self.d == 0

        return NotImplemented

    def __mul__(self, other: Union["Dual", float, int]) -> "Dual":
        """Dual multiplication.

        If passed a scalar, scalar multiplication is performed.
        If passed a dual, dual multiplication is performed.
        """
        if isinstance(other, (float, int)):
            return Dual(other * self.r, other * self.d)

        if isinstance(other, Dual):
            return Dual(self.r * other.r, self.r * other.d + self.d * other.r)

        return NotImplemented

    def __round__(self, places: int = 0) -> "Dual":
        """Return a dual with the component-wise rounded values of this dual."""
        return Dual(round(self.r, places), round(self.d, places))

    __rmul__ = __mul__

    def __str__(self) -> str:
        """Return the string representation of this dual."""
        return f"{self.r} + {self.d}" + "\u03B5"

    def __sub__(self, other: "Dual") -> "Dual":
        """Return a dual with the component-wise difference of this dual and the other."""
        if isinstance(other, Dual):
            return Dual(self.r - other.r, self.d - other.d)

        return NotImplemented

    def __truediv__(self, other: Union[float, int]) -> "Dual":
        """Return a dual with the component-wise scalar quotient of this dual."""
        if isinstance(other, (float, int)):
            reciprocal = 1 / other
            return reciprocal * self

        return NotImplemented

    def conjugate(self) -> None:
        """Conjugates the dual instance."""
        if isinstance(self.r, Quaternion) and isinstance(self.d, Quaternion):
            self.r.conjugate()
            self.d = -quaternion_conjugate(self.d)
        elif isinstance(self.r, (int, float)) and isinstance(self.d, (int, float)):
            self.d = -self.d
        else:
            raise NotImplementedError


def conjugate(dual: Dual) -> Dual:
    """Return the conjugate of the provided Dual quaternion."""
    if isinstance(dual.r, Quaternion) and isinstance(dual.d, Quaternion):
        return Dual(quaternion_conjugate(dual.r), -quaternion_conjugate(dual.d))

    raise NotImplementedError
