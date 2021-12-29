from typing import Generic, TypeVar, Union

from .quaternion import Quaternion, conjugate as quaternion_conjugate

T = TypeVar("T", float, int, Quaternion)


class Dual(Generic[T]):
    """Class for representing dual (numbers) of the form r + dε."""

    def __init__(self, r: T, d: T) -> None:
        self.r: T = r
        self.d: T = d

    def __abs__(self) -> "Dual":
        return Dual(abs(self.r), abs(self.d))

    def __round__(self, places: int) -> "Dual":
        return Dual(round(self.r, places), round(self.d, places))

    def __add__(self, other: "Dual") -> "Dual":
        return Dual(self.r + other.r, self.d + other.d)

    def __sub__(self, other: "Dual") -> "Dual":
        return Dual(self.r - other.r, self.d - other.d)

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

    __rmul__ = __mul__

    def __truediv__(self, other: Union[float, int]) -> "Dual":
        if isinstance(other, (float, int)):
            reciprocal = 1 / other
            return reciprocal * self

        return NotImplemented

    def __eq__(self, other) -> bool:
        if isinstance(other, Dual):
            return self.r == other.r and self.d == other.d

        if isinstance(other, int):
            if other == 0:
                # Used to check Dual == 0 (i.e. check if Dual is the zero Dual)
                # This is useful for unittest.assertAlmostEqual
                return self.r == 0 and self.d == 0

        return NotImplemented

    def __str__(self) -> str:
        return f"{self.r} + {self.d}" + "\u03B5"

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
