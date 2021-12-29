from typing import Iterable, Union

from . import dual, quaternion
from .vector3 import Vector3

Quaternion = quaternion.Quaternion
Dual = dual.Dual


class Transform:
    """Spatial rigid body transformation in three dimensions."""

    def __init__(self, d: Dual = None) -> None:
        # Default to the identity transformation (i.e., a transform that does not transform).
        self.dual = d or Dual(Quaternion(1, 0, 0, 0), Quaternion(0, 0, 0, 0))

    @classmethod
    def Identity(cls) -> "Transform":
        """Alias to construct an identity transform."""
        return cls()

    @classmethod
    def from_axis_angle_translation(
        cls, axis: Vector3 = None, angle: float = 0, translation: Vector3 = None
    ) -> "Transform":
        """Create a transform from axis, angle, and translation components."""
        axis = axis or Vector3()
        translation = translation or Vector3()

        return cls.from_orientation_translation(
            Quaternion.from_axis_angle(axis, angle), translation
        )

    @classmethod
    def from_orientation_translation(
        cls, orientation: Quaternion, translation: Vector3 = None
    ) -> "Transform":
        """Create a transform from orientation and translation."""
        translation = translation or Vector3()
        return cls(Dual(orientation, 0.5 * Quaternion.from_vector(translation) * orientation))

    def __call__(
        self, vector: Union[Iterable[Vector3], Vector3], as_type: str = "point"
    ) -> Union[Iterable[Vector3], Vector3]:
        """Apply this transform to a vector with call syntax."""
        if isinstance(vector, (list, tuple)):
            return [self.__call__(item, as_type) for item in vector]

        if not isinstance(vector, Vector3):
            return NotImplemented

        return self.transform(vector, as_type)

    def __mul__(self, other: "Transform") -> "Transform":
        """Compose this transform with another transform."""
        if isinstance(other, Transform):
            return Transform(self.dual * other.dual)

        return NotImplemented

    __rmul__ = __mul__

    @property
    def rotation(self) -> Quaternion:
        """Return the transform's rotation quaternion."""
        return self.dual.r

    @property
    def translation(self) -> Vector3:
        """Return the transform's translation vector."""
        # "Undo" what was done in the __init__ function by working backwards
        t = 2 * self.dual.d * quaternion.conjugate(self.dual.r)
        return t.vector

    def inverse(self) -> "Transform":
        """Return a the inverse of this transform."""
        return Transform(Dual(quaternion.conjugate(self.dual.r), quaternion.conjugate(self.dual.d)))

    def transform(self, vector: Vector3, as_type: str = "point") -> Vector3:
        """Apply the transform to the provided vector.

        Optionally treat the vector as a point and apply to its position.
        """
        q = Quaternion.from_vector(vector)
        if as_type == "vector":
            return self.dual.r.rotate(vector)

        if as_type == "point":
            d = Dual(Quaternion(), q)
            a = self.dual * d * dual.conjugate(self.dual)
            return a.d.vector

        raise KeyError(f"Unknown transform type: {as_type}")
