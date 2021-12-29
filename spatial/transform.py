from . import dual, quaternion
from .vector3 import Vector3

Quaternion = quaternion.Quaternion
Dual = dual.Dual


class Transform:
    """Spatial rigid body transformation in three dimensions."""

    def __init__(self, d: Dual = None) -> None:
        self.dual = d or Dual(Quaternion(1, 0, 0, 0), Quaternion(0, 0, 0, 0))

    @classmethod
    def from_axis_angle_translation(
        cls, axis: Vector3 = None, angle: float = 0, translation: Vector3 = None
    ) -> "Transform":
        """Create a Transformation from axis, angle, and translation components."""
        axis = axis or Vector3()
        translation = translation or Vector3()

        return cls.from_orientation_translation(
            Quaternion.from_axis_angle(axis, angle), translation
        )

    @classmethod
    def from_orientation_translation(
        cls, orientation: Quaternion, translation: Vector3 = None
    ) -> "Transform":
        """Create a Transformation from orientation and translation."""
        translation = translation or Vector3()
        return cls(Dual(orientation, 0.5 * Quaternion(0, *translation) * orientation))

    def __mul__(self, other: "Transform") -> "Transform":
        """Compose this Transformation with another Transformation."""
        if isinstance(other, Transform):
            return Transform(self.dual * other.dual)

        return NotImplemented

    @classmethod
    def Identity(cls) -> "Transform":
        """Construct an identity transformation (i.e., a transform that does not transform)."""
        return cls(Dual(Quaternion(1, 0, 0, 0), Quaternion(0, 0, 0, 0)))

    __rmul__ = __mul__

    def __call__(self, vector: Vector3, as_type: str = "point") -> Vector3:
        """Apply Transformation to a Vector3 with call syntax."""
        if isinstance(vector, (list, tuple)):
            return [self.__call__(item, as_type) for item in vector]

        if not isinstance(vector, Vector3):
            raise NotImplementedError

        return self.transform(vector, as_type)

    def transform(self, vector: Vector3, as_type: str) -> Vector3:
        """Apply the transform to the provided Vector3.

        Optionally treat the Vector3 as a point and apply a transformation to its position.
        """
        q = Quaternion(0, *vector.xyz)
        if as_type == "vector":
            d = Dual(q, Quaternion(0, 0, 0, 0))
            a = self.dual * d * dual.conjugate(self.dual)
            return Vector3(*a.r.xyz)

        if as_type == "point":
            d = Dual(Quaternion(), q)
            a = self.dual * d * dual.conjugate(self.dual)
            return Vector3(*a.d.xyz)

        raise KeyError

    # TODO: Make me a property
    def translation(self) -> Vector3:
        """Return the transform's translation Vector3."""
        # "Undo" what was done in the __init__ function by working backwards
        t = 2 * self.dual.d * quaternion.conjugate(self.dual.r)
        return Vector3(*t.xyz)

    # TODO: Make me a property
    def rotation(self) -> Quaternion:
        """Return the transformation's rotation quaternion."""
        return self.dual.r

    def inverse(self) -> "Transform":
        """Return a the inverse of this transformation."""
        return Transform(Dual(quaternion.conjugate(self.dual.r), quaternion.conjugate(self.dual.d)))
