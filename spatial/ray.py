from typing import Iterable

from .intersection import Intersection
from .transform import Transform
from .vector3 import Vector3


class Ray:
    """A ray given an origin and a direction vector."""

    def __init__(self, origin: Vector3, direction: Vector3) -> None:
        self.origin = origin
        try:
            self.direction = direction.normalize()
        except ZeroDivisionError:
            raise ValueError("The direction vectory must be non-zero") from ZeroDivisionError

    def __str__(self) -> str:
        """Return the string representation of this ray."""
        return f"{self.origin} + t * {self.direction}"

    def closest_intersection(self, collection: Iterable[object]) -> Intersection:
        """Return the closest intersection of the ray with the collection of objects."""
        closest = Intersection.Miss()

        for item in collection:
            # See if the item is intersectable, otherwise ignore it
            if callable(getattr(item, "intersect", None)):
                x = item.intersect(self)

                if x.closer_than(closest):
                    closest = x

        return closest

    def evaluate(self, t: float) -> Vector3:
        """Return the location along ray given parameter t."""
        return self.origin + t * self.direction

    def transform(self, transform: Transform) -> "Ray":
        """Return a transformed ray given the provided transform."""
        new_origin = transform(self.origin, as_type="point")
        new_direction = transform(self.direction, as_type="vector")

        return Ray(new_origin, new_direction)
