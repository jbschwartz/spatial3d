from collections import namedtuple
from typing import Any, Optional


class Intersection(namedtuple("Intersection", "t obj")):
    """An intersection between a ray and an object."""

    __slots__ = ()

    # TODO: It may be necessary to include the chain of intersections that occur to the caller.
    # e.g. Facet -> Mesh -> Link -> Serial
    # This could maybe be implemented with a `previous` attribute on the tuple
    # i.e. __new__(..., previous: 'Intersection')
    def __new__(cls, t: Optional[float], obj: Optional[Any]):
        """Construct a new intersection given a parametric location and the intersected object."""
        if t is not None and t < 0:
            raise ValueError("Intersection can not be behind ray")
        return super().__new__(cls, t, obj)

    @classmethod
    def Miss(cls) -> "Intersection":
        """Construct a new intersection representing no intersection."""
        return cls(None, None)

    @property
    def hit(self) -> bool:
        """Return true if there is a valid intersection location."""
        return self.t is not None

    def closer_than(self, other: "Intersection") -> bool:
        """Return true if this intersection is closer than another Intersection."""
        if self.t is None:
            return False

        return other.t is None or self.t < other.t
