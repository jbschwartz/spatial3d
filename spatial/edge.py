from functools import cached_property

from .vector3 import Vector3


class Edge:
    """An edge created by two points."""

    def __init__(self, start: Vector3, end: Vector3) -> None:
        self.start = start
        self.end = end

    def __eq__(self, other: object) -> bool:
        """Return True if this edge is equal to the other."""
        if isinstance(other, Edge):
            if self.start == other.start and self.end == other.end:
                return True

            if self.start == other.end and self.end == other.start:
                return True

            return False

        return NotImplemented

    @cached_property
    def vector(self) -> Vector3:
        """Return the edge's vector from start to end."""
        return self.end - self.start
