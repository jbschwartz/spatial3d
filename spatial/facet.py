from functools import cached_property
from typing import Iterable

from .aabb import AABB
from .edge import Edge
from .exceptions import DegenerateTriangleError
from .intersection import Intersection
from .ray import Ray
from .transform import Transform
from .vector3 import Vector3


class Facet:
    """A piece of surface geometry (typically a triangle)."""

    def __init__(self, vertices: Iterable[Vector3], normal: Vector3 = None) -> None:
        self.vertices = vertices
        self.normal = normal or self.computed_normal

    @cached_property
    def aabb(self) -> AABB:
        """Return the facet's axis aligned bounding box."""
        return AABB(self.vertices)

    @cached_property
    def computed_normal(self) -> Vector3:
        """Return a normal computed from the facet's edges."""
        try:
            return (self.edges[0].vector % self.edges[1].vector).normalize()
        except ZeroDivisionError:
            raise DegenerateTriangleError("Degenerate triangle found") from ZeroDivisionError

    @cached_property
    def edges(self) -> list[Edge]:
        """Return a list of edges."""
        edges = [Edge(v1, v2) for v1, v2 in zip(self.vertices, self.vertices[1:])]
        edges.append(Edge(self.vertices[-1], self.vertices[0]))

        return edges

    @property
    def is_triangle(self) -> bool:
        """Return true if the facet has three vertices."""
        return len(self.vertices) == 3

    def intersect(self, ray: Ray, check_back_facing: bool = False) -> Intersection:
        """Return the Intersection with parametric value of ray (or Intersection.Miss() for a miss).

        Returns Intersection.Miss() when the ray origin is in the triangle and the ray points away.
        Returns the ray origin when the ray origin is in the triangle and the ray points towards.

        This function implements the Moller-Trumbore intersection algorithm.
        """
        E1 = self.edges[0].vector
        E2 = -self.edges[2].vector
        P = ray.direction % E2

        det = P * E1

        if not check_back_facing and det < 0:
            # The ray intersects the back of the triangle
            return Intersection.Miss()

        try:
            inv_det = 1 / det
        except ZeroDivisionError:
            # The ray is parallel to the triangle
            return Intersection.Miss()

        T = ray.origin - self.vertices[0]
        Q = T % E1

        u = (P * T) * inv_det
        v = Q * ray.direction * inv_det

        # Checking if the point of intersection is outside the bounds of the triangle
        if not (0 <= u <= 1) or (v < 0) or (u + v > 1):
            return Intersection.Miss()

        t = Q * E2 / det

        return Intersection(t, self)

    def scale(self, scale: float = 1) -> "Facet":
        """Return a facet scaled by the provided scale factor."""
        transformed_vertices = [scale * v for v in self.vertices]

        return Facet(transformed_vertices, self.computed_normal)

    def transform(self, transform: Transform) -> "Facet":
        """Return a facet transformed with the provided transform."""
        transformed_normal = transform(self.normal, as_type="vector")
        transformed_vertices = [transform(v, as_type="point") for v in self.vertices]

        return Facet(transformed_vertices, transformed_normal)
