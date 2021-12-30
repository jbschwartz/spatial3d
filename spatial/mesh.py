from typing import Generator

from .aabb import AABB
from .facet import Facet
from .intersection import Intersection
from .kdtree import KDTreeNode
from .ray import Ray
from .transform import Transform
from .vector3 import Vector3


class Mesh:
    """A 3D mesh composed of facets."""

    def __init__(self, name: str = None, facets: list[Facet] = None):
        self.name = name
        self.facets = facets or []
        self.aabb = AABB()

        for facet in self.facets:
            self.aabb.expand(facet.vertices)

        self._accelerator = None

    @classmethod
    def from_file(cls, file_parser: object, file_path: str) -> "Mesh":
        """Construct a mesh from a file."""
        # TODO: The parsers are responsible for actually constructing the Mesh object
        #   Should this be so? Or should it be here?
        meshes = file_parser.parse(file_path)

        for mesh in meshes:
            mesh.accelerator = KDTreeNode

        return meshes

    @property
    def accelerator(self) -> object:
        """Return the spatial accelerator used."""
        return self._accelerator

    @accelerator.setter
    def accelerator(self, accelerator: type) -> None:
        """Set the spatial accelerator used."""
        self._accelerator = accelerator(self.aabb, self.facets)

    def transform(self, transform: Transform) -> "Mesh":
        """Return a mesh transformed by the provided transform."""
        transformed_facets = [facet.transform(transform) for facet in self.facets]

        return Mesh(self.name, transformed_facets)

    def scale(self, scale: float = 1.0) -> "Mesh":
        """Return a mesh scaled about the origin by the provided factor."""
        transformed_facets = [facet.scale(scale) for facet in self.facets]

        return Mesh(self.name, transformed_facets)

    def vertices(self) -> Generator[Vector3, None, None]:
        """Generate list of mesh vertices returned grouped by facet."""
        for facet in self.facets:
            yield facet.vertices[0]
            yield facet.vertices[1]
            yield facet.vertices[2]

    def append(self, facet: Facet) -> None:
        """Add a facet to the mesh."""
        self.aabb.expand(facet.vertices)

        self.facets.append(facet)

        if self.accelerator:
            self.accelerator.update(self, facet)

    def intersect(self, local_ray: Ray) -> Intersection:
        """Return the closest intersection between the ray and mesh.

        Return Intersection.Miss() for no intersection.
        """
        if self.accelerator:
            return self.accelerator.intersect(local_ray)

        # Otherwise we brute force the computation.
        return local_ray.closest_intersection(self.facets)
