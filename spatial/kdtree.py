from typing import Tuple

from .aabb import AABB
from .coordinate_axes import CoordinateAxes
from .facet import Facet
from .intersection import Intersection
from .ray import Ray

# The deepest level that the KDTree will branch during construction.
DEPTH_BOUND = 8


class KDTreeNode:
    """A node of the KDTree which holds a list of facets."""

    def __init__(self, aabb: AABB, facets: Facet) -> None:
        self.aabb = aabb
        self.facets = facets
        self.children: list[KDTreeNode] = []

    def is_leaf(self) -> bool:
        """Return true if the this node is a leaf node (i.e., it has no children)."""
        return len(self.children) == 0

    def can_branch(self, depth: int) -> bool:
        """Return true if this node can be split into two child nodes."""
        # TODO: Maybe use a more sophisticated cost function to evaluate whether we should branch.
        return len(self.facets) > 0 and depth < DEPTH_BOUND

    def splitting_plane(self, depth: int) -> Tuple[CoordinateAxes, float]:
        """Return a tuple with the splitting plane axis and value for the given depth."""
        # TODO: Maybe use a more sophisticated splitting plane generation like SAH.
        plane_axis = CoordinateAxes(depth % 3)
        plane_value = self.aabb.center[plane_axis]

        return plane_axis, plane_value

    def split_facets(
        self, plane_axis: CoordinateAxes, plane_value: float
    ) -> Tuple[list[Facet], list[Facet]]:
        """Split the node's list of facets into left and right lists based on splitting plane."""
        left, right = [], []

        for facet in self.facets:
            # Check minimum bound on left and maximum bound on right.
            # This covers both scenarios: facets belonging to one side only or both sides
            if facet.aabb.min[plane_axis] < plane_value:
                left.append(facet)
            if facet.aabb.max[plane_axis] > plane_value:
                right.append(facet)

        return left, right

    def candidate_children(
        self, plane_axis: CoordinateAxes, plane_value: float
    ) -> Tuple[Tuple[AABB, list[Facet]]]:
        """Get child AABBs and facet lists from a given splitting plane."""
        child_boxes = self.aabb.split(plane_axis, plane_value)
        child_facets = self.split_facets(plane_axis, plane_value)

        return zip(child_boxes, child_facets)

    def branch(self, depth: int = 0) -> None:
        """If possible, split the current KDTreeNode into two children nodes."""
        if not self.can_branch(depth):
            return

        splitting_plane = self.splitting_plane(depth)

        for child in self.candidate_children(*splitting_plane):
            node = KDTreeNode(*child)
            node.branch(depth + 1)

            self.children.append(node)

        # Interrior nodes to the KDTree should not have any facets (only leaf nodes should).
        # If we've gotten this far, this node is an interrior node
        self.facets = None

    def intersect(self, ray: Ray) -> Intersection:
        """Intersect ray with node and return closest found intersection.

        Return Intersection.Miss() for no intersections.
        """
        if not self.aabb.intersect(ray):
            return Intersection.Miss()

        if self.is_leaf():
            return ray.closest_intersection(self.facets)

        return ray.closest_intersection(self.children)


class KDTree:
    """A kd-tree acceleration structure for space partitioning objects."""

    def __init__(self, mesh: "Mesh") -> None:
        self.root = KDTreeNode(mesh.aabb, mesh.facets)

        self.construct()

    def construct(self) -> None:
        """Construct the kd-tree."""
        self.root.branch()

    def intersect(self, ray: Ray) -> Intersection:
        """Return the intersection between the ray and the objects in the kd-tree."""
        return self.root.intersect(ray)

    def update(self, mesh: "Mesh") -> None:
        """Overwrite the kd-tree with the provided mesh."""
        # The full tree needs to be reconstructed when the mesh is changed so just re-initialize
        self.__init__(mesh)
