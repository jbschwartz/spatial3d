import math
from functools import cached_property
from typing import Iterable, Optional, Tuple, Union

from .coordinate_axes import CoordinateAxes
from .vector3 import Vector3

Boundable = Union[Vector3, "AABB", Iterable[Vector3], Iterable["AABB"]]


class AABB:
    """An axis-aligned bounding box."""

    def __init__(self, objects: Optional[Boundable] = None) -> None:
        """Construct a bounding box that bounds a list of points or other bounding boxes."""
        self.min = Vector3(math.inf, math.inf, math.inf)
        self.max = -Vector3(math.inf, math.inf, math.inf)

        if objects is not None:
            self.expand(objects)

    def __str__(self) -> str:
        """Return the string representation of the minimum and maximum corner points."""
        return f"Min: {self.min}, Max: {self.max}"

    @cached_property
    def center(self) -> Vector3:
        """Return the center point of the bounding box."""
        # It's enough to check that one component is infinite to determine that all of them are
        # (assuming that the AABB is only manipulated by calls to AABB.expand)
        if math.isinf(self.min[0]):
            return Vector3(0, 0, 0)

        return self.min + (self.size / 2)

    @cached_property
    def corners(self) -> list[Vector3]:
        """Return all eight corner points of the bounding box."""
        size = self.size
        x = Vector3(x=size.x)
        y = Vector3(y=size.y)

        return [
            self.max,
            self.max - x,
            self.max - x - y,
            self.max - y,
            self.min,
            self.min + x,
            self.min + x + y,
            self.min + y,
        ]

    @property
    def size(self) -> Vector3:
        """Return the bounding box size for each coordinate axis."""
        if math.isinf(self.min[0]):
            return self.min

        return self.max - self.min

    def axis_extents(self, axis: CoordinateAxes) -> Tuple[float, float]:
        """Return the minimum and maximum values for the given axis."""
        return (self.min[axis], self.max[axis])

    def contains(self, point: Vector3) -> bool:
        """Return True if the bounding box contains the point."""
        return all(low <= value <= high for low, value, high in zip(self.min, point, self.max))

    def expand(self, objects: Boundable) -> None:
        """Expand the bounding box to include the passed points and bounding boxes."""
        # If the passed parameter looks iterable, try to break it up recursively
        if isinstance(objects, (list, tuple)):
            for obj in objects:
                self.expand(obj)
        elif isinstance(objects, AABB):
            # Expand the bounding box with the corner points
            self.expand([objects.min, objects.max])
        elif isinstance(objects, Vector3):
            for coordinate, value in enumerate(objects):
                self.min[coordinate] = min(value, self.min[coordinate])
                self.max[coordinate] = max(value, self.max[coordinate])

            # Invalidate the cached properties.
            if self.corners is not None:
                del self.corners

            if self.center is not None:
                del self.center
        else:
            raise TypeError(f"Unexpected type passed to AABB.expand: {type(objects)}")

    def intersect(self, ray, min_t: float = 0, max_t: float = math.inf) -> bool:
        """Return True if the provided ray intersects the bounding box."""
        t_intersection = [min_t, max_t]

        # Check bounding slab intersections per component (x, y, z)
        for minimum, maximum, origin, direction in zip(
            self.min, self.max, ray.origin, ray.direction
        ):
            try:
                inv_direction = 1 / direction
            except ZeroDivisionError:
                inv_direction = math.inf

            t_min = (minimum - origin) * inv_direction
            t_max = (maximum - origin) * inv_direction

            # Swap if reordering is necessary
            if t_min > t_max:
                t_min, t_max = t_max, t_min

            if t_min > t_intersection[0]:
                t_intersection[0] = t_min
            if t_max < t_intersection[1]:
                t_intersection[1] = t_max

            if t_intersection[0] > t_intersection[1]:
                return False

        return True

    def sphere_radius(self) -> float:
        """Return the radius of a bounding sphere which contains the bounding box."""
        return max([(self.center - corner).length() for corner in self.corners])

    def split(self, axis: CoordinateAxes, value: float) -> Tuple["AABB", "AABB"]:
        """Return two new child bounding boxes from splitting the existing bounding box.

        Raises ValueError if the split value is outside the bounds of the bounding box.

        Note that this function does not alter the existing bounding box.
        """
        if self.min[axis] >= value or self.max[axis] <= value:
            raise ValueError(
                f"The split plane ({value}) is outside of the bounding box: "
                "({self.min[axis]} , {self.max[axis]})"
            )

        left_max = Vector3(*self.max)
        left_max[axis] = value

        left = AABB([self.min, left_max])

        right_min = Vector3(*self.min)
        right_min[axis] = value

        right = AABB([right_min, self.max])

        return left, right
