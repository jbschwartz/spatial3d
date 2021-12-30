"""Euler Angles library."""

from .axes import Axes
from .order import Order


def angles(
    quaternion: "Quaternion", axes: Axes = Axes.ZYZ, order: Order = Order.INTRINSIC
) -> list[list[float]]:
    """Return the requested Euler angles and type from the provided quaternion."""
    if not isinstance(axes, Axes) or not isinstance(order, Order):
        raise TypeError("Unknown type passed to function")

    # Take advantage of extrinsic being the reverse axes order intrinsic solution reversed
    if order == Order.EXTRINSIC:
        axes = axes.reverse()

    solutions = axes.convert(quaternion)

    if order == Order.EXTRINSIC:
        solutions = [angles[::-1] for angles in solutions]

    return solutions
