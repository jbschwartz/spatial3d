import enum


class Order(enum.Enum):
    """The two types of Euler angles."""

    INTRINSIC = enum.auto()
    EXTRINSIC = enum.auto()
