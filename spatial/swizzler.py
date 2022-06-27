# This class is meant to be inherited from so the lack of public methods is normal.
# pylint: disable=too-few-public-methods

from typing import Any, List


class Swizzler:
    """A base class for composing component parameters at runtime.

    For example, a Vector3 v with three components (v.x, v.y, v.z) can be accessed arbitrarily:
        v.xyx returns [v.x, v.y, v.x]

    See Vector3 and Quaternion.
    """

    def __getattr__(self, name: str) -> List[Any]:
        """Return a list of values of the composed parameters.

        All parameters are considered to be a single letter long.

        This function raises an AttributeError if the parameter does not exist.
        """

        def allow(char: str) -> Any:
            """Return the value of the provided parameter."""
            if char not in self.__slots__:
                raise AttributeError

            return getattr(self, char)

        return list(map(allow, name))
