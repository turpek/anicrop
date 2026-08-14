from typing import Self

from anicrop.spatial import Region


class Canvas:
    def __init__(self, region: Region, bg_color: tuple[int, int, int, int] = (0, 0, 0, 0)):
        self._region = region
        self.bg_color = bg_color

    @classmethod
    def from_size(
        cls,
        width: int,
        height: int,
        bg_color: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Self:

        return cls(Region.from_size(width, height), bg_color=bg_color)

    @classmethod
    def from_rect(
        cls,
        x: int,
        y: int,
        width: int,
        height: int,
        bg_color: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Self:

        return cls(Region.from_rect(x, y, width, height), bg_color=bg_color)

    @property
    def size(self) -> tuple[int, int]:
        return self._region.size

    @property
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, value) -> None:
        if not isinstance(value, Region):
            raise TypeError(f"Expected Region, got {type(value).__name__}")
        self._region = value
