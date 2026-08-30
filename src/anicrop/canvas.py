from __future__ import annotations

from typing import Self

from anicrop.interfaces.canvas import AbstractCanvas
from anicrop.layout import CanvasLayoutStrategy
from anicrop.spatial import Point, Region


class Canvas(AbstractCanvas):
    def __init__(
        self,
        region: Region,
        bg_color: tuple[int, ...] | None = None,
    ):
        self._region = region
        self.bg_color = bg_color if bg_color is not None else (0, 0, 0, 0)
        self._layout = CanvasLayoutStrategy(self)

    @classmethod
    def from_size(
        cls,
        width: float,
        height: float,
        bg_color: tuple[int, ...] | None = None,
    ) -> Self:
        return cls(Region.from_size(width, height), bg_color=bg_color)

    @classmethod
    def from_rect(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
        bg_color: tuple[int, ...] | None = None,
    ) -> Self:
        return cls(Region.from_rect(x, y, width, height), bg_color=bg_color)

    @property
    def size(self) -> Point:
        return self._region.size

    @property
    def width(self) -> float:
        return self._region.width

    @property
    def height(self) -> float:
        return self._region.height

    @property
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, value) -> None:
        if not isinstance(value, Region):
            raise TypeError(f"Expected Region, got {type(value).__name__}")
        self._region = value

    @property
    def layout(self) -> CanvasLayoutStrategy:
        """Estratégia de layout para a moldura do Canvas."""
        return self._layout
