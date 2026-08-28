from __future__ import annotations
from typing import Self
from anicrop.enums import ImageFormat
from anicrop.interfaces.canvas import AbstractCanvas
from anicrop.spatial import Region
from anicrop.layout import CanvasLayoutStrategy


class Canvas(AbstractCanvas):
    def __init__(
        self,
        region: Region,
        bg_color: tuple[int, ...] | None = None,
        format: ImageFormat = ImageFormat.RGBA,
    ):
        self._region = region
        self._format = format
        if bg_color is None:
            self.bg_color = (0,) * format.channels
        else:
            self.bg_color = bg_color
        self._layout = CanvasLayoutStrategy(self)

    @classmethod
    def from_size(
        cls,
        width: int,
        height: int,
        bg_color: tuple[int, ...] | None = None,
        format: ImageFormat = ImageFormat.RGBA,
    ) -> Self:
        return cls(Region.from_size(width, height), bg_color=bg_color, format=format)

    @classmethod
    def from_rect(
        cls,
        x: int,
        y: int,
        width: int,
        height: int,
        bg_color: tuple[int, ...] | None = None,
        format: ImageFormat = ImageFormat.RGBA,
    ) -> Self:
        return cls(Region.from_rect(x, y, width, height), bg_color=bg_color, format=format)

    @property
    def format(self) -> ImageFormat:
        return self._format

    @format.setter
    def format(self, value: ImageFormat) -> None:
        if not isinstance(value, ImageFormat):
            raise TypeError(f"Expected ImageFormat, got {type(value).__name__}")
        self._format = value
        if self.bg_color is None or len(self.bg_color) != value.channels:
            self.bg_color = (0,) * value.channels

    @property
    def size(self) -> tuple[int, int]:
        return self._region.size

    @property
    def width(self) -> int:
        return self._region.width

    @property
    def height(self) -> int:
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
