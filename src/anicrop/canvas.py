from __future__ import annotations
from typing import Any, Self, Sequence, TYPE_CHECKING
from anicrop.spatial import Region
from anicrop.container import global_content_region

if TYPE_CHECKING:
    from anicrop.layer import Layer
    from anicrop.container import Container


class CanvasLayoutStrategy:
    """Estratégia de layout para a moldura do Canvas."""

    def __init__(self, target: Canvas) -> None:
        self.target = target

    def fit(self, ref: tuple[int, int, int, int] | Region | Canvas | Any) -> bool:
        return self._fit(self.target, self._resolve_region(ref))

    def align(
        self,
        ref: tuple[int, int, int, int] | Region | Canvas | Any,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        return self._align(self.target, self._resolve_region(ref), anchor_x, anchor_y)

    def resize_bounds(
        self,
        new_width: int,
        new_height: int,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = Region.from_size(new_width, new_height)
        return self._resize_bounds(self.target, ref_region, anchor_x, anchor_y)

    def fit_content(
        self,
        container: Container | Sequence[Layer],
    ) -> bool:
        return self._fit_content(self.target, container=container)

    @classmethod
    def _fit(cls, target: Canvas, ref_region: Region) -> bool:
        if not ref_region.overlaps(target.region) or target.region == ref_region:
            return False

        target.region = ref_region
        return True

    @classmethod
    def _align(
        cls,
        target: Canvas,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        new_region = target.region.align(ref_region, anchor_x, anchor_y)
        if target.region == new_region:
            return False
        target.region = new_region
        return True

    @classmethod
    def _resize_bounds(
        cls,
        target: Canvas,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = ref_region.align(target.region, anchor_x, anchor_y)
        return cls._fit(target, ref_region)

    @classmethod
    def _fit_content(
        cls,
        target: Canvas,
        container: Container | Sequence[Layer],
    ) -> bool:
        new_region = global_content_region(container)
        if new_region is None or new_region == target.region:
            return False

        target.region = new_region
        return True

    @staticmethod
    def _resolve_region(ref: Any) -> Region:
        if isinstance(ref, tuple):
            return Region.from_rect(*ref)
        elif hasattr(ref, "global_region"):
            return ref.global_region
        elif hasattr(ref, "region"):
            return ref.region
        return ref


class Canvas:
    def __init__(self, region: Region, bg_color: tuple[int, int, int, int] = (0, 0, 0, 0)):
        self._region = region
        self.bg_color = bg_color
        self._layout = CanvasLayoutStrategy(self)

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
