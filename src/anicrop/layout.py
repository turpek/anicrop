from __future__ import annotations
from anicrop import layer
from anicrop.spatial import Region
from typing import Any, Sequence

from anicrop.canvas import Canvas, CanvasLayoutStrategy
from anicrop.container import (
    BaseLayer,
    Container,
    GroupLayer,
    GroupLayoutStrategy,
)
from anicrop.interfaces.layout import LayoutStrategy
from anicrop.layer import Layer, LayerLayoutStrategy


def resolve_region(
    ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
) -> Region:
    if isinstance(ref, tuple):
        return Region.from_rect(*ref)
    elif isinstance(ref, Canvas):
        return ref.region
    elif isinstance(ref, BaseLayer):
        return ref.global_region
    return ref


class Layout:

    def _resolve_strategy(self, target: Any) -> type[LayoutStrategy]:
        if isinstance(target, GroupLayer):
            return GroupLayoutStrategy
        if isinstance(target, layer.Layer):
            return LayerLayoutStrategy
        if isinstance(target, Canvas):
            return CanvasLayoutStrategy
        raise TypeError(f"Unsupported target type: {type(target).__name__}")

    def fit(
        self,
        target: Canvas | BaseLayer,
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
    ) -> bool:
        ref_region = resolve_region(ref)
        strategy_class = self._resolve_strategy(target)
        return strategy_class._fit(target, ref_region)

    def align(
        self,
        target: Canvas | BaseLayer,
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = resolve_region(ref)
        strategy_class = self._resolve_strategy(target)
        return strategy_class._align(target, ref_region, anchor_x, anchor_y)

    def resize_bounds(
        self,
        target: Layer | Canvas | GroupLayer,
        new_width: int,
        new_height: int,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = Region.from_size(new_width, new_height)
        strategy_class = self._resolve_strategy(target)
        return strategy_class._resize_bounds(target, ref_region, anchor_x, anchor_y)

    def fit_content(
        self,
        target: Layer | GroupLayer | Canvas,
        container: Container | Sequence[Layer] | None = None,
    ) -> bool:
        """Fits the frame or boundary of the target to the visible content limits.

        Target behavior:
        - Layer: Fits the layer frame to its visible pixels and active edits.
        - GroupLayer: Frames the group bounding box to the combined content of its children.
        - Canvas: Adjusts the canvas dimensions to the bounding box of the layers passed
          in `container` (required when `target` is Canvas).

        Args:
            target: The Layer, GroupLayer, or Canvas to fit.
            container: Reference collection or sequence of layers (required for Canvas).

        Returns:
            True if the geometry or dimensions changed, False otherwise.
        """
        strategy_class = self._resolve_strategy(target)
        return strategy_class._fit_content(target, container=container)
