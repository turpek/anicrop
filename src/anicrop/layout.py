from __future__ import annotations
from anicrop.image import calculate_content_rect
from anicrop.transform import (
    calculate_new_rect,
    calculate_region_rect,
    mat_global,
    mat_inverse,
    transform_vector,
)
from anicrop import layer
from anicrop.spatial import Region
from anicrop.edit_layer import CropEditLayer
from collections.abc import Iterator
from functools import reduce
from operator import or_
from typing import Any, Protocol, Sequence, runtime_checkable, TYPE_CHECKING

from anicrop.canvas import Canvas
from anicrop.container import BaseLayer, Container, GroupLayer
from anicrop.geometry import LayerGeometry, FitGeometry, FitGroupGeometry

if TYPE_CHECKING:
    from anicrop.layer import Layer


@runtime_checkable
class LayoutStrategy(Protocol):
    """Protocolo estrutural para estratégias de layout de elementos."""

    @classmethod
    def fit(cls, target: Any, ref_region: Region) -> bool:
        ...

    @classmethod
    def align(
        cls,
        target: Any,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ...

    @classmethod
    def resize_bounds(
        cls,
        target: Any,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ...

    @classmethod
    def fit_content(cls, target: Any, *args: Any, **kwargs: Any) -> bool:
        ...


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


def _compute_layer_local_roi(target: Layer) -> Region | None:
    if not target._edits:
        return None

    accum_roi: Region | None = None

    for edit in target._edits:
        if not edit.visible:
            continue

        edit_roi = calculate_content_rect(edit.image) + edit.region.top_left

        if isinstance(edit, CropEditLayer):
            if accum_roi is None:
                accum_roi = edit_roi
            elif accum_roi.overlaps(edit_roi):
                accum_roi = accum_roi & edit_roi
            else:
                accum_roi = None
        else:
            if accum_roi is None:
                accum_roi = edit_roi
            else:
                accum_roi = accum_roi | edit_roi

    return accum_roi


def content_region(target: Layer) -> Region | None:
    """Calcula a ROI de conteúdo no espaço de coordenadas do Layer (somada com base.region.top_left)."""
    local_roi = _compute_layer_local_roi(target)
    if local_roi is None:
        return None
    return local_roi + target.base.region.top_left


def global_content_region(
    container: Layer | Container | Sequence[Layer],
) -> Region | None:
    """Calcula a Bounding Box de conteúdo de todos os elementos projetada no Espaço Global."""
    def _extract(item: layer.Layer | Container | Sequence[layer.Layer]) -> Iterator[Region]:
        if isinstance(item, layer.Layer):
            local_roi = _compute_layer_local_roi(item)
            if local_roi is None:
                return
            rect = calculate_region_rect(item.matrix, local_roi)
            yield Region.from_rect(*rect)
        else:
            for child in item:
                yield from _extract(child)

    regions = filter(None, _extract(container))
    try:
        first_region = next(regions)
    except StopIteration:
        return None

    return reduce(or_, regions, first_region)


def _resolve_target_fit_region(target: Layer, global_ref: Region) -> Region:
    """
    Calcula a região de enquadramento da camada no Canvas,
    compensando a translação intrínseca induzida pela transformação da camada.
    """
    (drift_x, drift_y, *_) = calculate_new_rect(target.transform.matrix, global_ref.size)
    return global_ref - (drift_x, drift_y)


def _resolve_target_content_region(
    target: Layer,
    global_roi: Region,
    ref_size: tuple[int, int],
) -> Region:
    """Calcula a região da moldura da camada no espaço do pai compensando o drift de rotação."""
    parent_roi_rect = calculate_region_rect(
        mat_inverse(target.parent.matrix),
        global_roi,
    )
    (drift_x, drift_y, *_) = calculate_new_rect(target.transform.matrix, ref_size)
    parent_x = parent_roi_rect[0] - drift_x
    parent_y = parent_roi_rect[1] - drift_y
    return Region.from_rect(parent_x, parent_y, *ref_size)


class LayerLayoutStrategy:

    @classmethod
    def fit(cls, target: Layer, ref_region: Region) -> bool:
        if target.global_region == ref_region:
            return False

        target_fit_region = _resolve_target_fit_region(target, ref_region)
        fit_strategy = FitGeometry(target, target_fit_region)
        target.frame = fit_strategy
        return True

    @classmethod
    def align(
        cls,
        target: Layer,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        new_global_region = target.global_region.align(ref_region, anchor_x, anchor_y)
        if target.global_region == new_global_region:
            return False

        dx, dy = transform_vector(
            mat_inverse(target.parent.matrix), target.global_region, new_global_region
        )
        target.region += (dx, dy)
        return True

    @classmethod
    def resize_bounds(
        cls,
        target: Layer,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        aligned_ref = ref_region.align(target.global_region, anchor_x, anchor_y)
        return cls.fit(target, aligned_ref)

    @classmethod
    def fit_content(cls, target: Layer, *args, **kwargs) -> bool:
        for edit in target._edits:
            if type(edit) is CropEditLayer:
                edit.visible = False

        global_roi = global_content_region(target)
        if global_roi is None or target.global_region == global_roi:
            return False

        local_roi = _compute_layer_local_roi(target)
        if local_roi is None:
            return False

        target_region = _resolve_target_content_region(target, global_roi, local_roi.size)
        target.frame = LayerGeometry(target, target_region)
        return True


class CanvasLayoutStrategy:

    @classmethod
    def fit(cls, target: Canvas, ref_region: Region) -> bool:
        if not ref_region.overlaps(target.region) or target.region == ref_region:
            return False

        target.region = ref_region
        return True

    @classmethod
    def align(
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
    def resize_bounds(
        cls,
        target: Canvas,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = ref_region.align(target.region, anchor_x, anchor_y)
        return cls.fit(target, ref_region)

    @classmethod
    def fit_content(
        cls,
        target: Canvas,
        container: Container | Sequence[Layer],
    ) -> bool:
        new_region = global_content_region(container)
        if new_region is None or new_region == target.region:
            return False

        target.region = new_region
        return True


class GroupLayoutStrategy:

    @classmethod
    def fit(cls, target: GroupLayer, ref_region: Region) -> bool:
        if target.global_region == ref_region:
            return False
        fit_strategy = FitGroupGeometry(target, ref_region)
        target.frame = fit_strategy
        return True

    @classmethod
    def align(
        cls,
        target: GroupLayer,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        new_global_region = target.global_region.align(ref_region, anchor_x, anchor_y)
        if target.global_region == new_global_region:
            return False

        dx, dy = transform_vector(
            mat_inverse(target.parent.matrix), target.global_region, new_global_region
        )
        target.transform.translate(dx, dy)
        return True

    @classmethod
    def resize_bounds(
        cls,
        target: GroupLayer,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        aligned_ref = ref_region.align(target.global_region, anchor_x, anchor_y)
        return cls.fit(target, aligned_ref)

    @classmethod
    def fit_content(cls, target: GroupLayer, *args, **kwargs) -> bool:
        global_roi = global_content_region(target)
        if global_roi is None or target.global_region == global_roi:
            return False
        return cls.fit(target, global_roi)


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
        return strategy_class.fit(target, ref_region)

    def align(
        self,
        target: Canvas | BaseLayer,
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = resolve_region(ref)
        strategy_class = self._resolve_strategy(target)
        return strategy_class.align(target, ref_region, anchor_x, anchor_y)

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
        return strategy_class.resize_bounds(target, ref_region, anchor_x, anchor_y)

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
        return strategy_class.fit_content(target, container=container)
