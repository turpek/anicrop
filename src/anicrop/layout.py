from __future__ import annotations
from typing import Any, Sequence, TYPE_CHECKING

from functools import reduce
from operator import or_
from collections.abc import Iterator

from anicrop.edit_layer import CropEditLayer
from anicrop.geometry import LayerGeometry, FitGeometry, FitGroupGeometry
from anicrop.image import calculate_content_rect
from anicrop.interfaces.canvas import AbstractCanvas
from anicrop.interfaces.container import AbstractContainer
from anicrop.interfaces.layer import AbstractBaseLayer, AbstractLayer
from anicrop.interfaces.layout import LayoutStrategy
from anicrop.spatial import Region
from anicrop.transform import (
    calculate_new_rect,
    calculate_region_rect,
    mat_inverse,
    transform_vector,
)

if TYPE_CHECKING:
    from anicrop.canvas import Canvas
    from anicrop.container import Container, GroupLayer
    from anicrop.layer import Layer


def _compute_layer_local_roi(target: AbstractLayer) -> Region | None:
    """Calcula a bounding box local do conteúdo do layer."""
    if not target.edits:
        return None

    accum_roi: Region | None = None

    for edit in target.edits:
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


def content_region(target: AbstractLayer) -> Region | None:
    """Calcula a ROI de conteúdo no espaço de coordenadas do Layer (somada com base.region.top_left)."""
    local_roi = _compute_layer_local_roi(target)
    if local_roi is None:
        return None
    return local_roi + target.base.region.top_left


def global_content_region(
    container: AbstractBaseLayer | AbstractContainer | Sequence[AbstractBaseLayer],
) -> Region | None:
    """Calcula a Bounding Box de conteúdo de todos os elementos projetada no Espaço Global."""
    def _extract(item: AbstractBaseLayer | AbstractContainer | Sequence[AbstractBaseLayer]) -> Iterator[Region]:
        if isinstance(item, AbstractLayer):
            local_roi = _compute_layer_local_roi(item)
            if local_roi is None:
                return
            rect = calculate_region_rect(item.matrix, local_roi)
            yield Region.from_rect(*rect)
        else:
            for child in item:  # type: ignore[union-attr]
                yield from _extract(child)

    regions = filter(None, _extract(container))
    try:
        first_region = next(regions)
    except StopIteration:
        return None

    return reduce(or_, regions, first_region)


def resolve_region(
    ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
) -> Region:
    if isinstance(ref, tuple):
        return Region.from_rect(*ref)
    elif isinstance(ref, AbstractCanvas):
        return ref.region
    elif isinstance(ref, AbstractBaseLayer):
        return ref.global_region
    return ref


def _resolve_target_fit_region(target: AbstractLayer, global_ref: Region) -> Region:
    """
    Calcula a região de enquadramento da camada no Canvas,
    compensando a translação intrínseca induzida pela transformação da camada.
    """
    (drift_x, drift_y, *_) = calculate_new_rect(target.transform.matrix, global_ref.size)
    return global_ref - (drift_x, drift_y)


def _resolve_target_content_region(
    target: AbstractLayer,
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


class LayerLayoutStrategy(LayoutStrategy):
    """Estratégia de layout para a moldura de uma camada individual (Layer)."""

    def __init__(self, target: Layer) -> None:
        self.target = target

    def fit(
        self,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
    ) -> bool:
        return self._fit(self.target, resolve_region(ref))

    def align(
        self,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        return self._align(self.target, resolve_region(ref), anchor_x, anchor_y)

    def resize_bounds(
        self,
        new_width: int,
        new_height: int,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = Region.from_size(new_width, new_height)
        return self._resize_bounds(self.target, ref_region, anchor_x, anchor_y)

    def fit_content(self, *args: Any, **kwargs: Any) -> bool:
        return self._fit_content(self.target, *args, **kwargs)

    @classmethod
    def _fit(cls, target: Layer, ref_region: Region) -> bool:
        if target.global_region == ref_region:
            return False

        target_fit_region = _resolve_target_fit_region(target, ref_region)
        fit_strategy = FitGeometry(target, target_fit_region)
        target.frame = fit_strategy
        return True

    @classmethod
    def _align(
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
    def _resize_bounds(
        cls,
        target: Layer,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        aligned_ref = ref_region.align(target.global_region, anchor_x, anchor_y)
        return cls._fit(target, aligned_ref)

    @classmethod
    def _fit_content(cls, target: Layer, *args: Any, **kwargs: Any) -> bool:
        for edit in target.edits:
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


class GroupLayoutStrategy(LayoutStrategy):
    """Estratégia de layout para a moldura do GroupLayer."""

    def __init__(self, target: GroupLayer) -> None:
        self.target = target

    def fit(self, ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer) -> bool:
        return self._fit(self.target, resolve_region(ref))

    def align(
        self,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        return self._align(self.target, resolve_region(ref), anchor_x, anchor_y)

    def resize_bounds(
        self,
        new_width: int,
        new_height: int,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = Region.from_size(new_width, new_height)
        return self._resize_bounds(self.target, ref_region, anchor_x, anchor_y)

    def fit_content(self, *args: Any, **kwargs: Any) -> bool:
        return self._fit_content(self.target, *args, **kwargs)

    @classmethod
    def _fit(cls, target: GroupLayer, ref_region: Region) -> bool:
        if target.global_region == ref_region:
            return False
        fit_strategy = FitGroupGeometry(target, ref_region)
        target.frame = fit_strategy
        return True

    @classmethod
    def _align(
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
    def _resize_bounds(
        cls,
        target: GroupLayer,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        aligned_ref = ref_region.align(target.global_region, anchor_x, anchor_y)
        return cls._fit(target, aligned_ref)

    @classmethod
    def _fit_content(cls, target: GroupLayer, *args: Any, **kwargs: Any) -> bool:
        global_roi = global_content_region(target)
        if global_roi is None or target.global_region == global_roi:
            return False
        return cls._fit(target, global_roi)


class CanvasLayoutStrategy(LayoutStrategy):
    """Estratégia de layout para a moldura do Canvas."""

    def __init__(self, target: Canvas) -> None:
        self.target = target

    def fit(self, ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer) -> bool:
        return self._fit(self.target, resolve_region(ref))

    def align(
        self,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        return self._align(self.target, resolve_region(ref), anchor_x, anchor_y)

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
        aligned_ref = ref_region.align(target.region, anchor_x, anchor_y)
        return cls._fit(target, aligned_ref)

    @classmethod
    def _fit_content(
        cls,
        target: Canvas,
        container: Container | Sequence[Layer] | None = None,
    ) -> bool:
        if container is None:
            raise ValueError("Canvas.fit_content requires a container or sequence of layers.")

        roi = global_content_region(container)
        if roi is None or not roi.overlaps(target.region) or target.region == roi:
            return False

        target.region = roi
        return True


class Layout:

    def _resolve_strategy(self, target: Any) -> type[LayoutStrategy]:
        if hasattr(target, "layout"):
            return type(target.layout)
        raise TypeError(f"Unsupported target type: {type(target).__name__}")

    def fit(
        self,
        target: Canvas | AbstractBaseLayer,
        ref: tuple[int, int, int, int] | Region | Canvas | AbstractBaseLayer,
    ) -> bool:
        ref_region = resolve_region(ref)
        strategy_class = self._resolve_strategy(target)
        return strategy_class._fit(target, ref_region)

    def align(
        self,
        target: Canvas | AbstractBaseLayer,
        ref: tuple[int, int, int, int] | Region | Canvas | AbstractBaseLayer,
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
        """Fits the frame or boundary of the target to the visible content limits."""
        strategy_class = self._resolve_strategy(target)
        return strategy_class._fit_content(target, container=container)
