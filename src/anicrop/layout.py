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
        ref_region = resolve_region(ref)
        if self.target.global_region == ref_region:
            return False

        target_fit_region = _resolve_target_fit_region(self.target, ref_region)
        fit_strategy = FitGeometry(self.target, target_fit_region)
        self.target.frame = fit_strategy
        return True

    def align(
        self,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = resolve_region(ref)
        new_global_region = self.target.global_region.align(ref_region, anchor_x, anchor_y)
        if self.target.global_region == new_global_region:
            return False

        dx, dy = transform_vector(
            mat_inverse(self.target.parent.matrix), self.target.global_region, new_global_region
        )
        self.target.region += (dx, dy)
        return True

    def resize_bounds(
        self,
        new_width: int,
        new_height: int,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = Region.from_size(new_width, new_height)
        aligned_ref = ref_region.align(self.target.global_region, anchor_x, anchor_y)
        return self.fit(aligned_ref)

    def fit_content(self, *args: Any, **kwargs: Any) -> bool:
        for edit in self.target.edits:
            if type(edit) is CropEditLayer:
                edit.visible = False

        global_roi = global_content_region(self.target)
        if global_roi is None or self.target.global_region == global_roi:
            return False

        local_roi = _compute_layer_local_roi(self.target)
        if local_roi is None:
            return False

        target_region = _resolve_target_content_region(self.target, global_roi, local_roi.size)
        self.target.frame = LayerGeometry(self.target, target_region)
        return True


class GroupLayoutStrategy(LayoutStrategy):
    """Estratégia de layout para a moldura do GroupLayer."""

    def __init__(self, target: GroupLayer) -> None:
        self.target = target

    def fit(self, ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer) -> bool:
        ref_region = resolve_region(ref)
        if self.target.global_region == ref_region:
            return False
        fit_strategy = FitGroupGeometry(self.target, ref_region)
        self.target.frame = fit_strategy
        return True

    def align(
        self,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = resolve_region(ref)
        new_global_region = self.target.global_region.align(ref_region, anchor_x, anchor_y)
        if self.target.global_region == new_global_region:
            return False

        dx, dy = transform_vector(
            mat_inverse(self.target.parent.matrix), self.target.global_region, new_global_region
        )
        self.target.transform.translate(dx, dy)
        return True

    def resize_bounds(
        self,
        new_width: int,
        new_height: int,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = Region.from_size(new_width, new_height)
        aligned_ref = ref_region.align(self.target.global_region, anchor_x, anchor_y)
        return self.fit(aligned_ref)

    def fit_content(self, *args: Any, **kwargs: Any) -> bool:
        global_roi = global_content_region(self.target)
        if global_roi is None or self.target.global_region == global_roi:
            return False
        return self.fit(global_roi)


class CanvasLayoutStrategy(LayoutStrategy):
    """Estratégia de layout para a moldura do Canvas."""

    def __init__(self, target: Canvas) -> None:
        self.target = target

    def fit(self, ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer) -> bool:
        ref_region = resolve_region(ref)
        if not ref_region.overlaps(self.target.region) or self.target.region == ref_region:
            return False

        self.target.region = ref_region
        return True

    def align(
        self,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = resolve_region(ref)
        new_region = self.target.region.align(ref_region, anchor_x, anchor_y)
        if self.target.region == new_region:
            return False
        self.target.region = new_region
        return True

    def resize_bounds(
        self,
        new_width: int,
        new_height: int,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = Region.from_size(new_width, new_height)
        aligned_ref = ref_region.align(self.target.region, anchor_x, anchor_y)
        return self.fit(aligned_ref)

    def fit_content(
        self,
        container: Container | Sequence[Layer] | None = None,
    ) -> bool:
        if container is None:
            raise ValueError("Canvas.fit_content requires a container or sequence of layers.")

        roi = global_content_region(container)
        if roi is None or not roi.overlaps(self.target.region) or self.target.region == roi:
            return False

        self.target.region = roi
        return True


class Layout:
    """Motor central de operações de layout e enquadramento espacial."""

    def fit(
        self,
        target: AbstractCanvas | AbstractBaseLayer,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
    ) -> bool:
        return target.layout.fit(ref)

    def align(
        self,
        target: AbstractCanvas | AbstractBaseLayer,
        ref: tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        return target.layout.align(ref, anchor_x, anchor_y)

    def resize_bounds(
        self,
        target: AbstractCanvas | AbstractBaseLayer,
        new_width: int,
        new_height: int,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        return target.layout.resize_bounds(new_width, new_height, anchor_x, anchor_y)

    def fit_content(
        self,
        target: AbstractCanvas | AbstractBaseLayer,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Ajusta a moldura do alvo aos limites do conteúdo visível."""
        return target.layout.fit_content(*args, **kwargs)
