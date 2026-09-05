from __future__ import annotations

import math
from collections.abc import Iterator
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, Any, Sequence

from anicrop.edit_layer import CropEditLayer
from anicrop.geometry import FitGeometry, FitGroupGeometry, LayerGeometry
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
    from anicrop.viewport import Viewport

LayoutRef = tuple[int, int, int, int] | Region | AbstractCanvas | AbstractBaseLayer


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

    def _extract(
        item: AbstractBaseLayer | AbstractContainer | Sequence[AbstractBaseLayer],
    ) -> Iterator[Region]:
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
    ref: LayoutRef,
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
    ref_size: tuple[float, float],
) -> Region:
    """Calcula a região da moldura da camada no espaço do pai compensando o drift de rotação."""
    parent_roi_rect = calculate_region_rect(
        mat_inverse(target.parent.matrix),
        global_roi,
    )
    (drift_x, drift_y, *_) = calculate_new_rect(target.transform.matrix, ref_size)
    parent_x = parent_roi_rect[0] - drift_x
    parent_y = parent_roi_rect[1] - drift_y
    return Region.from_rect(parent_x, parent_y, ref_size[0], ref_size[1])


class LayerLayoutStrategy(LayoutStrategy):
    """Estratégia de layout para a moldura de uma camada individual (Layer)."""

    def __init__(self, target: Layer) -> None:
        self.target = target

    def fit(
        self,
        ref: LayoutRef,
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
        ref: LayoutRef,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = resolve_region(ref)
        new_global_region = self.target.global_region.align(
            ref_region, anchor_x, anchor_y
        )
        if self.target.global_region == new_global_region:
            return False

        dx, dy = transform_vector(
            mat_inverse(self.target.parent.matrix),
            self.target.global_region,
            new_global_region,
        )
        self.target.region += (dx, dy)
        return True

    def resize_bounds(
        self,
        new_width: float,
        new_height: float,
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

        target_region = _resolve_target_content_region(
            self.target, global_roi, local_roi.size
        )
        self.target.frame = LayerGeometry(self.target, target_region)
        return True


class GroupLayoutStrategy(LayoutStrategy):
    """Estratégia de layout para a moldura do GroupLayer."""

    def __init__(self, target: GroupLayer) -> None:
        self.target = target

    def fit(
        self,
        ref: LayoutRef,
    ) -> bool:
        ref_region = resolve_region(ref)
        if self.target.global_region == ref_region:
            return False
        fit_strategy = FitGroupGeometry(self.target, ref_region)
        self.target.frame = fit_strategy
        return True

    def align(
        self,
        ref: LayoutRef,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = resolve_region(ref)
        new_global_region = self.target.global_region.align(
            ref_region, anchor_x, anchor_y
        )
        if self.target.global_region == new_global_region:
            return False

        dx, dy = transform_vector(
            mat_inverse(self.target.parent.matrix),
            self.target.global_region,
            new_global_region,
        )
        self.target.transform.translate(dx, dy)
        return True

    def resize_bounds(
        self,
        new_width: float,
        new_height: float,
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

    def fit(
        self,
        ref: LayoutRef,
    ) -> bool:
        ref_region = resolve_region(ref)
        t_region = self.target.region
        if not ref_region.overlaps(t_region) or t_region == ref_region:
            return False

        self.target.region = ref_region
        return True

    def align(
        self,
        ref: LayoutRef,
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
        new_width: float,
        new_height: float,
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
            raise ValueError(
                "Canvas.fit_content requires a container or sequence of layers."
            )

        roi = global_content_region(container)
        if (
            roi is None
            or not roi.overlaps(self.target.region)
            or self.target.region == roi
        ):
            return False

        self.target.region = roi
        return True


class ViewportLayoutStrategy(LayoutStrategy):
    """Estratégia de layout para a moldura da Viewport (Câmera)."""

    def __init__(self, target: Viewport) -> None:
        self.target = target

    def fit(
        self,
        ref: LayoutRef,
    ) -> bool:
        ref_region = resolve_region(ref)
        w_ref, h_ref = ref_region.size
        if w_ref <= 0 or h_ref <= 0:
            return False

        w_view, h_view = self.target.size
        s = min(w_view / w_ref, h_view / h_ref)
        fit_sx = self.target._fit.sx if self.target._fit.sx > 0 else 1.0
        new_zoom = s / fit_sx

        ref_center_x = ref_region.x.start + w_ref / 2.0
        ref_center_y = ref_region.y.start + h_ref / 2.0
        canvas_center_x = self.target.canvas_size[0] / 2.0
        canvas_center_y = self.target.canvas_size[1] / 2.0
        pan_x = self.target._fit.sx * (ref_center_x - canvas_center_x)
        pan_y = self.target._fit.sy * (ref_center_y - canvas_center_y)
        new_region = Region.from_rect(pan_x, pan_y, w_view, h_view)

        if math.isclose(self.target.zoom, new_zoom) and self.target.region == new_region:
            return False

        self.target.zoom = new_zoom
        self.target.region = new_region
        return True

    def align(
        self,
        ref: LayoutRef,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = resolve_region(ref)
        w_view, h_view = self.target.size
        s_total_x = self.target.scale_factor
        s_total_y = self.target.scale.sy * self.target._fit.sy
        if s_total_x <= 0 or s_total_y <= 0:
            return False

        w_visible = w_view / s_total_x
        h_visible = h_view / s_total_y

        vis_start_x = ref_region.x.start + (ref_region.width - w_visible) * anchor_x
        vis_start_y = ref_region.y.start + (ref_region.height - h_visible) * anchor_y
        vis_center_x = vis_start_x + w_visible / 2.0
        vis_center_y = vis_start_y + h_visible / 2.0

        canvas_center_x = self.target.canvas_size[0] / 2.0
        canvas_center_y = self.target.canvas_size[1] / 2.0
        pan_x = self.target._fit.sx * (vis_center_x - canvas_center_x)
        pan_y = self.target._fit.sy * (vis_center_y - canvas_center_y)
        new_region = Region.from_rect(pan_x, pan_y, w_view, h_view)

        if self.target.region == new_region:
            return False

        self.target.region = new_region
        return True

    def resize_bounds(
        self,
        new_width: float,
        new_height: float,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        old_w, old_h = self.target.size
        if new_width == old_w and new_height == old_h:
            return False

        scale_x = self.target.scale.sx if self.target.scale.sx > 0 else 1.0
        scale_y = self.target.scale.sy if self.target.scale.sy > 0 else 1.0

        delta_pan_x = (anchor_x - 0.5) * (new_width - old_w) / scale_x
        delta_pan_y = (anchor_y - 0.5) * (new_height - old_h) / scale_y

        cur_pan_x, cur_pan_y = self.target.region.top_left
        new_pan_x = cur_pan_x + delta_pan_x
        new_pan_y = cur_pan_y + delta_pan_y

        self.target.region = Region.from_rect(
            new_pan_x, new_pan_y, new_width, new_height
        )
        return True

    def fit_content(
        self,
        container: AbstractContainer | Sequence[AbstractBaseLayer] | None = None,
    ) -> bool:
        if container is None:
            return self.fit(self.target._canvas)

        roi = global_content_region(container)
        if roi is None or not roi.overlaps(self.target._canvas.region):
            return False

        effective_roi = roi & self.target._canvas.region
        return self.fit(effective_roi)


class Layout:
    """Motor central de operações de layout e enquadramento espacial."""

    def fit(
        self,
        target: AbstractCanvas | AbstractBaseLayer | Viewport,
        ref: LayoutRef,
    ) -> bool:
        return target.layout.fit(ref)

    def align(
        self,
        target: AbstractCanvas | AbstractBaseLayer | Viewport,
        ref: LayoutRef,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        return target.layout.align(ref, anchor_x, anchor_y)

    def resize_bounds(
        self,
        target: AbstractCanvas | AbstractBaseLayer | Viewport,
        new_width: float,
        new_height: float,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        return target.layout.resize_bounds(new_width, new_height, anchor_x, anchor_y)

    def fit_content(
        self,
        target: AbstractCanvas | AbstractBaseLayer | Viewport,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Ajusta a moldura do alvo aos limites do conteúdo visível."""
        return target.layout.fit_content(*args, **kwargs)
