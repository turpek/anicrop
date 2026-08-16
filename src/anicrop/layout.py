from __future__ import annotations

from collections.abc import Iterator
from functools import reduce
from operator import or_
from typing import Any, Sequence

from anicrop.canvas import Canvas
from anicrop.container import BaseLayer, Container, GroupLayer
from anicrop.geometry import FitGeometry, FitGroupGeometry

from anicrop.image import calculate_content_rect
from anicrop.layer import Layer
from anicrop.spatial import Region
from anicrop.transform import (
    calculate_region_rect,
    mat_global,
    mat_inverse,
    transform_vector,
)


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


def content_region(target: Layer) -> Region | None:
    """Calcula a ROI de conteúdo no espaço de coordenadas do Layer (somada com base.region.top_left)."""
    if not target._edits:
        return None

    def roi(edit):
        return calculate_content_rect(edit.image) + edit.region.top_left

    content_roi = reduce(or_, [roi(e) for e in target._edits])
    return content_roi + target.base.region.top_left


def global_content_region(
    container: Layer | Container | Sequence[Layer],
) -> Region | None:
    """Calcula a Bounding Box de conteúdo de todos os elementos projetada no Espaço Global."""
    def _extract(item: Layer | Container | Sequence[Layer]) -> Iterator[Region]:
        if isinstance(item, Layer):
            if not item._edits:
                return

            def roi(edit):
                return calculate_content_rect(edit.image) + edit.region.top_left

            local_roi = reduce(or_, [roi(e) for e in item._edits])
            rect = calculate_region_rect(mat_global(item), local_roi)
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


class LayerLayoutStrategy:

    @classmethod
    def fit(cls, target: Layer, ref_region: Region) -> bool:
        if target.global_region == ref_region:
            return False
        fit_strategy = FitGeometry(target, ref_region)
        target.control.set_strategy(fit_strategy)
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
        content_roi = content_region(target)
        if content_roi is None:
            return False
        parent_mat = target.parent.matrix
        global_roi = Region.from_rect(*calculate_region_rect(parent_mat, content_roi))
        if target.global_region == global_roi:
            return False
        return cls.fit(target, global_roi)


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
        target.control.set_strategy(fit_strategy)
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

    STRATEGIES: dict[type, Any] = {
        Layer: LayerLayoutStrategy,
        GroupLayer: GroupLayoutStrategy,
        Canvas: CanvasLayoutStrategy,
    }

    def fit(
        self,
        target: Canvas | BaseLayer,
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
    ) -> bool:
        ref_region = resolve_region(ref)
        strategy_class = self.STRATEGIES[type(target)]
        return strategy_class.fit(target, ref_region)

    def align(
        self,
        target: Canvas | BaseLayer,
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        ref_region = resolve_region(ref)
        strategy_class = self.STRATEGIES[type(target)]
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
        strategy_class = self.STRATEGIES[type(target)]
        return strategy_class.resize_bounds(target, ref_region, anchor_x, anchor_y)

    def fit_content(
        self,
        target: Layer | GroupLayer | Canvas,
        container: Container | Sequence[Layer] | None = None,
    ) -> bool:

        strategy_class = self.STRATEGIES[type(target)]
        return strategy_class.fit_content(target, container=container)
