from anicrop.canvas import Canvas
from anicrop.container import BaseLayer, Container, GroupLayer
from anicrop.image import calculate_content_bbox
from anicrop.layer import Layer
from anicrop.spatial import Region
from functools import reduce
from operator import or_
from typing import Callable, Sequence


def _resolve_group(resolve_layer: Callable, group: GroupLayer, *args):
    if len(group) == 0:
        return False

    result = False
    for child in group:
        if isinstance(child, Layer):
            result |= resolve_layer(child, *args)
        else:
            result |= _resolve_group(resolve_layer, child, *args)
    return result


def _resolve_region(
    ref: tuple[tuple[int, int], tuple[int, int]] | Region | Canvas | BaseLayer,
):
    if isinstance(ref, tuple):
        return Region.from_rect(*ref)
    elif isinstance(ref, (BaseLayer, Canvas)):
        return ref.region
    return ref


def content_region(target: Layer | GroupLayer) -> Region:
    if not target._edits:
        return None

    def roi(edit):
        return calculate_content_bbox(edit.image) + edit.region.top_left

    content_roi = reduce(or_, [roi(e) for e in target._edits])
    content_roi += target.region.top_left
    return content_roi


class LayerLayoutStrategy:

    @classmethod
    def _resolve_layer(cls, target: Layer, new_region: Region) -> bool:
        old_region = target.region
        offset = (old_region - new_region).top_left
        target.region = new_region

        for edit in target._edits:
            edit._region += offset
        return True

    @classmethod
    def crop(cls, target: Layer, ref_region: Region) -> bool:
        if not ref_region.overlaps(target.region):
            return False

        if target.region == ref_region:
            return False

        old_region = target.region
        new_region = old_region & ref_region
        return cls._resolve_layer(target, new_region)

    @classmethod
    def fit(cls, target: Layer, ref_region: Region) -> bool:
        if target.region == ref_region or not ref_region.overlaps(target.region):
            return False
        return cls._resolve_layer(target, ref_region)

    @classmethod
    def align(
        cls,
        target: Layer,
        ref_region: Region,
        factor_x: float = 0.5,
        factor_y: float = 0.5,
    ) -> bool:
        old_region = target.region
        new_region = old_region.align(ref_region, factor_x, factor_y)

        if old_region == new_region:
            return False

        target.region = new_region
        return True

    @classmethod
    def resize_bounds(
        cls,
        target: Layer,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ):
        return cls.fit(target, ref_region.align(target.region, anchor_x, anchor_y))

    @classmethod
    def fit_content(cls, target: Layer, *args, **kwargs) -> bool:
        content_roi = content_region(target)
        if target.region == content_roi:
            return False
        return cls._resolve_layer(target, content_roi)


class CanvasLayoutStrategy:
    @classmethod
    def crop(cls, target: Canvas, ref_region: Region) -> bool:
        if not ref_region.overlaps(target.region) or target.region == ref_region:
            return False

        target._region &= ref_region
        return True

    @classmethod
    def fit(cls, target: Canvas, ref_region: Region) -> bool:
        if not ref_region.overlaps(target.region) or target.region == ref_region:
            return False

        target._region = ref_region
        return True

    @classmethod
    def align(
        cls,
        target: Canvas,
        ref_region: Region,
        factor_x: float = 0.5,
        factor_y: float = 0.5,
    ) -> bool:
        new_region = target._region.align(ref_region, factor_x, factor_y)
        if target._region == new_region:
            return False
        target._region = new_region
        return True

    @classmethod
    def resize_bounds(
        cls,
        target: Canvas,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ):
        ref_region = ref_region.align(target.region, anchor_x, anchor_y)
        return cls.fit(target, ref_region)

    @classmethod
    def _extract_items(
        cls,
        item: Layer | Container,
    ) -> Region:
        if isinstance(item, Layer):
            yield content_region(item)
        else:
            for child in item:
                yield from cls._extract_items(child)

    @classmethod
    def fit_content(
        cls,
        target: Canvas,
        container: Container | Sequence[Layer],
    ) -> bool:

        if len(container) == 0:
            return False

        regions = filter(None, cls._extract_items(container))

        try:
            new_region = next(regions)
        except StopIteration:
            return False

        new_region = reduce(or_, regions, new_region)
        if new_region == target._region:
            return False

        target._region = new_region

        return True


class GroupLayoutStrategy:

    @classmethod
    def crop(cls, target: GroupLayer, ref_region: Region) -> bool:
        return _resolve_group(LayerLayoutStrategy.crop, target, ref_region)

    @classmethod
    def fit(cls, target: GroupLayer, ref_region: Region) -> bool:
        return _resolve_group(LayerLayoutStrategy.fit, target, ref_region)

    @classmethod
    def align(
        cls,
        target: GroupLayer,
        ref_region: Region,
        factor_x: float = 0.5,
        factor_y: float = 0.5,
    ) -> bool:
        return _resolve_group(
            LayerLayoutStrategy.align, target, ref_region, factor_x, factor_y
        )

    @classmethod
    def resize_bounds(
        cls,
        target: GroupLayer,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        return _resolve_group(
            LayerLayoutStrategy.resize_bounds, target, ref_region, anchor_x, anchor_y
        )

    @classmethod
    def fit_content(cls, target: GroupLayer, *args, **kwargs) -> bool:
        return _resolve_group(LayerLayoutStrategy.fit_content, target)


class Layout:

    STRATEGIES = {
        Layer: LayerLayoutStrategy,
        GroupLayer: GroupLayoutStrategy,
        Canvas: CanvasLayoutStrategy,
    }

    def crop(
        self,
        target: Canvas | BaseLayer,
        ref: tuple[tuple[int, int], tuple[int, int]] | Region | Canvas | BaseLayer,
    ) -> bool:
        ref_region = _resolve_region(ref)
        strategy_class = self.STRATEGIES[type(target)]
        return strategy_class.crop(target, ref_region)

    def fit(
        self,
        target: Canvas | BaseLayer,
        ref: tuple[tuple[int, int], tuple[int, int]] | Region | Canvas | BaseLayer,
    ) -> bool:
        ref_region = _resolve_region(ref)
        strategy_class = self.STRATEGIES[type(target)]
        return strategy_class.fit(target, ref_region)

    def align(
        self,
        target: Canvas | BaseLayer,
        ref: tuple[tuple[int, int], tuple[int, int]] | Region | Canvas | BaseLayer,
        factor_x: float = 0.5,
        factor_y: float = 0.5,
    ) -> bool:
        ref_region = _resolve_region(ref)
        strategy_class = self.STRATEGIES[type(target)]
        return strategy_class.align(target, ref_region, factor_x, factor_y)

    def resize_bounds(
        self,
        target: Layer | Canvas | GroupLayer,
        new_width: int,
        new_height: int,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ):
        ref_region = Region.from_size(new_width, new_height)
        strategy_class = self.STRATEGIES[type(target)]
        return strategy_class.resize_bounds(target, ref_region, anchor_x, anchor_y)

    def fit_content(
        self,
        target: Layer | GroupLayer | Canvas,
        container: Container | Sequence[Layer] = None,
    ) -> bool:
        strategy_class = self.STRATEGIES[type(target)]
        return strategy_class.fit_content(target, container=container)
