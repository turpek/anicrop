from anicrop.canvas import Canvas
from anicrop.container import BaseLayer, Container, GroupLayer
from anicrop.image import calculate_content_bbox
from anicrop.layer import Layer
from anicrop.spatial import Region
from functools import reduce
from operator import or_
from typing import Callable, Iterable


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
        if not target._edits:
            return False

        def content_region(edit):
            return calculate_content_bbox(edit.image) + edit.region.top_left

        content_region = reduce(or_, [content_region(e) for e in target._edits])
        content_region += target.region.top_left

        if target.region == content_region:
            return False
        return cls._resolve_layer(target, content_region)


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
    def _resolve_content(
        cls,
        item: Layer | GroupLayer,
    ) -> Region:
        ...
        if not item._edits:
            return None

        def content_region(edit):
            return calculate_content_bbox(edit.image) + edit.region.top_left

        content_region = reduce(or_, [content_region(e) for e in item._edits])
        content_region += item.region.top_left
        return content_region

    @classmethod
    def _resolve_loop(
        cls,
        target_region: Region | None,
        item: Layer | Container,
    ) -> Region:

        if isinstance(item, Layer):
            content_region = cls._resolve_content(item)
            if target_region is None:
                return content_region
            if content_region:
                return target_region | content_region
            return target_region

        for child in item:
            reg = cls._resolve_loop(target_region, child)
            if reg and target_region is None:
                target_region = reg
            elif reg:
                target_region |= reg

        return target_region

    @classmethod
    def fit_content(
        cls,
        target: Canvas,
        container: Container | Iterable[Layer] = None,
    ) -> bool:

        new_region = cls._resolve_loop(None, container)
        if new_region:
            target._region = new_region
            return True
        return False


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
        container: Container | Iterable[Layer] = None,
    ) -> bool:
        strategy_class = self.STRATEGIES[type(target)]
        return strategy_class.fit_content(target, container=container)
