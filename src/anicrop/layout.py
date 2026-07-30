from anicrop.canvas import Canvas
from anicrop.container import BaseLayer, GroupLayer
from anicrop.image import calculate_content_bbox
from anicrop.layer import Layer
from anicrop.spatial import Region
from functools import reduce
from operator import or_
from typing import Callable


class Layout:

    def _resolve_region(
        self,
        ref: tuple[tuple[int, int], tuple[int, int]] | Region | Canvas | BaseLayer,
    ):
        if isinstance(ref, tuple):
            return Region.from_rect(*ref)
        elif isinstance(ref, (BaseLayer, Canvas)):
            return ref.region
        return ref

    def _resolve_group(self, resolve_layer: Callable, group: GroupLayer, *args):
        if len(group) == 0:
            return False

        result = False
        for child in group:
            if isinstance(child, Layer):
                result |= resolve_layer(child, *args)
            else:
                result |= self._resolve_group(resolve_layer, child, *args)
        return result

    def _resolve_layer(self, target: Layer, new_region: Region) -> bool:
        old_region = target.region
        offset = (old_region - new_region).top_left
        target.region = new_region

        for edit in target._edits:
            edit._region += offset
        return True

    def _crop_layer(self, target: Layer, ref_region: Region) -> bool:
        if not ref_region.overlaps(target.region):
            return False

        if target.region == ref_region:
            return False

        old_region = target.region
        new_region = old_region & ref_region
        return self._resolve_layer(target, new_region)

    def _fit_layer(self, target: Layer, ref_region: Region) -> bool:
        if target.region == ref_region:
            return False
        if not ref_region.overlaps(target.region):
            return False
        if target.region == ref_region:
            return False
        return self._resolve_layer(target, ref_region)

    def _align_layer(
        self,
        target: Layer,
        ref_region: Region,
        x_factor: float = 0.5,
        y_factor: float = 0.5,
    ) -> bool:
        old_region = target.region
        new_region = old_region.align(ref_region, x_factor, y_factor)

        if old_region == new_region:
            return False
        return self._resolve_layer(target, new_region)

    def _resize_bounds_layer(
        self,
        target: Layer | Canvas | GroupLayer,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ):
        return self.fit(target, ref_region.align(target.region, anchor_x, anchor_y))

    def _fit_content_layer(self, target: Layer | GroupLayer) -> bool:
        if not target._edits:
            return False

        def content_region(edit):
            return calculate_content_bbox(edit.image) + edit.region.top_left

        content_region = reduce(or_, [content_region(e) for e in target._edits])
        content_region += target.region.top_left

        if target.region == content_region:
            return False
        return self.fit(target, content_region)

    def _crop_canvas(self, target, ref_region: Region) -> bool:
        if not ref_region.overlaps(target.region):
            return False
        if target.region == ref_region:
            return False
        target._region &= ref_region
        return True

    def _fit_canvas(self, target: Layer, ref_region: Region) -> bool:
        if not ref_region.overlaps(target.region):
            return False
        if target.region == ref_region:
            return False
        target._region = ref_region
        return True

    def _align_canvas(
        self,
        target: Layer,
        ref_region: Region,
        x_factor: float = 0.5,
        y_factor: float = 0.5,
    ) -> bool:
        new_region = target._region.align(ref_region, x_factor, y_factor)
        if target._region == new_region:
            return False
        target._region = new_region
        return True

    def _resize_bounds_canvas(
        self,
        target: Layer | Canvas | GroupLayer,
        ref_region: Region,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ):
        return self.fit(target, ref_region.align(target.region, anchor_x, anchor_y))

    def crop(
        self,
        target: Canvas | BaseLayer,
        ref: tuple[tuple[int, int], tuple[int, int]] | Region | Canvas | BaseLayer,
    ) -> bool:

        ref_region = self._resolve_region(ref)

        if isinstance(target, Layer):
            return self._crop_layer(target, ref_region)

        elif isinstance(target, GroupLayer):
            return self._resolve_group(self._crop_layer, target, ref_region)

        elif isinstance(target, Canvas):
            return self._crop_canvas(target, ref_region)

        return False

    def fit(
        self,
        target: Canvas | BaseLayer,
        ref: tuple[tuple[int, int], tuple[int, int]] | Region | Canvas | BaseLayer,
    ) -> bool:

        ref_region = self._resolve_region(ref)

        if isinstance(target, Layer):
            return self._fit_layer(target, ref_region)

        elif isinstance(target, GroupLayer):
            return self._resolve_group(self._fit_layer, target, ref_region)

        elif isinstance(target, Canvas):
            return self._fit_canvas(target, ref_region)

    def align(
        self,
        target: Canvas | BaseLayer,
        ref: tuple[tuple[int, int], tuple[int, int]] | Region | Canvas | BaseLayer,
        x_factor: float = 0.5,
        y_factor: float = 0.5,
    ) -> bool:
        ref_region = self._resolve_region(ref)

        if isinstance(target, Layer):
            return self._align_layer(target, ref_region, x_factor, y_factor)

        elif isinstance(target, GroupLayer):
            return self._resolve_group(
                self._align_layer, target, ref_region, x_factor, y_factor
            )

        elif isinstance(target, Canvas):
            return self._align_canvas(target, ref_region, x_factor, y_factor)

    def resize_bounds(
        self,
        target: Layer | Canvas | GroupLayer,
        new_width: int,
        new_height: int,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ):
        ref_region = Region.from_size(new_width, new_height)
        if isinstance(target, Layer):
            return self._resize_bounds_layer(target, ref_region, anchor_x, anchor_y)
        elif isinstance(target, GroupLayer):
            return self._resolve_group(
                self._resize_bounds_layer, target, ref_region, anchor_x, anchor_y
            )
        elif isinstance(target, Canvas):
            return self._resize_bounds_layer(target, ref_region, anchor_x, anchor_y)

    def fit_content(self, target: Layer | GroupLayer) -> bool:
        if isinstance(target, Layer):
            return self._fit_content_layer(target)
        elif isinstance(target, GroupLayer):
            return self._resolve_group(self._fit_content_layer, target)
