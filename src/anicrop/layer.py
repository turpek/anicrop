from __future__ import annotations
from collections import deque
from typing import Optional, TYPE_CHECKING

from anicrop.container import _NULL_CONTAINER, BaseLayer
from anicrop.enums import BlendMode, RenderFlags, WarpMode
from anicrop.geometry import LayerGeometry
from anicrop.image import Image
from anicrop.spatial import Region, Span
from anicrop.type import Id
from anicrop.transform import (
    mat_global,
    mat_inverse,
)

import numpy as np
from anicrop.edit_layer import EditLayer, EDIT_LAYER_MAP

if TYPE_CHECKING:
    from anicrop.canvas import Canvas


class LayerContent:
    """Gerenciador de manipulação, transformação e ajuste de conteúdo/pixels em uma camada específica."""

    def __init__(self, target: Layer) -> None:
        self.target = target

    def crop(
        self,
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
    ) -> bool:
        return self._crop(self.target, ref)

    def resize(
        self,
        width: int,
        height: int,
    ) -> bool:
        return self._resize(self.target, width, height)

    def fit(
        self,
        ref: tuple | Region | Canvas | BaseLayer,
    ) -> bool:
        return self._fit(self.target, ref)

    def flip_x(self) -> bool:
        return self._flip_x(self.target)

    def flip_y(self) -> bool:
        return self._flip_y(self.target)

    @classmethod
    def _crop(
        cls,
        target: Layer,
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
    ) -> bool:
        from anicrop.layout import LayerLayoutStrategy, resolve_region
        from anicrop.enums import ImageFormat

        crop_region = resolve_region(ref)

        if not LayerLayoutStrategy._fit(target, crop_region):
            return False

        mask_region = target.global_region
        mask_image = Image.new(mask_region.size, ImageFormat.GRAY, color=255)
        target.add_edit(mask_image, mask_region, blend_mode=BlendMode.CLIP)
        return True

    @classmethod
    def _resize(
        cls,
        target: Layer,
        width: int,
        height: int,
    ) -> bool:
        if width <= 0 or height <= 0:
            raise ValueError(f"Dimensões inválidas para resize: ({width}, {height}). Devem ser positivas.")

        cur_w, cur_h = target.global_region.size
        if (cur_w, cur_h) == (width, height):
            return False

        scale_x = width / cur_w
        scale_y = height / cur_h

        target.transform.scale(scale_x, scale_y)
        return True

    @classmethod
    def _fit(
        cls,
        target: Layer,
        ref: tuple | Region | Canvas | BaseLayer,
    ) -> bool:
        from anicrop.layout import resolve_region
        from anicrop.transform import transform_vector

        if isinstance(ref, tuple) and len(ref) == 2 and isinstance(ref[1], Region):
            ref_region = ref[1]
        else:
            ref_region = resolve_region(ref)

        cur_region = target.global_region

        if cur_region == ref_region:
            return False

        scale_x = ref_region.width / cur_region.width
        scale_y = ref_region.height / cur_region.height

        if scale_x <= 0 or scale_y <= 0:
            return False

        target.transform.scale(scale_x, scale_y)

        new_global_region = target.global_region.align(ref_region, 0.0, 0.0)
        dx, dy = transform_vector(
            mat_inverse(target.parent.matrix),
            target.global_region,
            new_global_region,
        )
        target.transform.translate(dx, dy)
        return True

    @classmethod
    def _flip_x(cls, target: Layer) -> bool:
        target.transform.scale(-1, 1)
        return True

    @classmethod
    def _flip_y(cls, target: Layer) -> bool:
        target.transform.scale(1, -1)
        return True


class Layer(BaseLayer):

    def __init__(
        self,
        image: Image,
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        name: str = 'Layer',
        canvas: Optional[Canvas] = None,
    ):

        self.parent = _NULL_CONTAINER
        region = Region.from_size(*image.size)
        super().__init__(self.parent, LayerGeometry, region, opacity, blend_mode, name)

        self._id = Id()
        self._edits: deque[EditLayer] = deque()
        self._opacity_mask: Optional[np.ndarray] = None
        self._parent_inverse = np.identity(3, dtype=np.float32)
        self._canvas = canvas

        self.add_edit(image, region, blend_mode)
        self._image = self._edits[0]
        self._old_matrix = np.zeros((3, 3))
        self._render_flags = RenderFlags.ALL_DIRTY
        self._warp_mode = WarpMode.AFFINE

    def __repr__(self) -> str:
        return f"Layer(x={self.x.start}, y={self.y.start}, size={self.image.size})"

    def __eq__(self, other):
        return isinstance(other, Layer) and self._id == other._id

    def __hash__(self):
        return hash(self._id)

    def _resolve_render(self) -> RenderFlags:
        current_matrix = mat_global(self)

        # 2. Compara a parte 2x2 (rotação/escala)
        if not np.allclose(current_matrix[:2, :2], self._old_matrix[:2, :2]):
            self._render_flags |= RenderFlags.PIXELS

        # 3. Compara a translação [tx, ty]
        if not np.allclose(current_matrix[:2, 2], self._old_matrix[:2, 2]):
            self._render_flags |= RenderFlags.POSITION

        if not np.allclose(current_matrix[2, :2], [0, 0]):
            self._warp_mode = WarpMode.PERSPECTIVE
        else:
            self._warp_mode = WarpMode.AFFINE

        return self._render_flags

    def _commit_render_state(self):
        self._old_matrix = mat_global(self)
        self._render_flags = RenderFlags.NONE

    @property
    def format(self):
        return self.image.format

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self._canvas.size if self._canvas else self.base.region.size

    @property
    def content(self) -> LayerContent:
        """Gerenciador de manipulação, transformação e ajuste de conteúdo/pixels."""
        return LayerContent(self)

    @property
    def region(self) -> Region:
        return self.control.frame.region

    @region.setter
    def region(self, other: Region) -> None:
        if not isinstance(other, Region):
            raise TypeError(f"Expected Region, got {type(other).__name__}")
        self.control.sync(other)

    @property
    def x(self) -> Span:
        return self.base.region.x

    @x.setter
    def x(self, value: int | Span):
        self.control.set_x(value)

    @property
    def y(self) -> Span:
        return self.base.region.y

    @y.setter
    def y(self, value: int | Span):
        self.control.set_y(value)

    @property
    def image(self) -> Image:
        return self._image.image

    def add_edit(
        self,
        image: Image,
        region: Region,
        blend_mode: BlendMode = BlendMode.NORMAL,
        name: str | None = None,
        visible: bool = True,
    ) -> EditLayer:

        matrix = mat_inverse(self.matrix)
        edit_cls = EDIT_LAYER_MAP.get(blend_mode, EditLayer)
        edit_name = name or blend_mode.default_name
        edit = edit_cls(image, region, matrix, blend_mode, edit_name, visible)
        self._edits.append(edit)
        return edit
