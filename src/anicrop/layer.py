from __future__ import annotations
from collections import deque
from typing import Any, Optional, TYPE_CHECKING

from anicrop.container import (
    _NULL_CONTAINER,
    BaseLayer,
    _compute_layer_local_roi,
    global_content_region,
)
from anicrop.enums import BlendMode, RenderFlags, WarpMode, ImageFormat
from anicrop.geometry import LayerGeometry, FitGeometry
from anicrop.image import Image
from anicrop.spatial import Region, Span
from anicrop.type import Id
from anicrop.transform import (
    calculate_new_rect,
    calculate_region_rect,
    mat_global,
    mat_inverse,
    transform_vector,
)

import numpy as np
from anicrop.edit_layer import EditLayer, EDIT_LAYER_MAP, CropEditLayer

if TYPE_CHECKING:
    from anicrop.canvas import Canvas


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
    """Estratégia de layout para a moldura de uma camada individual (Layer)."""

    def __init__(self, target: Layer) -> None:
        self.target = target

    def fit(
        self,
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
    ) -> bool:
        return self._fit(self.target, self._resolve_region(ref))

    def align(
        self,
        ref: tuple[int, int, int, int] | Region | Canvas | BaseLayer,
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> bool:
        return self._align(self.target, self._resolve_region(ref), anchor_x, anchor_y)

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

    @staticmethod
    def _resolve_region(ref: Any) -> Region:
        if isinstance(ref, tuple):
            return Region.from_rect(*ref)
        elif isinstance(ref, Region):
            return ref
        elif isinstance(ref, BaseLayer):
            return ref.global_region
        elif hasattr(ref, "region"):
            return ref.region
        return ref


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
        crop_region = LayerLayoutStrategy._resolve_region(ref)

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
        if isinstance(ref, tuple) and len(ref) == 2 and isinstance(ref[1], Region):
            ref_region = ref[1]
        else:
            ref_region = LayerLayoutStrategy._resolve_region(ref)

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
        self._content = LayerContent(self)
        self._layout = LayerLayoutStrategy(self)

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
        return self._content

    @property
    def layout(self) -> LayerLayoutStrategy:
        """Estratégia de layout para a moldura desta camada."""
        return self._layout

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
