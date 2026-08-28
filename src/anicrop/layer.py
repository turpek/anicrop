from __future__ import annotations
from collections import deque
from typing import Optional, TYPE_CHECKING

from anicrop.container import (
    _NULL_CONTAINER,
    BaseLayer,
)
from anicrop.content import LayerContent
from anicrop.interfaces.layer import AbstractLayer
from anicrop.layout import LayerLayoutStrategy
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
    pass


class Layer(BaseLayer, AbstractLayer):

    def __init__(
        self,
        image: Image,
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        name: str = 'Layer',
    ):

        self.parent = _NULL_CONTAINER
        region = Region.from_size(*image.size)
        super().__init__(
            self.parent, LayerGeometry, region, opacity, blend_mode, name, format=image.format
        )

        self._id = Id()
        self._edits: deque[EditLayer] = deque()
        self._opacity_mask: Optional[np.ndarray] = None
        self._parent_inverse = np.identity(3, dtype=np.float32)

        self.add_edit(image, region, blend_mode)
        self._old_matrix = np.zeros((3, 3))
        self._render_flags = RenderFlags.ALL_DIRTY
        self._warp_mode = WarpMode.AFFINE
        self._content = LayerContent(self)
        self._layout = LayerLayoutStrategy(self)

    def __repr__(self) -> str:
        return f"Layer(x={self.x.start}, y={self.y.start}, size={self.region.size})"

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
    def edits(self) -> tuple[EditLayer, ...]:
        """Coleção de edições e patches locais da camada."""
        return tuple(self._edits)

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
