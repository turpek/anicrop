from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Optional, overload

import numpy as np
from ovld import ovld

from anicrop.container import (
    _NULL_CONTAINER,
    BaseLayer,
)
from anicrop.content import LayerContentStrategy
from anicrop.edit_layer import EDIT_LAYER_MAP, EditLayer
from anicrop.enums import BlendMode, ImageFormat, RenderFlags, WarpMode
from anicrop.geometry import LayerGeometry
from anicrop.image import Image
from anicrop.interfaces.layer import AbstractLayer
from anicrop.layout import LayerLayoutStrategy
from anicrop.spatial import Region, Span
from anicrop.transform import (
    mat_global,
    mat_inverse,
)
from anicrop.type import Id


class Layer(BaseLayer, AbstractLayer):
    def _init_base(
        self,
        region: Region,
        opacity: float,
        blend_mode: BlendMode,
        name: str,
        format: ImageFormat,
    ) -> None:
        self.parent = _NULL_CONTAINER
        super().__init__(
            self.parent, LayerGeometry, region, opacity, blend_mode, name, format=format
        )
        self._id = Id()
        self._edits: deque[EditLayer] = deque()
        self._opacity_mask: Optional[np.ndarray] = None
        self._parent_inverse = np.identity(3, dtype=np.float32)
        self._old_matrix = np.zeros((3, 3))
        self._render_flags = RenderFlags.ALL_DIRTY
        self._warp_mode = WarpMode.AFFINE
        self._content = LayerContentStrategy(self)
        self._layout = LayerLayoutStrategy(self)

    if TYPE_CHECKING:

        @overload
        def __init__(
            self,
            image: Image,
            *,
            opacity: float = 1.0,
            blend_mode: BlendMode = BlendMode.NORMAL,
            name: str = "Layer",
        ) -> None:
            pass

        @overload
        def __init__(
            self,
            region: Region,
            *,
            opacity: float = 1.0,
            blend_mode: BlendMode = BlendMode.NORMAL,
            name: str = "Layer",
            format: ImageFormat = ImageFormat.RGBA,
        ) -> None:
            pass

        @overload
        def __init__(
            self,
            size: tuple[float, float],
            *,
            opacity: float = 1.0,
            blend_mode: BlendMode = BlendMode.NORMAL,
            name: str = "Layer",
            format: ImageFormat = ImageFormat.RGBA,
        ) -> None:
            pass

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
    else:

        @ovld
        def __init__(
            self,
            image: Image,
            *,
            opacity: float = 1.0,
            blend_mode: BlendMode = BlendMode.NORMAL,
            name: str = "Layer",
        ) -> None:
            self._init_base(
                Region.from_size(*image.size), opacity, blend_mode, name, image.format
            )
            self.add_edit(image, self.base.region, blend_mode)

        @ovld
        def __init__(  # noqa: F811
            self,
            region: Region,
            *,
            opacity: float = 1.0,
            blend_mode: BlendMode = BlendMode.NORMAL,
            name: str = "Layer",
            format: ImageFormat = ImageFormat.RGBA,
        ) -> None:
            self._init_base(region, opacity, blend_mode, name, format)

        @ovld
        def __init__(  # noqa: F811
            self,
            size: tuple,
            *,
            opacity: float = 1.0,
            blend_mode: BlendMode = BlendMode.NORMAL,
            name: str = "Layer",
            format: ImageFormat = ImageFormat.RGBA,
        ) -> None:
            self._init_base(Region.from_size(*size), opacity, blend_mode, name, format)

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
    def content(self) -> LayerContentStrategy:
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
        global_matrix: np.ndarray | None = None,
    ) -> EditLayer:

        inv_mat = mat_inverse(self.matrix)
        matrix = inv_mat if global_matrix is None else inv_mat @ global_matrix
        edit_cls = EDIT_LAYER_MAP.get(blend_mode, EditLayer)
        edit_name = name or blend_mode.default_name
        edit = edit_cls(image, region, matrix, blend_mode, edit_name, visible)
        self._edits.append(edit)
        return edit
