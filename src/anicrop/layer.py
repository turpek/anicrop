from __future__ import annotations

from anicrop.blend import BlendMode
from anicrop.image import Image
from anicrop.spatial import Region, Span
from anicrop.type import Rotation, RotationInput, Scale, ScaleInput
from anicrop.transform import (
    calculate_new_bbox_from_layer,
    mat_global,
    mat_inverse,
    mat_position,
    Transform,
    TransformComposer,
)
from collections import deque
from typing import Optional

import numpy as np


class EditLayer:
    """Represents a destructive edit applied to a base layer.

    Attributes:
        image: The pixel data of the edit (the 'patch').
        position: The region where this edit is applied relative to the parent layer.
        name: An optional name for the edit (useful for history/undo stacks).
        opacity: The opacity of this specific edit (0.0 to 1.0).
        blend_mode: How this edit blends with the base layer.
    """
    def __init__(
        self,
        image: Image,
        region: Region,
        matrix: np.ndarray,
        blend_mode: BlendMode = BlendMode.NORMAL,
        name: str = 'Edit'
    ):

        self._image = image
        self._region = region
        self.blend_mode = blend_mode
        self.name = name
        self._matrix = matrix

    @property
    def region(self) -> Region:
        return self._region

    @property
    def image(self) -> Image:
        return self._image

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def local_matrix(self) -> np.ndarray:
        return mat_position(self.region) @ self.matrix

    def offset(self, offset_x: int, offset_y: int) -> None:
        self._region += (offset_x, offset_y)


class Layer:

    def __init__(
        self,
        image: Image,
        opacity: float = 1.0,
        rotation: float = 0.0,
        scale: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        name: str = 'Layer'
    ):
        self._name = name
        self._opacity = opacity
        self._rotation = Rotation(rotation)
        self._scale = Scale(scale, scale)
        self._blend_mode = blend_mode
        self._region = Region.from_size(*image.size)
        self._edits: deque[EditLayer] = deque()
        self._transform: Optional[TransformComposer] = None

        self.add_edit(image, self._region, blend_mode)
        self._image = self._edits[0]

    def __repr__(self) -> str:
        return f"Layer(x={self.x.start}, y={self.y.start}, size={self.image.size})"

    @property
    def format(self):
        return self.image.format

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def x(self) -> Span:
        return self.region.x

    @x.setter
    def x(self, value: int | Span):
        if isinstance(value, Span):
            self._region = Region(value, self.y)
        elif isinstance(value, int):
            self._region = Region(Span(value, self.x.length), self.y)

    @property
    def y(self) -> Span:
        return self.region.y

    @y.setter
    def y(self, value: int | Span):
        if isinstance(value, Span):
            self._region = Region(self.x, value)
        elif isinstance(value, int):
            self._region = Region(self.x, Span(value, self.y.length))

    @property
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, opacity: float) -> None:
        self._opacity = opacity

    @property
    def rotation(self) -> Rotation:
        return self._rotation

    @rotation.setter
    def rotation(self, value: Rotation | RotationInput) -> None:
        self._rotation = self._rotation.from_input(value)

    @property
    def scale(self) -> Scale:
        return self._scale

    @scale.setter
    def scale(self, value: Scale | ScaleInput) -> None:
        self._scale = self._scale.from_input(value)

    @property
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, region: Region) -> None:
        if not isinstance(region, Region):
            raise TypeError(f"Expected Region, got {type(region).__name__}")
        self._region = region

    @property
    def image(self) -> Image:
        return self._image.image

    @property
    def blend_mode(self) -> BlendMode:
        return self._blend_mode

    @blend_mode.setter
    def blend_mode(self, blend_mode: BlendMode) -> None:
        self._blend_mode = blend_mode

    @property
    def canvas_region(self) -> Region:
        """Retorna o BBox (AABB) real do layer no espaço do Canvas."""

        x, y, w, h = calculate_new_bbox_from_layer(self)
        return Region(Span(x, w), Span(y, h))

    def add_edit(
        self,
        image: Image,
        region: Region,
        blend_mode: BlendMode = BlendMode.NORMAL
    ) -> None:

        name = f'Edit-{len(self._edits) + 1}'
        matrix = mat_inverse(mat_global(self))
        self._edits.append(EditLayer(image, region, matrix, blend_mode, name))

    @property
    def transform_used(self) -> bool:
        return isinstance(self._transform, TransformComposer)

    def transform_clear(self) -> None:
        self._transform = None

    @property
    def transform(self) -> TransformComposer:
        if self._transform is None:
            self._transform = TransformComposer(self.region.size)
        return self._transform

    def set_transform(self, transform: Transform) -> None:
        self._transform = TransformComposer(self.region.size)
        self.transform._add_transform(transform, self.region.size)
