from __future__ import annotations
from enum import Enum
from anicrop.image import Image
from anicrop.spatial import Region, Span
from anicrop.type import Rotation, RotationInput, Scale, ScaleInput
from anicrop.transform import calculate_new_bbox_from_layer, mat_global, mat_inverse
from collections import deque
import numpy as np


class BlendMode(Enum):
    """Defines how an edit layer blends with the underlying content."""
    NORMAL = 'normal'
    MULTIPLY = 'multiply'


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
        self.blend_mode: BlendMode = blend_mode
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
        self._image = image
        self._opacity = opacity
        self._rotation = Rotation(rotation)
        self._scale = Scale(1.0, 1.0)
        self._blend_mode = blend_mode
        self._region = Region.from_size(image.width, image.height)
        self._edits: deque[EditLayer] = deque()

    def __repr__(self) -> str:
        return f"Layer(x={self.x.start}, y={self.y.start}, size={self.image.size})"

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
        return self._image

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

    def add_edit(self, image: Image, region: Region, blend_mode: BlendMode = BlendMode.NORMAL) -> None:
        name = f'Edit-{len(self._edits) + 1}'
        matrix = mat_inverse(mat_global(self))
        self._edits.append(EditLayer(image, region, matrix, blend_mode, name))
