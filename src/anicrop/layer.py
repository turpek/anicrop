from __future__ import annotations
from enum import Enum
from anicrop.image import Image
from anicrop.spatial import Region, Span
from anicrop.type import Rotation, RotationInput, Scale, ScaleInput
from anicrop.transform import calculate_new_bbox_from_layer, mat_global, mat_inverse
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
            opacity: float = 1.0,
            blend_mode: BlendMode = BlendMode.NORMAL,
            name: str = 'Edit'
    ):
        """Initializes the EditLayer.

        The layer's region is automatically calculated based on the non-transparent
        content of the provided image. The internal image data is cropped to this
        bounding box to optimize storage.

        Args:
            image: The source image for the layer.
            opacity: The layer opacity (0.0 to 1.0).
            blend_mode: The blending mode to use.
            name: A descriptive name for the layer.

        Raises:
            ValueError: If the provided image is fully transparent (has no content).
        """
        self.name = name
        self._region = self._calculate_content_bbox(image)
        bbox_image = image[self.region].copy()
        self.image = Image(bbox_image, image.format)
        self.opacity = Float(opacity)
        self._rotate = Float(0.0)
        self._scale = Float(1.0)
        self.blend_mode: BlendMode = blend_mode

    def _calculate_content_bbox(self, image: Image) -> Region:
        """Calculates the bounding box of the non-transparent content.

        Iterates through the alpha channel to find the minimum and maximum
        coordinates that contain visible pixels.

        Args:
            image: The image to analyze.

        Returns:
            A Region object representing the smallest rectangle containing all
            non-transparent pixels. If the image has no alpha channel, returns
            the full image region.

        Raises:
            ValueError: If the image has an alpha channel but contains no visible pixels.
        """
        if not image.has_alpha:
            return Region.from_size(image.width, image.height)

        alpha = image[..., -1]
        if not np.any(alpha):
            raise ValueError("EditLayer cannot be created from a fully transparent image.")
        axis_y, axis_x = np.where(alpha > 0)

        start_x, end_x = int(axis_x.min()), int(axis_x.max())
        start_y, end_y = int(axis_y.min()), int(axis_y.max())
        width = end_x - start_x + 1
        height = end_y - start_y + 1
        return Region(Span(start_x, width), Span(start_y, height))

    @property
    def rotate(self) -> float:
        return self._rotate

    @rotate.setter
    def rotate(self, rotate: float) -> None:
        self._rotate = Float(rotate)

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        self._scale = Float(scale)

    @property
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, region: Region) -> None:
        if not isinstance(region, Region):
            raise TypeError(f"Expected Region, got {type(region).__name__}")
        self._region = region


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
