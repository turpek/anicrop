from __future__ import annotations
from enum import Enum
from anicrop.image import Image
from anicrop.spatial import Region, Span
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
        self.opacity = opacity
        self._rotate = 0.0
        self._scale = 1.0
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

        return Region(Span(start_x, end_x + 1), Span(start_y, end_y + 1))

    @property
    def rotate(self) -> float:
        return self._rotate

    @rotate.setter
    def rotate(self, rotate: float) -> None:
        self._rotate = rotate

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        self._scale = scale

    @property
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, region: Region) -> None:
        if not isinstance(region, Region):
            raise TypeError(f"Expected Region, got {type(region).__name__}")
        self._region = region
