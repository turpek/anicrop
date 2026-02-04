from __future__ import annotations
from enum import Enum
from anicrop.image import Image
from anicrop.spatial import Region, Span
import numpy as np


class BlendMode(Enum):
    """Defines how an edit layer blends with the underlying content."""
    NORMAL = 'normal'


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
        self.name = name
        self.region = self._calculate_content_bbox(image)
        self.image = Image(image[self.region], image.format)
        self.opacity = opacity
        self.blend_mode: BlendMode = blend_mode

    def _calculate_content_bbox(self, image: Image) -> Region:
        if not image.has_alpha:
            return Region.from_size(image.width, image.height)

        alpha = image[..., -1]
        if not np.any(alpha):
            raise ValueError("EditLayer cannot be created from a fully transparent image.")
        axis_y, axis_x = np.where(alpha > 0)

        start_x, end_x = int(axis_x.min()), int(axis_x.max())
        start_y, end_y = int(axis_y.min()), int(axis_y.max())

        return Region(Span(start_x, end_x + 1), Span(start_y, end_y + 1))
