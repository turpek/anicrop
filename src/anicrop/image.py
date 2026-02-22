"""Provides the Image class, a wrapper for image data processing."""
from __future__ import annotations
from numpy import ndarray
from anicrop.spatial import Region, Span
from typing import Any
from enum import StrEnum
import numpy as np


class ImageFormat(StrEnum):
    GRAY = "gray"
    GRAY_ALPHA = "gray_alpha"
    RGB = "rgb"
    RGBA = "rgba"
    CMYK = "cmyk"
    CMYK_ALPHA = "cmyk_alpha"

    @property
    def has_alpha(self) -> bool:
        return self in {
            ImageFormat.GRAY_ALPHA,
            ImageFormat.RGBA,
            ImageFormat.CMYK_ALPHA,
        }

    @property
    def channels(self) -> int:
        return {
            ImageFormat.GRAY: 1,
            ImageFormat.GRAY_ALPHA: 2,
            ImageFormat.RGB: 3,
            ImageFormat.RGBA: 4,
            ImageFormat.CMYK: 4,
            ImageFormat.CMYK_ALPHA: 5,
        }[self]

    def same_spaces(self, other: ImageFormat) -> bool:
        color_spaces = {
            "gray": "gray",
            "gray_alpha": "gray",
            "rgb": "rgb",
            "rgba": "rgb",
            "cmyk": "cmyk",
            "cmyk_alpha": "cmyk",
        }
        return color_spaces[other] == color_spaces[self]


class Image:
    """A wrapper around a NumPy ndarray to provide an image-centric API.

    This class facilitates spatial indexing using Region objects and offers
    convenient properties for accessing image dimensions (width, height, channels).
    It ensures that the underlying image data is a valid 2D or 3D array.
    """
    def __init__(self, image: ndarray, image_format: ImageFormat):
        """Initializes the Image object.

        Args:
            image: A 2D (grayscale) or 3D (color) NumPy ndarray.

        Raises:
            ValueError: If the image array is not 2D/3D, has zero dimensions,
                        or has no channels in a 3D configuration.
        """
        if image.ndim not in (2, 3):
            raise ValueError("image array must be 2D or 3D")

        elif image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image dimensions must be greater than zero")
        elif image.ndim == 2:
            image = image[..., np.newaxis]
        elif image.ndim == 3 and image.shape[2] == 0:
            raise ValueError("image must have at least one channel")

        self._data = image
        self._channels = image.shape[2]
        self._format = image_format
        self._validate_format()

    def __region_to_slice(self, region: Region) -> tuple[slice, slice]:
        """Converts a Region object to a tuple of slices for NumPy indexing."""
        return (
            slice(region.y.start, region.y.end),
            slice(region.x.start, region.x.end),
        )

    def __to_indexer(self, key: Any) -> Any:
        """Translates a key, potentially a Region, into a valid NumPy indexer."""
        if isinstance(key, Region):
            return self.__region_to_slice(key)

        elif isinstance(key, tuple):
            if any(isinstance(arg, Region) for arg in key[1:]):
                raise TypeError("Region argument is only valid at the first position")

            elif isinstance(key[0], Region):
                return self.__region_to_slice(key[0]) + key[1:]

        return key

    def __getitem__(self, key: Region | Any) -> ndarray:
        """Retrieves a part of the image using indexing.

        Supports standard NumPy indexing and spatial indexing with a Region object.
        When a Region is used, it can be the sole index or the first element
        in a tuple for further channel/slice selection.

        Args:
            key: A Region object, a standard NumPy index, or a tuple
                 starting with a Region.

        Returns:
            The selected ndarray slice of the image data.
        """
        return self._data[self.__to_indexer(key)]

    def __setitem__(self, key: Region | Any, value: Any) -> None:
        """Sets a part of the image using indexing.

        Supports standard NumPy indexing and spatial indexing with a Region object.
        When a Region is used, it can be the sole index or the first element
        in a tuple for further channel/slice selection.

        Args:
            key: A Region object, a standard NumPy index, or a tuple
                 starting with a Region.
            value: The value or ndarray to assign to the specified slice.
        """
        self._data[self.__to_indexer(key)] = value

    def _validate_format(self):
        channels = self.channels
        formt = self.format
        if channels != formt.channels:
            raise ValueError(
                f"Image format '{formt}' expects {formt.channels} channels, "
                f"but data has {channels}."
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying image data as a tuple."""
        return self._data.shape

    @property
    def width(self) -> int:
        """The width of the image in pixels."""
        return self._data.shape[1]

    @property
    def height(self) -> int:
        """The height of the image in pixels."""
        return self._data.shape[0]

    @property
    def size(self) -> tuple[int, int]:
        """The (width, height) of the image as a tuple."""
        return self.width, self.height

    @property
    def channels(self) -> int:
        """The number of channels in the image (1 for grayscale)."""
        return self._channels

    @property
    def format(self) -> ImageFormat:
        return self._format

    @property
    def has_alpha(self) -> bool:
        return self._format.has_alpha

    def view(self, region: Ellipsis | Region) -> Image:
        return Image(self[region], self.format)

    def crop(self, region: Ellipsis | Region) -> Image:
        return Image(self[region].copy(), self.format)


def calculate_content_bbox(image: Image) -> Region:
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
