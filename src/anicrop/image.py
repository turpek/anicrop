"""Provides the Image class, a wrapper for image data processing."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from types import EllipsisType
from typing import Any, cast

import cv2
import numpy as np
import zarr
from numpy import ndarray
from PIL import Image as PILImage

from anicrop.enums import ImageFormat
from anicrop.interfaces.io import AbstractImageIO, SaveOptions
from anicrop.io.registry import get_backend
from anicrop.persistence.manager import manager_global
from anicrop.spatial import Region, Span


class Image:
    """A wrapper around a NumPy ndarray to provide an image-centric API.

    This class facilitates spatial indexing using Region objects and offers
    convenient properties for accessing image dimensions (width, height, channels).
    It ensures that the underlying image data is a valid 2D or 3D array.
    """

    def __init__(self, image: ndarray | zarr.Array, image_format: ImageFormat):
        """Initializes the Image object.

        Args:
            image: A 2D (grayscale) or 3D (color) NumPy ndarray or Zarr Array.

        Raises:
            ValueError: If the image array is not 2D/3D, has zero dimensions,
                        or has no channels in a 3D configuration.
        """
        if image.ndim not in (2, 3):
            raise ValueError("image array must be 2D or 3D")

        elif image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image dimensions must be greater than zero")
        elif isinstance(image, np.ndarray) and image.ndim == 2:
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
            slice(int(round(region.y.start)), int(round(region.y.end))),
            slice(int(round(region.x.start)), int(round(region.x.end))),
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
        return cast(ndarray, self._data[self.__to_indexer(key)])

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

    def clear_rect(
        self,
        region: Region,
        fill_value: int | float | tuple[int, ...] | np.ndarray = 0,
        invert: bool = False,
    ) -> bool:
        """Limpa ou preenche uma região retangular da imagem.

        Args:
            region: A região espacial retangular.
            fill_value: O valor de preenchimento (padrão 0).
            invert: Se False (padrão), preenche a área DENTRO da região.
                    Se True, preenche a área FORA da região (inversão da seleção).

        Returns:
            True se os pixels foram alterados, False se a região não intersecta a imagem.
        """
        canvas_region = Region.from_size(self.width, self.height)
        if not canvas_region.overlaps(region):
            return False

        clipped = canvas_region & region

        if not invert:
            self[clipped] = fill_value
        else:
            x1, y1 = clipped.top_left.to_int()
            x2, y2 = clipped.bottom_right.to_int()

            self._data[:y1, :] = fill_value
            self._data[y2:, :] = fill_value
            self._data[y1:y2, :x1] = fill_value
            self._data[y1:y2, x2:] = fill_value

        return True

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

    @property
    def is_zarr(self) -> bool:
        """Indica se os dados da imagem estão armazenados em um array Zarr."""
        return not isinstance(self._data, np.ndarray)

    @classmethod
    def new(
        cls,
        size: tuple[int | float, int | float] | Sequence[int | float],
        fmt: ImageFormat,
        color: int | tuple[int, ...] = 0,
        threshold_pixels: int = 4096 * 4096,
    ) -> Image:
        """Creates a new Image with the specified dimensions and format.

        Uses Zarr if width * height > threshold_pixels, or NumPy ndarray otherwise.
        """
        width = int(round(size[0]))
        height = int(round(size[1]))
        channels = fmt.channels
        shape = (height, width, channels)

        if width * height > threshold_pixels:
            zarr_dir = manager_global.workspace_path / f"{uuid.uuid4().hex}.zarr"

            zarr_chunks = (min(512, height), min(512, width), channels)
            z_arr = zarr.open_array(
                str(zarr_dir),
                mode="w",
                shape=shape,
                chunks=zarr_chunks,
                dtype=np.uint8,
            )
            if color != 0:
                z_arr[...] = color
            return cls(z_arr, fmt)

        if color == 0 or (isinstance(color, (tuple, list)) and not any(color)):
            buffer = np.zeros(shape, dtype=np.uint8)
        else:
            if isinstance(color, (tuple, list)) and len(color) != channels:
                if len(color) < channels:
                    color = tuple(color) + (255,) * (channels - len(color))
                else:
                    color = tuple(color[:channels])
            buffer = np.full(shape, color, dtype=np.uint8)
        return cls(buffer, fmt)

    def resize(self, target_size: tuple[int | float, int | float]) -> Image:
        """Redimensiona a imagem usando a fábrica inteligente Image.new."""
        new_w = int(round(target_size[0]))
        new_h = int(round(target_size[1]))
        img_data = self[...]
        resized_data = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if resized_data.ndim == 2:
            resized_data = resized_data[..., np.newaxis]

        new_img = Image.new((new_w, new_h), self._format)
        new_img[...] = resized_data
        return new_img

    def view(self, region: EllipsisType | Region = ...) -> Image:
        return Image(self[region], self.format)

    def crop(self, region: EllipsisType | Region = ...) -> Image:
        return Image(self[region].copy(), self.format)

    def bgr(self, region: EllipsisType | Region = ...) -> np.ndarray:
        """Extrai a matriz NumPy da região convertida para o formato BGR/BGRA do OpenCV."""
        frame = self[region]

        if self.format == ImageFormat.RGBA:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        elif self.format == ImageFormat.RGB:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.format == ImageFormat.GRAY_ALPHA:
            gray_bgr = cv2.cvtColor(frame[..., 0], cv2.COLOR_GRAY2BGR)
            return np.dstack([gray_bgr, frame[..., 1]])
        elif self.format == ImageFormat.GRAY:
            return frame[..., 0] if frame.ndim == 3 else frame
        return frame

    def save(
        self,
        file_path: str | Path,
        options: SaveOptions | None = None,
        backend: AbstractImageIO | str | None = None,
    ) -> None:
        """Salva a imagem no disco no caminho especificado."""
        io_backend = get_backend(backend)
        io_backend.write(file_path, self._data, self.format, options=options)

    @classmethod
    def open(
        cls,
        file_path: str | Path,
        image_format: ImageFormat | None = None,
        backend: AbstractImageIO | str | None = None,
        shrink: int = 1,
        roi: Region | None = None,
    ) -> Image:
        file_path_str = str(file_path)
        io_backend = get_backend(backend)

        width, height = io_backend.get_size(file_path_str)
        if width >= 8192 or height >= 8192:
            resolved_fmt = image_format or ImageFormat.RGBA
            return cls._open_with_pillow_zarr(file_path_str, resolved_fmt)

        data, resolved_fmt, _ = io_backend.read(
            file_path_str,
            format=image_format,
            shrink=shrink,
            roi=roi,
        )
        return cls(data, resolved_fmt)

    @classmethod
    def _open_with_pillow_zarr(cls, file_path: str, image_format: ImageFormat) -> Image:
        mode_map = {
            ImageFormat.GRAY: "L",
            ImageFormat.GRAY_ALPHA: "LA",
            ImageFormat.RGB: "RGB",
            ImageFormat.RGBA: "RGBA",
            ImageFormat.CMYK: "CMYK",
        }
        mode = mode_map.get(image_format)

        zarr_dir = manager_global.workspace_path / f"{uuid.uuid4().hex}.zarr"

        with PILImage.open(file_path) as opened_img:
            pil_img = opened_img.convert(mode) if mode else opened_img
            width, height = pil_img.size
            channels = image_format.channels

            zarr_shape = (height, width, channels)
            zarr_chunks = (512, 512, channels)

            z_arr = zarr.open_array(
                str(zarr_dir),
                mode="w",
                shape=zarr_shape,
                chunks=zarr_chunks,
                dtype=np.uint8,
            )

            chunk_size = 512
            for y in range(0, height, chunk_size):
                for x in range(0, width, chunk_size):
                    y_end = min(y + chunk_size, height)
                    x_end = min(x + chunk_size, width)
                    box = (x, y, x_end, y_end)
                    tile = pil_img.crop(box)
                    tile_np = np.array(tile)

                    if tile_np.ndim == 2:
                        tile_np = tile_np[..., np.newaxis]

                    z_arr[y:y_end, x:x_end] = tile_np

        return cls(zarr.open_array(str(zarr_dir), mode="r"), image_format)


def calculate_content_rect(image: Image) -> Region:
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
    rows = np.any(alpha > 0, axis=1)
    if not np.any(rows):
        raise ValueError("EditLayer cannot be created from a fully transparent image.")

    cols = np.any(alpha > 0, axis=0)

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    start_y, end_y = int(row_indices[0]), int(row_indices[-1])
    start_x, end_x = int(col_indices[0]), int(col_indices[-1])

    width = end_x - start_x + 1
    height = end_y - start_y + 1
    return Region(Span(start_x, width), Span(start_y, height))
