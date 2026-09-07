"""Provides the Image class, a wrapper for image data processing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import EllipsisType
from typing import Any, cast

import cv2
import numpy as np
from numpy import ndarray

from anicrop.buffer import ArrayBuffer, MMapBuffer
from anicrop.color import convert_image_format
from anicrop.config import config
from anicrop.enums import ImageFormat
from anicrop.interfaces.buffer import AbstractImageBuffer
from anicrop.interfaces.io import AbstractImageIO, SaveOptions
from anicrop.io.registry import get_backend
from anicrop.spatial import Region, Span


def set_memory_threshold(threshold_pixels: int | None) -> None:
    """Define o threshold global de pixels para alocação em RAM antes de usar paginação em disco.

    Passe None para desativar a paginação em disco e forçar 100% de alocação em memória RAM.
    """
    config.memory_threshold = threshold_pixels


def get_memory_threshold() -> int | None:
    """Retorna o threshold global de pixels configurado atualmente."""
    return config.memory_threshold


type ImageIndexer = Region | slice | tuple[Any, ...] | EllipsisType | int


class Image:
    """A wrapper around an AbstractImageBuffer to provide an image-centric API.

    This class facilitates spatial indexing using Region objects and offers
    convenient properties for accessing image dimensions (width, height, channels).
    It ensures that the underlying image data is a valid 2D or 3D array.
    """

    def __init__(
        self,
        image: AbstractImageBuffer | ndarray,
        image_format: ImageFormat,
        threshold_pixels: int | None | EllipsisType = ...,
    ):
        """Initializes the Image object.

        Args:
            image: An AbstractImageBuffer or 2D/3D NumPy ndarray.
            image_format: The color format of the image.
            threshold_pixels: Pixel threshold to offload ndarray to MMapBuffer in disk.

        Raises:
            TypeError: If image is not an AbstractImageBuffer or ndarray.
            ValueError: If the image array is not 2D/3D, has zero dimensions,
                        or has no channels in a 3D configuration.
        """
        if isinstance(image, AbstractImageBuffer):
            self._data = image
        elif isinstance(image, np.ndarray):
            if image.ndim not in (2, 3):
                raise ValueError("image array must be 2D or 3D")
            if image.shape[0] == 0 or image.shape[1] == 0:
                raise ValueError("image dimensions must be greater than zero")
            if image.ndim == 3 and image.shape[2] == 0:
                raise ValueError("image must have at least one channel")

            arr = image[..., np.newaxis] if image.ndim == 2 else image
            threshold = (
                config.memory_threshold if threshold_pixels is ... else threshold_pixels
            )
            if isinstance(image, np.memmap):
                self._data = MMapBuffer(cast(np.memmap, arr))
            elif threshold is not None and (arr.shape[0] * arr.shape[1] > threshold):
                self._data = MMapBuffer.from_array(arr)
            else:
                self._data = ArrayBuffer(arr)
        else:
            raise TypeError(
                f"image must be an AbstractImageBuffer or ndarray, got {type(image).__name__}"
            )

        if self._data.ndim not in (2, 3):
            raise ValueError("image array must be 2D or 3D")
        elif self._data.shape[0] == 0 or self._data.shape[1] == 0:
            raise ValueError("image dimensions must be greater than zero")
        elif self._data.ndim == 3 and self._data.shape[2] == 0:
            raise ValueError("image must have at least one channel")

        self._channels = self._data.shape[2] if self._data.ndim == 3 else 1
        self._format = image_format
        self._validate_format()

    @staticmethod
    def _normalize_key(
        key: ImageIndexer,
    ) -> tuple[slice | int | EllipsisType, ...] | slice | int | EllipsisType:
        """Normaliza chaves de fatiamento convertendo instâncias de Region em tuplas de slice."""
        if isinstance(key, Region):
            return key.to_slice()

        if isinstance(key, tuple):
            if any(isinstance(arg, Region) for arg in key[1:]):
                raise TypeError("Region argument is only valid at the first position")
            if isinstance(key[0], Region):
                return key[0].to_slice() + key[1:]

        return key

    def __getitem__(self, key: ImageIndexer) -> ndarray:
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
        return cast(ndarray, self._data[self._normalize_key(key)])

    def __setitem__(self, key: ImageIndexer, value: Any) -> None:
        """Sets a part of the image using indexing.

        Supports standard NumPy indexing and spatial indexing with a Region object.
        When a Region is used, it can be the sole index or the first element
        in a tuple for further channel/slice selection.

        Args:
            key: A Region object, a standard NumPy index, or a tuple
                 starting with a Region.
            value: The value or ndarray to assign to the specified slice.
        """
        self._data[self._normalize_key(key)] = value

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

            self[:y1, :] = fill_value
            self[y2:, :] = fill_value
            self[y1:y2, :x1] = fill_value
            self[y1:y2, x2:] = fill_value

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
    def dtype(self) -> np.dtype:
        """The data type (dtype) of the underlying image array."""
        return self._data.dtype

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Protocolo NumPy para conversão direta via np.asarray(image)."""
        return np.asarray(self._data, dtype=dtype)

    def to_dtype(self, target_dtype: Any) -> Image:
        """Converte a imagem para outro tipo de dado (dtype) de forma segura e não-destrutiva."""
        target_dt = np.dtype(target_dtype)
        if self.dtype == target_dt:
            return self

        arr = self[...]
        converted: np.ndarray
        if self.dtype == np.uint8 and target_dt == np.uint16:
            converted = arr.astype(np.uint16) * 257
        elif self.dtype == np.uint8 and np.issubdtype(target_dt, np.floating):
            converted = arr.astype(target_dt) / 255.0
        elif self.dtype == np.uint16 and target_dt == np.uint8:
            converted = (arr >> 8).astype(np.uint8)
        elif self.dtype == np.uint16 and np.issubdtype(target_dt, np.floating):
            converted = arr.astype(target_dt) / 65535.0
        elif np.issubdtype(self.dtype, np.floating) and target_dt == np.uint8:
            converted = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        elif np.issubdtype(self.dtype, np.floating) and target_dt == np.uint16:
            converted = np.clip(np.round(arr * 65535.0), 0, 65535).astype(np.uint16)
        else:
            converted = arr.astype(target_dt)

        return Image(converted, self.format)

    def to_uint8(self) -> Image:
        """Retorna uma nova Image convertida para uint8 com escala segura de bits."""
        return self.to_dtype(np.uint8)

    def get_lod(self, level: int) -> Image:
        """Retorna uma nova Image no nível de resolução solicitado (1/2^level)."""
        if level <= 0:
            return self

        if hasattr(self._data, "get_lod"):
            threshold = config.memory_threshold
            return Image(
                self._data.get_lod(level, threshold_pixels=threshold), self.format
            )

        factor = 2.0 ** (-level)
        new_w = max(1, int(self.width * factor))
        new_h = max(1, int(self.height * factor))
        return self.resize((new_w, new_h))

    @classmethod
    def new(
        cls,
        size: tuple[int | float, int | float] | Sequence[int | float],
        fmt: ImageFormat,
        color: int | float | tuple[int | float, ...] = 0,
        threshold_pixels: int | None | EllipsisType = ...,
        dtype: Any = ...,
    ) -> Image:
        """Creates a new Image with the specified dimensions, format and dtype.

        Uses MMapBuffer if threshold is configured and width * height > threshold,
        or NumPy ndarray (RAM) otherwise. Defaults to config.dtype if not specified.
        """
        threshold = (
            config.memory_threshold if threshold_pixels is ... else threshold_pixels
        )

        resolved_dtype = config.dtype if dtype is ... else np.dtype(dtype)
        width = int(round(size[0]))
        height = int(round(size[1]))
        channels = fmt.channels
        shape = (height, width, channels)

        if np.issubdtype(resolved_dtype, np.floating):
            alpha_default = 1.0
        elif resolved_dtype == np.uint16:
            alpha_default = 65535
        else:
            alpha_default = 255

        if threshold is not None and width * height > threshold:
            mmap_buf = MMapBuffer.create_empty(shape, dtype=resolved_dtype)
            if color != 0:
                if isinstance(color, (tuple, list)) and len(color) != channels:
                    if len(color) < channels:
                        color = tuple(color) + (alpha_default,) * (channels - len(color))
                    else:
                        color = tuple(color[:channels])
                mmap_buf[...] = color
                mmap_buf.flush()
            return cls(mmap_buf, fmt)

        if color == 0 or (isinstance(color, (tuple, list)) and not any(color)):
            buffer = np.zeros(shape, dtype=resolved_dtype)
        else:
            if isinstance(color, (tuple, list)) and len(color) != channels:
                if len(color) < channels:
                    color = tuple(color) + (alpha_default,) * (channels - len(color))
                else:
                    color = tuple(color[:channels])
            buffer = np.full(shape, color, dtype=resolved_dtype)
        return cls(buffer, fmt)

    def resize(self, target_size: tuple[int | float, int | float]) -> Image:
        """Redimensiona a imagem usando a fábrica inteligente Image.new."""
        new_w = int(round(target_size[0]))
        new_h = int(round(target_size[1]))
        img_data = self[...]
        resized_data = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if resized_data.ndim == 2:
            resized_data = resized_data[..., np.newaxis]

        new_img = Image.new((new_w, new_h), self._format, dtype=self.dtype)
        new_img[...] = resized_data
        return new_img

    def view(self, region: EllipsisType | Region = ...) -> Image:
        return Image(self[region], self.format)

    def crop(self, region: EllipsisType | Region = ...) -> Image:
        return Image(self[region].copy(), self.format)

    def to_format(self, target_format: ImageFormat) -> Image:
        """Converte a imagem para o formato especificado utilizando a tabela de estratégias de conversão."""
        if self.format == target_format:
            return Image(self[...].copy(), target_format)
        converted_data = convert_image_format(self[...], self.format, target_format)
        return Image(converted_data, target_format)

    def bgr(self, region: EllipsisType | Region = ...) -> np.ndarray:
        """Extrai a matriz NumPy da região convertida para o formato BGR/BGRA do OpenCV."""
        frame = self[region]

        if self.format == ImageFormat.RGBA:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        elif self.format == ImageFormat.PRGBA:
            rgba_data = convert_image_format(frame, ImageFormat.PRGBA, ImageFormat.RGBA)
            return cv2.cvtColor(rgba_data, cv2.COLOR_RGBA2BGRA)
        elif self.format in (ImageFormat.RGB, ImageFormat.RGBX):
            return cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)
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
        dtype: Any = ...,
    ) -> Image:
        file_path_str = str(file_path)
        io_backend = get_backend(backend)
        target_dtype = (
            config.dtype if dtype is ... else (np.dtype(dtype) if dtype is not None else None)
        )

        width, height = io_backend.get_size(file_path_str)
        target_w = roi.width if roi is not None else width
        target_h = roi.height if roi is not None else height
        effective_w = int(round(target_w / max(1, shrink)))
        effective_h = int(round(target_h / max(1, shrink)))

        threshold = config.memory_threshold
        if threshold is not None and (effective_w * effective_h) >= threshold:
            resolved_fmt = image_format or ImageFormat.RGBA
            data, resolved_fmt = io_backend.read_large(
                file_path_str, format=resolved_fmt
            )
            img = cls(data, resolved_fmt)
            if target_dtype is not None and img.dtype != target_dtype:
                return img.to_dtype(target_dtype)
            return img

        data, resolved_fmt, _ = io_backend.read(
            file_path_str,
            format=image_format,
            shrink=shrink,
            roi=roi,
        )
        img = cls(data, resolved_fmt)
        if target_dtype is not None and img.dtype != target_dtype:
            return img.to_dtype(target_dtype)
        return img


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
