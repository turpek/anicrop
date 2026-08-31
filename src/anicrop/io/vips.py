from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

os.environ.setdefault("VIPS_WARNING", "0")

try:
    _stderr_fd = os.dup(2)
    with open(os.devnull, "w") as _devnull:
        os.dup2(_devnull.fileno(), 2)
        try:
            import pyvips  # type: ignore[import-untyped]
        finally:
            os.dup2(_stderr_fd, 2)
            os.close(_stderr_fd)
except Exception:
    pyvips = None

from anicrop.color import convert_image_format
from anicrop.enums import ImageFormat
from anicrop.interfaces.io import AbstractImageIO, SaveOptions

if TYPE_CHECKING:
    import zarr

    from anicrop.spatial import Region


def is_vips_available() -> bool:
    """Retorna True se pyvips estiver instalado e pronto para uso."""
    return pyvips is not None


def _vips_decode(file_path: str | Path, shrink: int) -> pyvips.Image:
    """Decodifica a imagem usando libvips com suporte a shrink nativo."""
    path_str = str(file_path)
    if shrink > 1:
        try:
            return pyvips.Image.new_from_file(path_str, shrink=shrink)
        except Exception:
            img = pyvips.Image.new_from_file(path_str, access="sequential")
            return img.shrink(shrink, shrink)
    return pyvips.Image.new_from_file(path_str, access="sequential")


def _vips_auto_detect_format(vips_img: pyvips.Image) -> tuple[pyvips.Image, ImageFormat]:
    """Detecta o formato nativo e ajusta o espaço de cor para sRGB se necessário."""
    bands = vips_img.bands

    if bands == 1:
        return vips_img, ImageFormat.GRAY
    elif bands == 2:
        return vips_img, ImageFormat.GRAY_ALPHA
    elif bands == 3:
        if vips_img.interpretation != "srgb":
            vips_img = vips_img.colourspace("srgb")
        return vips_img, ImageFormat.RGB
    elif bands == 4:
        return vips_img, ImageFormat.RGBA
    else:
        # Imagens multiespectrais/CMYK
        vips_img = vips_img.colourspace("srgb")
        return vips_img, ImageFormat.RGB


def _vips_convert_to_requested_format(
    vips_img: pyvips.Image,
    target_format: ImageFormat,
) -> pyvips.Image:
    """Converte o pipeline da libvips para o formato explicitamente solicitado."""
    bands = vips_img.bands

    if target_format == ImageFormat.RGBA:
        if bands == 1:
            return vips_img.colourspace("srgb").bandjoin(255)
        elif bands == 2:
            gray = vips_img[0].colourspace("srgb")
            return gray.bandjoin(vips_img[1])
        elif bands == 3:
            return vips_img.bandjoin(255)
        return vips_img

    elif target_format == ImageFormat.RGB:
        if bands == 1:
            return vips_img.colourspace("srgb")
        elif bands == 2:
            return vips_img[0].colourspace("srgb")
        elif bands == 3:
            return vips_img
        return vips_img.extract_band(0, n=3)

    elif target_format == ImageFormat.GRAY:
        if bands == 1:
            return vips_img
        elif bands == 2:
            return vips_img.extract_band(0)
        return vips_img.colourspace("b-w").extract_band(0)

    elif target_format == ImageFormat.GRAY_ALPHA:
        if bands == 1:
            return vips_img.bandjoin(255)
        elif bands == 2:
            return vips_img
        gray = vips_img.colourspace("b-w").extract_band(0)
        alpha = (
            vips_img[3]
            if bands == 4
            else (pyvips.Image.black(vips_img.width, vips_img.height) + 255)
        )
        return gray.bandjoin(alpha)

    return vips_img


def _vips_to_numpy(vips_img: pyvips.Image) -> np.ndarray:
    """Converte a imagem libvips para ndarray NumPy sem cópias redundantes."""
    mem = vips_img.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(
        (vips_img.height, vips_img.width, vips_img.bands)
    )
    if arr.ndim == 2:
        return arr[..., np.newaxis]
    return arr


def _numpy_to_vips(data: np.ndarray, format: ImageFormat) -> pyvips.Image:
    """Cria uma imagem libvips a partir de uma matriz NumPy em memória."""
    h, w = data.shape[:2]
    bands = format.channels
    contiguous = np.ascontiguousarray(data)
    return pyvips.Image.new_from_memory(contiguous.data, w, h, bands, "uchar")


def _vips_save_file(
    vips_img: pyvips.Image,
    file_path: Path,
    format: ImageFormat,
    options: SaveOptions,
) -> None:
    """Grava o arquivo no disco com as opções de qualidade e compressão configuradas."""
    ext = file_path.suffix.lower()
    path_str = str(file_path)

    if ext in (".jpg", ".jpeg"):
        if format in (ImageFormat.RGBA, ImageFormat.GRAY_ALPHA) or vips_img.hasalpha():
            vips_img = vips_img.flatten(background=list(options.bg_color[:3]))
        vips_img.jpegsave(path_str, Q=options.quality, strip=options.strip_metadata)
    elif ext == ".png":
        vips_img.pngsave(
            path_str, compression=options.compression_level, strip=options.strip_metadata
        )
    elif ext == ".webp":
        vips_img.webpsave(
            path_str,
            Q=options.quality,
            lossless=options.lossless,
            strip=options.strip_metadata,
        )
    else:
        vips_img.write_to_file(path_str)


class PyvipsBackend(AbstractImageIO):
    """Backend de leitura e escrita de imagens de alta performance baseado na libvips."""

    def __init__(self) -> None:
        if pyvips is None:
            raise ImportError(
                "pyvips não está instalado. Instale com 'uv add pyvips' para utilizar este backend."
            )

    def get_size(self, file_path: str | Path) -> tuple[int, int]:
        """Extrai as dimensões da imagem lendo apenas o cabeçalho (sem decodificar pixels)."""
        vips_img = pyvips.Image.new_from_file(str(file_path), access="sequential")
        return (vips_img.width, vips_img.height)

    def read(
        self,
        file_path: str | Path,
        format: ImageFormat | None = None,
        shrink: int = 1,
        roi: Region | None = None,
    ) -> tuple[np.ndarray, ImageFormat, tuple[int, int]]:
        """Lê e decodifica a imagem com suporte a streaming e shrink nativo."""
        path_str = str(file_path)
        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"Não foi possível carregar a imagem em: {file_path}"
            )

        # Obtém tamanho original
        orig_img = pyvips.Image.new_from_file(path_str, access="sequential")
        orig_size = (orig_img.width, orig_img.height)

        vips_img = _vips_decode(path_str, shrink)

        if format is None:
            vips_img, resolved_format = _vips_auto_detect_format(vips_img)
        elif format in (ImageFormat.PRGBA, ImageFormat.RGBX):
            base_req = (
                ImageFormat.RGBA if format == ImageFormat.PRGBA else ImageFormat.RGB
            )
            vips_img = _vips_convert_to_requested_format(vips_img, base_req)
            resolved_format = base_req
        else:
            vips_img = _vips_convert_to_requested_format(vips_img, format)
            resolved_format = format

        if roi is not None:
            vips_img = vips_img.crop(roi.x.start, roi.y.start, roi.width, roi.height)

        data = _vips_to_numpy(vips_img)

        if format == ImageFormat.PRGBA:
            data = convert_image_format(data, ImageFormat.RGBA, ImageFormat.PRGBA)
            resolved_format = ImageFormat.PRGBA
        elif format == ImageFormat.RGBX:
            data = convert_image_format(data, ImageFormat.RGB, ImageFormat.RGBX)
            resolved_format = ImageFormat.RGBX

        return data, resolved_format, orig_size

    def write(
        self,
        file_path: str | Path,
        data: np.ndarray | zarr.Array,
        format: ImageFormat,
        options: SaveOptions | None = None,
    ) -> None:
        """Codifica e grava a imagem no disco usando libvips multithread."""
        options = options or SaveOptions()
        path = Path(file_path)

        img_arr = np.asarray(data)

        if format == ImageFormat.PRGBA:
            img_arr = convert_image_format(img_arr, ImageFormat.PRGBA, ImageFormat.RGBA)
            format = ImageFormat.RGBA
        elif format == ImageFormat.RGBX:
            img_arr = img_arr[..., :3]
            format = ImageFormat.RGB

        vips_img = _numpy_to_vips(img_arr, format)
        _vips_save_file(vips_img, path, format, options)

    def read_large(
        self,
        file_path: str | Path,
        format: ImageFormat | None = None,
    ) -> tuple[Any, ImageFormat]:
        """Abre imagens de altíssima resolução (>=8192px) via streaming sob demanda em C."""
        resolved_fmt = format or ImageFormat.RGBA
        stream_buf = VipsStreamingBuffer(file_path, target_format=resolved_fmt)
        return stream_buf, resolved_fmt


class VipsStreamingBuffer:
    """Buffer de streaming sob demanda baseado em libvips para imagens gigantes."""

    def __init__(
        self, file_path: str | Path, target_format: ImageFormat = ImageFormat.RGBA
    ) -> None:
        if pyvips is None:
            raise RuntimeError("pyvips não está disponível no sistema.")

        self._path = str(file_path)
        self._target_format = target_format
        self._vimg = pyvips.Image.new_from_file(self._path, access="random")

        # Converte para sRGB se necessário
        if self._vimg.bands >= 3 and self._vimg.interpretation != "srgb":
            self._vimg = self._vimg.colourspace("srgb")

        # Se o formato alvo pedir RGBA e a imagem tiver 3 bandas, adiciona canal alfa
        if target_format == ImageFormat.RGBA and self._vimg.bands == 3:
            alpha = (pyvips.Image.black(self._vimg.width, self._vimg.height) + 255).cast(
                "uchar"
            )
            self._vimg = self._vimg.bandjoin(alpha)
        elif target_format == ImageFormat.RGB and self._vimg.bands == 4:
            self._vimg = self._vimg.extract_band(0, n=3)

        self.shape = (self._vimg.height, self._vimg.width, self._vimg.bands)
        self.dtype = np.dtype(np.uint8)
        self.ndim = 3

    @property
    def size(self) -> tuple[int, int]:
        return (self._vimg.width, self._vimg.height)

    def __getitem__(self, key: Any) -> np.ndarray:
        from anicrop.spatial import Region

        if isinstance(key, Region):
            x = int(round(key.x.start))
            y = int(round(key.y.start))
            w = int(round(key.width))
            h = int(round(key.height))
        elif isinstance(key, tuple) and len(key) >= 2:
            slice_y, slice_x = key[0], key[1]
            y = slice_y.start or 0
            x = slice_x.start or 0
            h = (slice_y.stop or self.shape[0]) - y
            w = (slice_x.stop or self.shape[1]) - x
        elif key is Ellipsis:
            mem = self._vimg.write_to_memory()
            return np.frombuffer(mem, dtype=np.uint8).reshape(self.shape)
        else:
            raise TypeError(
                f"Tipo de índice não suportado no VipsStreamingBuffer: {type(key)}"
            )

        x = max(0, min(x, self._vimg.width - 1))
        y = max(0, min(y, self._vimg.height - 1))
        w = max(1, min(w, self._vimg.width - x))
        h = max(1, min(h, self._vimg.height - y))

        cropped = self._vimg.crop(x, y, w, h)
        mem = cropped.write_to_memory()
        return np.frombuffer(mem, dtype=np.uint8).reshape((h, w, self._vimg.bands))
