from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import os
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
        else:
            vips_img = _vips_convert_to_requested_format(vips_img, format)
            resolved_format = format

        if roi is not None:
            vips_img = vips_img.crop(roi.x.start, roi.y.start, roi.width, roi.height)

        data = _vips_to_numpy(vips_img)
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
        vips_img = _numpy_to_vips(img_arr, format)
        _vips_save_file(vips_img, path, format, options)
