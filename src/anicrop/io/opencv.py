from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image as PILImage

from anicrop.enums import ImageFormat
from anicrop.interfaces.io import AbstractImageIO, SaveOptions

if TYPE_CHECKING:
    import zarr
    from anicrop.spatial import Region


def _decode_raw(file_path: str | Path) -> tuple[np.ndarray, int]:
    """Decodifica o array cru do arquivo e identifica o número de canais nativos."""
    raw_data = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if raw_data is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem em: {file_path}")

    native_channels = 1 if raw_data.ndim == 2 else raw_data.shape[2]
    return raw_data, native_channels


def _auto_detect_format(data: np.ndarray, channels: int) -> tuple[np.ndarray, ImageFormat]:
    """Detecta o formato nativo da imagem crua e converte BGR para o padrão RGB do anicrop."""
    if channels == 1:
        return data[..., np.newaxis], ImageFormat.GRAY
    elif channels == 2:
        return data, ImageFormat.GRAY_ALPHA
    elif channels == 3:
        return cv2.cvtColor(data, cv2.COLOR_BGR2RGB), ImageFormat.RGB
    elif channels == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA), ImageFormat.RGBA
    else:
        return cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2RGB), ImageFormat.RGB


def _convert_to_requested_format(
    data: np.ndarray,
    channels: int,
    target_format: ImageFormat,
) -> np.ndarray:
    """Converte os dados lidos do OpenCV para o formato explicitamente solicitado."""
    h, w = data.shape[:2]

    if target_format == ImageFormat.RGBA:
        if channels == 1:
            rgb = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            return np.dstack([rgb, alpha])
        elif channels == 2:
            gray, alpha = data[:, :, 0], data[:, :, 1]
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            return np.dstack([rgb, alpha])
        elif channels == 3:
            return cv2.cvtColor(data, cv2.COLOR_BGR2RGBA)
        return cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)

    elif target_format == ImageFormat.RGB:
        if channels == 1:
            return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        elif channels == 2:
            return cv2.cvtColor(data[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif channels == 3:
            return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)

    elif target_format == ImageFormat.GRAY:
        if channels == 1:
            return data if data.ndim == 3 else data[..., np.newaxis]
        elif channels == 2:
            return data[:, :, 0:1]
        gray = cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2GRAY)
        return gray[..., np.newaxis]

    elif target_format == ImageFormat.GRAY_ALPHA:
        if channels == 1:
            gray = data if data.ndim == 2 else data[:, :, 0]
            alpha = np.full(gray.shape, 255, dtype=np.uint8)
            return np.dstack([gray, alpha])
        elif channels == 2:
            return data
        elif channels == 3:
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            alpha = np.full(gray.shape, 255, dtype=np.uint8)
            return np.dstack([gray, alpha])
        gray = cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2GRAY)
        return np.dstack([gray, data[:, :, 3]])

    return data


def _apply_shrink_and_roi(
    data: np.ndarray,
    orig_size: tuple[int, int],
    shrink: int,
    roi: Region | None,
) -> np.ndarray:
    """Aplica subamostragem (shrink) e recorte de região (ROI) aos pixels decodificados."""
    orig_w, orig_h = orig_size

    if shrink > 1:
        target_w = max(1, orig_w // shrink)
        target_h = max(1, orig_h // shrink)
        data = cv2.resize(data, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if data.ndim == 2:
            data = data[..., np.newaxis]

    if roi is not None:
        data = data[roi.y.start:roi.y.end, roi.x.start:roi.x.end]

    return data


def _flatten_alpha_to_background(
    img_arr: np.ndarray,
    format: ImageFormat,
    bg_color: tuple[int, ...],
) -> np.ndarray:
    """Mescla o canal alfa contra a cor sólida de fundo para formatos sem suporte a transparência."""
    if format == ImageFormat.RGBA:
        alpha = img_arr[..., 3:4].astype(np.float32) / 255.0
        bg = np.array(bg_color[:3], dtype=np.float32)
        rgb = img_arr[..., :3].astype(np.float32)
        flattened = (rgb * alpha + bg * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(flattened, cv2.COLOR_RGB2BGR)

    elif format == ImageFormat.GRAY_ALPHA:
        alpha = img_arr[..., 1:2].astype(np.float32) / 255.0
        bg_gray = float(bg_color[0])
        gray = img_arr[..., 0:1].astype(np.float32)
        flattened = (gray * alpha + bg_gray * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        return flattened[:, :, 0]

    return img_arr


def _prepare_bgr_for_export(
    data: np.ndarray,
    format: ImageFormat,
    ext: str,
    options: SaveOptions,
) -> np.ndarray:
    """Converte a matriz do anicrop para o layout de canais esperado pelo OpenCV."""
    is_jpeg = ext in ('.jpg', '.jpeg')

    if is_jpeg and format in (ImageFormat.RGBA, ImageFormat.GRAY_ALPHA):
        return _flatten_alpha_to_background(data, format, options.bg_color)

    if format == ImageFormat.RGBA:
        return cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA)
    elif format == ImageFormat.RGB:
        return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    elif format == ImageFormat.GRAY_ALPHA:
        gray_bgr = cv2.cvtColor(data[..., 0], cv2.COLOR_GRAY2BGR)
        return np.dstack([gray_bgr, data[..., 1]])
    elif format == ImageFormat.GRAY:
        return data[..., 0] if data.ndim == 3 else data

    return data


def _build_opencv_params(ext: str, options: SaveOptions) -> list[int]:
    """Mapeia as SaveOptions para a lista de parâmetros de gravação do OpenCV."""
    params: list[int] = []
    if ext in ('.jpg', '.jpeg'):
        params.extend([cv2.IMWRITE_JPEG_QUALITY, options.quality])
    elif ext == '.png':
        params.extend([cv2.IMWRITE_PNG_COMPRESSION, options.compression_level])
    elif ext == '.webp':
        quality_val = 101 if options.lossless else options.quality
        params.extend([cv2.IMWRITE_WEBP_QUALITY, quality_val])
    return params


class OpenCVBackend(AbstractImageIO):
    """Backend modular de leitura e escrita de imagens baseado no OpenCV."""

    def get_size(self, file_path: str | Path) -> tuple[int, int]:
        """Extrai as dimensões lendo apenas o cabeçalho do arquivo."""
        with PILImage.open(str(file_path)) as pil_img:
            return pil_img.size

    def read(
        self,
        file_path: str | Path,
        format: ImageFormat | None = None,
        shrink: int = 1,
        roi: Region | None = None,
    ) -> tuple[np.ndarray, ImageFormat, tuple[int, int]]:
        """Lê e decodifica a imagem a partir do disco."""
        raw_data, native_channels = _decode_raw(file_path)
        orig_size = (raw_data.shape[1], raw_data.shape[0])

        if format is None:
            data, resolved_format = _auto_detect_format(raw_data, native_channels)
        else:
            data = _convert_to_requested_format(raw_data, native_channels, format)
            resolved_format = format

        data = _apply_shrink_and_roi(data, orig_size, shrink, roi)
        return data, resolved_format, orig_size

    def write(
        self,
        file_path: str | Path,
        data: np.ndarray | zarr.Array,
        format: ImageFormat,
        options: SaveOptions | None = None,
    ) -> None:
        """Codifica e grava a matriz de pixels no disco."""
        options = options or SaveOptions()
        path = Path(file_path)
        ext = path.suffix.lower()

        img_arr = np.asarray(data)
        bgr_data = _prepare_bgr_for_export(img_arr, format, ext, options)
        params = _build_opencv_params(ext, options)

        success = cv2.imwrite(str(path), bgr_data, params)
        if not success:
            raise IOError(f"Falha ao salvar imagem em: {file_path}")
