from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from anicrop.enums import ImageFormat

try:
    from anicrop.native.color import (  # type: ignore[import-untyped]
        prgba_to_rgb as _cy_prgba_to_rgb,
    )
    from anicrop.native.color import (
        prgba_to_rgba as _cy_prgba_to_rgba,
    )
    from anicrop.native.color import (
        prgba_to_rgbx as _cy_prgba_to_rgbx,
    )
    from anicrop.native.color import (
        rgb_to_rgbx as _cy_rgb_to_rgbx,
    )
    from anicrop.native.color import (
        rgba_to_prgba as _cy_rgba_to_prgba,
    )
    from anicrop.native.color import (
        rgbx_to_rgb as _cy_rgbx_to_rgb,
    )

    _HAS_CY_COLOR = True
except ImportError:
    _cy_rgba_to_prgba = None
    _cy_prgba_to_rgba = None
    _cy_rgb_to_rgbx = None
    _cy_rgbx_to_rgb = None
    _cy_prgba_to_rgb = None
    _cy_prgba_to_rgbx = None
    _HAS_CY_COLOR = False

# =========================================================================
# Estratégias Atômicas de Conversão de Formatos
# =========================================================================


def _rgba_to_prgba(data: np.ndarray) -> np.ndarray:
    """Pré-multiplica os canais RGB pelo canal alfa (RGBA -> PRGBA)."""
    h, w = data.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    if _HAS_CY_COLOR and _cy_rgba_to_prgba is not None:
        _cy_rgba_to_prgba(data, out)
        return out

    alpha = data[..., 3:4].astype(np.float32) / 255.0
    rgb_f = data[..., :3].astype(np.float32)
    out[..., :3] = np.clip(np.round(rgb_f * alpha), 0, 255).astype(np.uint8)
    out[..., 3] = data[..., 3]
    return out


def _prgba_to_rgba(data: np.ndarray) -> np.ndarray:
    """Desmultiplica os canais RGB pelo canal alfa (PRGBA -> RGBA)."""
    h, w = data.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    if _HAS_CY_COLOR and _cy_prgba_to_rgba is not None:
        _cy_prgba_to_rgba(data, out)
        return out

    alpha = data[..., 3:4].astype(np.float32)
    safe_alpha = np.where(alpha == 0, 1.0, alpha)
    rgb_f = data[..., :3].astype(np.float32)
    out[..., :3] = np.clip(
        np.round((rgb_f * 255.0) / safe_alpha),
        0,
        255,
    ).astype(np.uint8)
    out[..., 3] = data[..., 3]
    return out


def _rgb_to_rgbx(data: np.ndarray) -> np.ndarray:
    """Adiciona um canal de padding de 32 bits (RGB -> RGBX)."""
    h, w = data.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    if _HAS_CY_COLOR and _cy_rgb_to_rgbx is not None:
        _cy_rgb_to_rgbx(data, out)
        return out

    out[..., :3] = data[..., :3]
    out[..., 3] = 255
    return out


def _rgbx_to_rgb(data: np.ndarray) -> np.ndarray:
    """Descarta o canal de padding de 32 bits (RGBX -> RGB)."""
    h, w = data.shape[:2]
    out = np.empty((h, w, 3), dtype=np.uint8)
    if _HAS_CY_COLOR and _cy_rgbx_to_rgb is not None:
        _cy_rgbx_to_rgb(data, out)
        return out

    return np.ascontiguousarray(data[..., :3])


def _rgb_to_rgba(data: np.ndarray) -> np.ndarray:
    """Adiciona um canal alfa totalmente opaco (RGB -> RGBA)."""
    return _rgb_to_rgbx(data)


def _rgba_to_rgb(data: np.ndarray) -> np.ndarray:
    """Descarta o canal alfa (RGBA -> RGB)."""
    return _rgbx_to_rgb(data)


def _rgb_to_prgba(data: np.ndarray) -> np.ndarray:
    """Converte RGB opaco para PRGBA com alfa opaco (RGB -> PRGBA)."""
    return _rgb_to_rgbx(data)


def _prgba_to_rgb(data: np.ndarray) -> np.ndarray:
    """Desmultiplica e descarta o canal alfa (PRGBA -> RGB)."""
    h, w = data.shape[:2]
    out = np.empty((h, w, 3), dtype=np.uint8)
    if _HAS_CY_COLOR and _cy_prgba_to_rgb is not None:
        _cy_prgba_to_rgb(data, out)
        return out

    alpha = data[..., 3:4].astype(np.float32)
    safe_alpha = np.where(alpha == 0, 1.0, alpha)
    rgb_f = data[..., :3].astype(np.float32)
    return np.clip(np.round((rgb_f * 255.0) / safe_alpha), 0, 255).astype(np.uint8)


def _rgbx_to_rgba(data: np.ndarray) -> np.ndarray:
    """Converte RGBX opaco para RGBA definindo o alfa como 255 (RGBX -> RGBA)."""
    out = data.copy()
    out[..., 3] = 255
    return out


def _rgba_to_rgbx(data: np.ndarray) -> np.ndarray:
    """Converte RGBA para RGBX definindo o 4º canal como padding (RGBA -> RGBX)."""
    out = data.copy()
    out[..., 3] = 255
    return out


def _rgbx_to_prgba(data: np.ndarray) -> np.ndarray:
    """Converte RGBX opaco para PRGBA com alfa opaco (RGBX -> PRGBA)."""
    out = data.copy()
    out[..., 3] = 255
    return out


def _prgba_to_rgbx(data: np.ndarray) -> np.ndarray:
    """Desmultiplica PRGBA e converte para RGBX (PRGBA -> RGBX)."""
    h, w = data.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    if _HAS_CY_COLOR and _cy_prgba_to_rgbx is not None:
        _cy_prgba_to_rgbx(data, out)
        return out

    alpha = data[..., 3:4].astype(np.float32)
    safe_alpha = np.where(alpha == 0, 1.0, alpha)
    rgb_f = data[..., :3].astype(np.float32)
    out[..., :3] = np.clip(
        np.round((rgb_f * 255.0) / safe_alpha),
        0,
        255,
    ).astype(np.uint8)
    out[..., 3] = 255
    return out


def _gray_to_rgb(data: np.ndarray) -> np.ndarray:
    """Expande escala de cinza para 3 canais RGB (GRAY -> RGB)."""
    src = data if data.ndim == 2 else data[..., 0]
    return cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)


def _rgb_to_gray(data: np.ndarray) -> np.ndarray:
    """Converte RGB para escala de cinza com peso perceptivo (RGB -> GRAY)."""
    gray = cv2.cvtColor(data[..., :3], cv2.COLOR_RGB2GRAY)
    return gray[..., np.newaxis]


def _gray_to_rgba(data: np.ndarray) -> np.ndarray:
    """Converte escala de cinza para RGBA com alfa opaco (GRAY -> RGBA)."""
    src = data if data.ndim == 2 else data[..., 0]
    return cv2.cvtColor(src, cv2.COLOR_GRAY2RGBA)


def _rgba_to_gray(data: np.ndarray) -> np.ndarray:
    """Converte RGBA para escala de cinza descartando o alfa (RGBA -> GRAY)."""
    return _rgb_to_gray(data[..., :3])


def _gray_to_prgba(data: np.ndarray) -> np.ndarray:
    """Converte escala de cinza para PRGBA com alfa opaco (GRAY -> PRGBA)."""
    src = data if data.ndim == 2 else data[..., 0]
    return cv2.cvtColor(src, cv2.COLOR_GRAY2RGBA)


def _prgba_to_gray(data: np.ndarray) -> np.ndarray:
    """Desmultiplica PRGBA e converte para escala de cinza (PRGBA -> GRAY)."""
    rgb = _prgba_to_rgb(data)
    return _rgb_to_gray(rgb)


def _gray_to_rgbx(data: np.ndarray) -> np.ndarray:
    """Converte escala de cinza para RGBX (GRAY -> RGBX)."""
    src = data if data.ndim == 2 else data[..., 0]
    return cv2.cvtColor(src, cv2.COLOR_GRAY2RGBA)


def _rgbx_to_gray(data: np.ndarray) -> np.ndarray:
    """Converte RGBX para escala de cinza (RGBX -> GRAY)."""
    return _rgb_to_gray(data[..., :3])


def _gray_to_gray_alpha(data: np.ndarray) -> np.ndarray:
    """Adiciona canal alfa opaco a escala de cinza (GRAY -> GRAY_ALPHA)."""
    h, w = data.shape[:2]
    out = np.empty((h, w, 2), dtype=np.uint8)
    out[..., 0] = data if data.ndim == 2 else data[..., 0]
    out[..., 1] = 255
    return out


def _gray_alpha_to_gray(data: np.ndarray) -> np.ndarray:
    """Descarta o canal alfa da escala de cinza (GRAY_ALPHA -> GRAY)."""
    return np.ascontiguousarray(data[..., 0:1])


def _gray_alpha_to_rgba(data: np.ndarray) -> np.ndarray:
    """Converte GRAY_ALPHA para RGBA preservando o alfa (GRAY_ALPHA -> RGBA)."""
    h, w = data.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., :3] = cv2.cvtColor(data[..., 0], cv2.COLOR_GRAY2RGB)
    out[..., 3] = data[..., 1]
    return out


def _rgba_to_gray_alpha(data: np.ndarray) -> np.ndarray:
    """Converte RGBA para GRAY_ALPHA preservando o alfa (RGBA -> GRAY_ALPHA)."""
    h, w = data.shape[:2]
    out = np.empty((h, w, 2), dtype=np.uint8)
    out[..., 0] = cv2.cvtColor(data[..., :3], cv2.COLOR_RGB2GRAY)
    out[..., 1] = data[..., 3]
    return out


def _gray_alpha_to_prgba(data: np.ndarray) -> np.ndarray:
    """Converte GRAY_ALPHA para PRGBA (GRAY_ALPHA -> PRGBA)."""
    h, w = data.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    alpha = data[..., 1:2].astype(np.float32) / 255.0
    gray_f = data[..., 0:1].astype(np.float32)
    premul_gray = np.clip(np.round(gray_f * alpha), 0, 255).astype(np.uint8)
    out[..., :3] = cv2.cvtColor(premul_gray[..., 0], cv2.COLOR_GRAY2RGB)
    out[..., 3] = data[..., 1]
    return out


def _prgba_to_gray_alpha(data: np.ndarray) -> np.ndarray:
    """Converte PRGBA para GRAY_ALPHA (PRGBA -> GRAY_ALPHA)."""
    rgba = _prgba_to_rgba(data)
    return _rgba_to_gray_alpha(rgba)


def _gray_alpha_to_rgb(data: np.ndarray) -> np.ndarray:
    """Descarta o alfa e expande para RGB (GRAY_ALPHA -> RGB)."""
    return cv2.cvtColor(data[..., 0], cv2.COLOR_GRAY2RGB)


def _rgb_to_gray_alpha(data: np.ndarray) -> np.ndarray:
    """Converte RGB para GRAY_ALPHA com alfa opaco (RGB -> GRAY_ALPHA)."""
    h, w = data.shape[:2]
    out = np.empty((h, w, 2), dtype=np.uint8)
    out[..., 0] = cv2.cvtColor(data[..., :3], cv2.COLOR_RGB2GRAY)
    out[..., 1] = 255
    return out


def _gray_alpha_to_rgbx(data: np.ndarray) -> np.ndarray:
    """Descarta o alfa e converte para RGBX (GRAY_ALPHA -> RGBX)."""
    src = data[..., 0]
    return cv2.cvtColor(src, cv2.COLOR_GRAY2RGBA)


def _rgbx_to_gray_alpha(data: np.ndarray) -> np.ndarray:
    """Converte RGBX para GRAY_ALPHA com alfa opaco (RGBX -> GRAY_ALPHA)."""
    return _rgb_to_gray_alpha(data[..., :3])


# =========================================================================
# Tabela de Despacho de Conversão de Formatos (Strategy Dispatch Table)
# =========================================================================

FormatConverter = Callable[[np.ndarray], np.ndarray]

FORMAT_CONVERTERS: dict[tuple[ImageFormat, ImageFormat], FormatConverter] = {
    # RGBA <-> PRGBA
    (ImageFormat.RGBA, ImageFormat.PRGBA): _rgba_to_prgba,
    (ImageFormat.PRGBA, ImageFormat.RGBA): _prgba_to_rgba,
    # RGB <-> RGBX
    (ImageFormat.RGB, ImageFormat.RGBX): _rgb_to_rgbx,
    (ImageFormat.RGBX, ImageFormat.RGB): _rgbx_to_rgb,
    # RGB <-> RGBA
    (ImageFormat.RGB, ImageFormat.RGBA): _rgb_to_rgba,
    (ImageFormat.RGBA, ImageFormat.RGB): _rgba_to_rgb,
    # RGB <-> PRGBA
    (ImageFormat.RGB, ImageFormat.PRGBA): _rgb_to_prgba,
    (ImageFormat.PRGBA, ImageFormat.RGB): _prgba_to_rgb,
    # RGBX <-> RGBA
    (ImageFormat.RGBX, ImageFormat.RGBA): _rgbx_to_rgba,
    (ImageFormat.RGBA, ImageFormat.RGBX): _rgba_to_rgbx,
    # RGBX <-> PRGBA
    (ImageFormat.RGBX, ImageFormat.PRGBA): _rgbx_to_prgba,
    (ImageFormat.PRGBA, ImageFormat.RGBX): _prgba_to_rgbx,
    # GRAY <-> RGB
    (ImageFormat.GRAY, ImageFormat.RGB): _gray_to_rgb,
    (ImageFormat.RGB, ImageFormat.GRAY): _rgb_to_gray,
    # GRAY <-> RGBA
    (ImageFormat.GRAY, ImageFormat.RGBA): _gray_to_rgba,
    (ImageFormat.RGBA, ImageFormat.GRAY): _rgba_to_gray,
    # GRAY <-> PRGBA
    (ImageFormat.GRAY, ImageFormat.PRGBA): _gray_to_prgba,
    (ImageFormat.PRGBA, ImageFormat.GRAY): _prgba_to_gray,
    # GRAY <-> RGBX
    (ImageFormat.GRAY, ImageFormat.RGBX): _gray_to_rgbx,
    (ImageFormat.RGBX, ImageFormat.GRAY): _rgbx_to_gray,
    # GRAY <-> GRAY_ALPHA
    (ImageFormat.GRAY, ImageFormat.GRAY_ALPHA): _gray_to_gray_alpha,
    (ImageFormat.GRAY_ALPHA, ImageFormat.GRAY): _gray_alpha_to_gray,
    # GRAY_ALPHA <-> RGBA
    (ImageFormat.GRAY_ALPHA, ImageFormat.RGBA): _gray_alpha_to_rgba,
    (ImageFormat.RGBA, ImageFormat.GRAY_ALPHA): _rgba_to_gray_alpha,
    # GRAY_ALPHA <-> PRGBA
    (ImageFormat.GRAY_ALPHA, ImageFormat.PRGBA): _gray_alpha_to_prgba,
    (ImageFormat.PRGBA, ImageFormat.GRAY_ALPHA): _prgba_to_gray_alpha,
    # GRAY_ALPHA <-> RGB
    (ImageFormat.GRAY_ALPHA, ImageFormat.RGB): _gray_alpha_to_rgb,
    (ImageFormat.RGB, ImageFormat.GRAY_ALPHA): _rgb_to_gray_alpha,
    # GRAY_ALPHA <-> RGBX
    (ImageFormat.GRAY_ALPHA, ImageFormat.RGBX): _gray_alpha_to_rgbx,
    (ImageFormat.RGBX, ImageFormat.GRAY_ALPHA): _rgbx_to_gray_alpha,
}


def convert_image_format(
    data: np.ndarray,
    src_fmt: ImageFormat,
    dst_fmt: ImageFormat,
) -> np.ndarray:
    """Converte a matriz NumPy de pixels entre dois formatos de imagem em tempo O(1)."""
    if src_fmt == dst_fmt:
        return data.copy()

    converter = FORMAT_CONVERTERS.get((src_fmt, dst_fmt))
    if converter is not None:
        return converter(data)

    raise ValueError(
        f"Conversão de formato não suportada de '{src_fmt}' para '{dst_fmt}'."
    )
