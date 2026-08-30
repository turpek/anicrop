from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from anicrop.enums import ImageFormat

# =========================================================================
# Estratégias Atômicas de Conversão de Formatos
# =========================================================================


def _rgba_to_prgba(data: np.ndarray) -> np.ndarray:
    """Pré-multiplica os canais RGB pelo canal alfa (RGBA -> PRGBA)."""
    f_data = data.astype(np.float32)
    alpha = f_data[..., 3:4] / 255.0
    premul_rgb = np.clip(np.round(f_data[..., :3] * alpha), 0, 255).astype(np.uint8)
    return np.concatenate([premul_rgb, data[..., 3:4]], axis=-1)


def _prgba_to_rgba(data: np.ndarray) -> np.ndarray:
    """Desmultiplica os canais RGB pelo canal alfa (PRGBA -> RGBA)."""
    f_data = data.astype(np.float32)
    alpha = f_data[..., 3:4]
    alpha_mask = alpha[..., 0] > 0
    rgb = np.zeros_like(f_data[..., :3])
    rgb[alpha_mask] = np.clip(
        np.round((f_data[..., :3][alpha_mask] * 255.0) / alpha[alpha_mask]),
        0,
        255,
    )
    return np.concatenate([rgb.astype(np.uint8), data[..., 3:4]], axis=-1)


def _rgb_to_rgbx(data: np.ndarray) -> np.ndarray:
    """Adiciona um canal de padding de 32 bits (RGB -> RGBX)."""
    pad = np.full((*data.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([data[..., :3], pad], axis=-1)


def _rgbx_to_rgb(data: np.ndarray) -> np.ndarray:
    """Descarta o canal de padding de 32 bits (RGBX -> RGB)."""
    return data[..., :3].copy()


def _rgb_to_rgba(data: np.ndarray) -> np.ndarray:
    """Adiciona um canal alfa totalmente opaco (RGB -> RGBA)."""
    alpha = np.full((*data.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([data[..., :3], alpha], axis=-1)


def _rgba_to_rgb(data: np.ndarray) -> np.ndarray:
    """Descarta o canal alfa (RGBA -> RGB)."""
    return data[..., :3].copy()


def _rgb_to_prgba(data: np.ndarray) -> np.ndarray:
    """Converte RGB opaco para PRGBA com alfa opaco (RGB -> PRGBA)."""
    alpha = np.full((*data.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([data[..., :3], alpha], axis=-1)


def _prgba_to_rgb(data: np.ndarray) -> np.ndarray:
    """Desmultiplica e descarta o canal alfa (PRGBA -> RGB)."""
    rgba = _prgba_to_rgba(data)
    return rgba[..., :3].copy()


def _rgbx_to_rgba(data: np.ndarray) -> np.ndarray:
    """Converte RGBX opaco para RGBA definindo o alfa como 255 (RGBX -> RGBA)."""
    alpha = np.full((*data.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([data[..., :3], alpha], axis=-1)


def _rgba_to_rgbx(data: np.ndarray) -> np.ndarray:
    """Converte RGBA para RGBX definindo o 4º canal como padding (RGBA -> RGBX)."""
    pad = np.full((*data.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([data[..., :3], pad], axis=-1)


def _rgbx_to_prgba(data: np.ndarray) -> np.ndarray:
    """Converte RGBX opaco para PRGBA com alfa opaco (RGBX -> PRGBA)."""
    alpha = np.full((*data.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([data[..., :3], alpha], axis=-1)


def _prgba_to_rgbx(data: np.ndarray) -> np.ndarray:
    """Desmultiplica PRGBA e converte para RGBX (PRGBA -> RGBX)."""
    rgb = _prgba_to_rgb(data)
    return _rgb_to_rgbx(rgb)


def _gray_to_rgb(data: np.ndarray) -> np.ndarray:
    """Expande escala de cinza para 3 canais RGB (GRAY -> RGB)."""
    src = data if data.ndim == 3 else data[..., np.newaxis]
    return np.repeat(src[..., 0:1], 3, axis=-1)


def _rgb_to_gray(data: np.ndarray) -> np.ndarray:
    """Converte RGB para escala de cinza com peso perceptivo (RGB -> GRAY)."""
    gray = cv2.cvtColor(data[..., :3], cv2.COLOR_RGB2GRAY)
    return gray[..., np.newaxis]


def _gray_to_rgba(data: np.ndarray) -> np.ndarray:
    """Converte escala de cinza para RGBA com alfa opaco (GRAY -> RGBA)."""
    rgb = _gray_to_rgb(data)
    return _rgb_to_rgba(rgb)


def _rgba_to_gray(data: np.ndarray) -> np.ndarray:
    """Converte RGBA para escala de cinza descartando o alfa (RGBA -> GRAY)."""
    return _rgb_to_gray(data[..., :3])


def _gray_to_prgba(data: np.ndarray) -> np.ndarray:
    """Converte escala de cinza para PRGBA com alfa opaco (GRAY -> PRGBA)."""
    rgb = _gray_to_rgb(data)
    return _rgb_to_prgba(rgb)


def _prgba_to_gray(data: np.ndarray) -> np.ndarray:
    """Desmultiplica PRGBA e converte para escala de cinza (PRGBA -> GRAY)."""
    rgb = _prgba_to_rgb(data)
    return _rgb_to_gray(rgb)


def _gray_to_rgbx(data: np.ndarray) -> np.ndarray:
    """Converte escala de cinza para RGBX (GRAY -> RGBX)."""
    rgb = _gray_to_rgb(data)
    return _rgb_to_rgbx(rgb)


def _rgbx_to_gray(data: np.ndarray) -> np.ndarray:
    """Converte RGBX para escala de cinza (RGBX -> GRAY)."""
    return _rgb_to_gray(data[..., :3])


def _gray_to_gray_alpha(data: np.ndarray) -> np.ndarray:
    """Adiciona canal alfa opaco a escala de cinza (GRAY -> GRAY_ALPHA)."""
    src = data if data.ndim == 3 else data[..., np.newaxis]
    alpha = np.full((*data.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([src[..., 0:1], alpha], axis=-1)


def _gray_alpha_to_gray(data: np.ndarray) -> np.ndarray:
    """Descarta o canal alfa da escala de cinza (GRAY_ALPHA -> GRAY)."""
    return data[..., 0:1].copy()


def _gray_alpha_to_rgba(data: np.ndarray) -> np.ndarray:
    """Converte GRAY_ALPHA para RGBA preservando o alfa (GRAY_ALPHA -> RGBA)."""
    rgb = np.repeat(data[..., 0:1], 3, axis=-1)
    return np.concatenate([rgb, data[..., 1:2]], axis=-1)


def _rgba_to_gray_alpha(data: np.ndarray) -> np.ndarray:
    """Converte RGBA para GRAY_ALPHA preservando o alfa (RGBA -> GRAY_ALPHA)."""
    gray = _rgb_to_gray(data[..., :3])
    return np.concatenate([gray, data[..., 3:4]], axis=-1)


def _gray_alpha_to_prgba(data: np.ndarray) -> np.ndarray:
    """Converte GRAY_ALPHA para PRGBA (GRAY_ALPHA -> PRGBA)."""
    rgba = _gray_alpha_to_rgba(data)
    return _rgba_to_prgba(rgba)


def _prgba_to_gray_alpha(data: np.ndarray) -> np.ndarray:
    """Converte PRGBA para GRAY_ALPHA (PRGBA -> GRAY_ALPHA)."""
    rgba = _prgba_to_rgba(data)
    return _rgba_to_gray_alpha(rgba)


def _gray_alpha_to_rgb(data: np.ndarray) -> np.ndarray:
    """Descarta o alfa e expande para RGB (GRAY_ALPHA -> RGB)."""
    return np.repeat(data[..., 0:1], 3, axis=-1)


def _rgb_to_gray_alpha(data: np.ndarray) -> np.ndarray:
    """Converte RGB para GRAY_ALPHA com alfa opaco (RGB -> GRAY_ALPHA)."""
    gray = _rgb_to_gray(data)
    return _gray_to_gray_alpha(gray)


def _gray_alpha_to_rgbx(data: np.ndarray) -> np.ndarray:
    """Descarta o alfa e converte para RGBX (GRAY_ALPHA -> RGBX)."""
    rgb = _gray_alpha_to_rgb(data)
    return _rgb_to_rgbx(rgb)


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
