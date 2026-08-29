from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

import numpy as np


from anicrop.enums import BlendMode, ImageFormat
from anicrop.image import Image
from anicrop.spatial import Region

if TYPE_CHECKING:
    from anicrop.container import BaseLayer


def blend_rendered_images(
    images: Iterable[tuple[BaseLayer, Image, Region]],
    buffer: Image,
) -> Image:
    """Realiza a composição das imagens renderizadas em ordem reversa diretamente no buffer de destino."""
    for base_layer, image, region in images:
        blend = BLEND_MODE[base_layer.blend_mode]
        blend(buffer.view(region), image, base_layer.opacity)
    return buffer


try:
    from anicrop.native.blend import (  # type: ignore[import-untyped]
        blend_normal as _cy_blend_normal,
        blend_normal_linear as _cy_blend_normal_linear,
        hard_masking as _cy_hard_masking,
    )
    _HAS_CY_BLEND = True
except ImportError:
    _cy_blend_normal = None
    _cy_blend_normal_linear = None
    _cy_hard_masking = None
    _HAS_CY_BLEND = False


def _blend_normal_linear_numpy(b_view: np.ndarray, e_view: np.ndarray, opacity: float) -> None:
    b_has_alpha = b_view.shape[-1] in (2, 4)
    e_has_alpha = e_view.shape[-1] in (2, 4)

    # 1. Criar a máscara (Otimização)
    if e_has_alpha:
        mask = e_view[..., -1] > 0
    else:
        h, w = b_view.shape[:2]
        mask = np.ones((h, w), dtype=bool)

    if not np.any(mask):
        return

    b_channels = 1 if b_view.shape[-1] in (1, 2) else 3
    e_channels = 1 if e_view.shape[-1] in (1, 2) else 3

    # 2. EXTRAÇÃO E LINEARIZAÇÃO DO OVERLAY (EDIT)
    rgb_e_srgb = e_view[mask, :e_channels].astype(np.float32) / 255.0

    if b_channels == 3 and e_channels == 1:
        rgb_e_srgb = np.repeat(rgb_e_srgb, 3, axis=-1)
    elif b_channels == 1 and e_channels == 3:
        rgb_e_srgb = (0.299 * rgb_e_srgb[..., 0:1] + 0.587 * rgb_e_srgb[..., 1:2] + 0.114 * rgb_e_srgb[..., 2:3])

    # LINEARIZA: Eleva a 2.2 para remover a curva da tela
    rgb_e_lin = rgb_e_srgb ** 2.2

    if e_has_alpha:
        alpha_e = (e_view[mask, -1:].astype(np.float32) / 255.0) * opacity
    else:
        alpha_e = np.full((np.count_nonzero(mask), 1), opacity, dtype=np.float32)

    # 3. MATEMÁTICA E DESLINEARIZAÇÃO (BASE)
    rgb_b_srgb = b_view[mask, :b_channels].astype(np.float32) / 255.0
    rgb_b_lin = rgb_b_srgb ** 2.2  # Lineariza o fundo

    if b_has_alpha:
        # Fundo COM transparência
        alpha_b = b_view[mask, -1:].astype(np.float32) / 255.0

        # Calcula o Alpha resultante da mesclagem (mesma fórmula Porter-Duff)
        out_a = alpha_e + alpha_b * (1.0 - alpha_e)
        out_a_safe = np.where(out_a == 0, 1.0, out_a)

        # A Mágica: Mistura as luzes LINEARES usando os Alphas
        out_rgb_lin = (rgb_e_lin * alpha_e + rgb_b_lin * alpha_b * (1.0 - alpha_e)) / out_a_safe

        # DESLINEARIZA: Eleva a (1 / 2.2) para devolver a curva sRGB que o monitor espera ver
        out_rgb_srgb = out_rgb_lin ** (1.0 / 2.2)

        # Injeta de volta (Cor sRGB e Alpha original)
        b_view[mask, :b_channels] = np.clip(out_rgb_srgb * 255, 0, 255).astype(np.uint8)
        b_view[mask, -1:] = np.clip(out_a * 255, 0, 255).astype(np.uint8)

    else:
        # Fundo SÓLIDO (Ex: RGB puro ou Grayscale)
        out_rgb_lin = (rgb_e_lin * alpha_e) + (rgb_b_lin * (1.0 - alpha_e))

        # Deslineariza o resultado final
        out_rgb_srgb = out_rgb_lin ** (1.0 / 2.2)

        b_view[mask, :b_channels] = np.clip(out_rgb_srgb * 255, 0, 255).astype(np.uint8)


def blend_normal_linear(base: Image, edit: Image, opacity: float = 1.0) -> None:
    """Realiza o blend de forma fisicamente correta (Linear Blending).

    Converte as cores sRGB para espaço linear antes da mistura (evitando o escurecimento
    dos tons médios) e converte de volta para sRGB ao final.
    """
    if opacity <= 0.0:
        return

    base_arr = base[...]
    edit_arr = edit[...]
    h, w = min(base_arr.shape[0], edit_arr.shape[0]), min(base_arr.shape[1], edit_arr.shape[1])

    b_view = base_arr[:h, :w]
    e_view = edit_arr[:h, :w]

    if _HAS_CY_BLEND and _cy_blend_normal_linear is not None:
        _cy_blend_normal_linear(b_view, e_view, opacity)
        return

    _blend_normal_linear_numpy(b_view, e_view, opacity)


def _blend_normal_numpy(b_view: np.ndarray, e_view: np.ndarray, opacity: float) -> None:
    b_has_alpha = b_view.shape[-1] in (2, 4)
    e_has_alpha = e_view.shape[-1] in (2, 4)

    b_channels = 1 if b_view.shape[-1] in (1, 2) else 3
    e_channels = 1 if e_view.shape[-1] in (1, 2) else 3

    # Fast-Path: Cópia direta para imagens 100% sólidas com opacidade total (1.0)
    if opacity >= 1.0:
        if e_view.shape[-1] == b_view.shape[-1] and (not e_has_alpha or np.all(e_view[..., -1] == 255)):
            np.copyto(b_view, e_view)
            return
        if e_view.shape[-1] == 3 and b_view.shape[-1] == 4:
            np.copyto(b_view[..., :3], e_view)
            b_view[..., 3] = 255
            return
        if e_view.shape[-1] == 1 and b_view.shape[-1] == 4:
            np.copyto(b_view[..., :3], np.repeat(e_view, 3, axis=-1))
            b_view[..., 3] = 255
            return
        if e_view.shape[-1] == 1 and b_view.shape[-1] == 3:
            np.copyto(b_view, np.repeat(e_view, 3, axis=-1))
            return
        if e_view.shape[-1] == 1 and b_view.shape[-1] == 2:
            np.copyto(b_view[..., :1], e_view)
            b_view[..., 1] = 255
            return

    # 1. Criar a máscara (Otimização)
    if e_has_alpha:
        mask = e_view[..., -1] > 0
    else:
        h, w = b_view.shape[:2]
        mask = np.ones((h, w), dtype=bool)

    if not np.any(mask):
        return

    # 2. Extrair dados da imagem Edit (Cima)
    rgb_e = e_view[mask, :e_channels].astype(np.float32)
    if b_channels == 3 and e_channels == 1:
        rgb_e = np.repeat(rgb_e, 3, axis=-1)
    elif b_channels == 1 and e_channels == 3:
        rgb_e = (0.299 * rgb_e[..., 0:1] + 0.587 * rgb_e[..., 1:2] + 0.114 * rgb_e[..., 2:3])

    if e_has_alpha:
        alpha_e = (e_view[mask, -1:].astype(np.float32) / 255.0) * opacity
    else:
        alpha_e = np.full((np.count_nonzero(mask), 1), opacity, dtype=np.float32)

    # 3. Matemática baseada no formato do Fundo (Base)
    rgb_b = b_view[mask, :b_channels].astype(np.float32)

    if b_has_alpha:
        # Fundo COM transparência (Usa Porter-Duff Over)
        alpha_b = b_view[mask, -1:].astype(np.float32) / 255.0

        # Calcula o Alpha resultante da mesclagem das duas camadas
        out_a = alpha_e + alpha_b * (1.0 - alpha_e)
        out_a_safe = np.where(out_a == 0, 1.0, out_a)

        out_rgb = (rgb_e * alpha_e + rgb_b * alpha_b * (1.0 - alpha_e)) / out_a_safe

        # Injeta de volta (Cor e Alpha novo)
        b_view[mask, :b_channels] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        b_view[mask, -1:] = np.clip(out_a * 255, 0, 255).astype(np.uint8)

    else:
        # Fundo SÓLIDO (Ex: RGB puro ou Grayscale)
        out_rgb = (rgb_e * alpha_e) + (rgb_b * (1.0 - alpha_e))
        b_view[mask, :b_channels] = np.clip(out_rgb, 0, 255).astype(np.uint8)


def blend_normal(base: Image, edit: Image, opacity: float = 1.0) -> None:
    """Realiza o blend de forma segura usando a fórmula Porter-Duff 'Over',

    preservando as bordas suaves (anti-aliasing) em fundos transparentes.
    """
    if opacity <= 0.0:
        return

    base_arr = base[...]
    edit_arr = edit[...]

    if _HAS_CY_BLEND and _cy_blend_normal is not None:
        _cy_blend_normal(base_arr, edit_arr, opacity)
        return

    h, w = min(base_arr.shape[0], edit_arr.shape[0]), min(base_arr.shape[1], edit_arr.shape[1])
    b_view = base_arr[:h, :w]
    e_view = edit_arr[:h, :w]
    e_has_alpha = e_view.shape[-1] in (2, 4)

    # Fast-Path: Cópia direta para imagens 100% sólidas com opacidade total (1.0)
    if opacity >= 1.0:
        if e_view.shape[-1] == b_view.shape[-1] and (not e_has_alpha or np.all(e_view[..., -1] == 255)):
            np.copyto(b_view, e_view)
            return
        if e_view.shape[-1] == 3 and b_view.shape[-1] == 4:
            np.copyto(b_view[..., :3], e_view)
            b_view[..., 3] = 255
            return
        if e_view.shape[-1] == 1 and b_view.shape[-1] == 4:
            np.copyto(b_view[..., :3], np.repeat(e_view, 3, axis=-1))
            b_view[..., 3] = 255
            return
        if e_view.shape[-1] == 1 and b_view.shape[-1] == 3:
            np.copyto(b_view, np.repeat(e_view, 3, axis=-1))
            return
        if e_view.shape[-1] == 1 and b_view.shape[-1] == 2:
            np.copyto(b_view[..., :1], e_view)
            b_view[..., 1] = 255
            return

    _blend_normal_numpy(b_view, e_view, opacity)


def hard_masking_overlay_with_alpha(
    base: Image,
    overlay: Image,
    color_channels: int,
    opacity: float,
) -> None:

    if opacity == 0:
        return

    mask = overlay[..., -1:] > 0
    np.copyto(base[..., :color_channels], overlay[..., :color_channels], where=mask)

    if base.has_alpha:
        if opacity < 1.0:
            alpha_modificado = (overlay[..., -1:] * opacity).astype(np.uint8)
            np.copyto(base[..., -1:], alpha_modificado, where=mask)
        else:
            np.copyto(base[..., -1:], overlay[..., -1:], where=mask)


def hard_masking_overlay_without_alpha(
    base: Image,
    overlay: Image,
    color_channels: int,
    opacity: float,
) -> None:

    if opacity == 0:
        return

    base[..., :color_channels] = overlay[..., :color_channels]
    if base.has_alpha:
        alpha_value = int(255 * opacity) if opacity < 1 else 255
        base[..., -1] = alpha_value


def _hard_masking_numpy(base: Image, overlay: Image, opacity: float = 1.0) -> Image:
    color_channels = 1 if overlay.format in (ImageFormat.GRAY, ImageFormat.GRAY_ALPHA) else 3
    if overlay.has_alpha:
        hard_masking_overlay_with_alpha(base, overlay, color_channels, opacity)
    else:
        hard_masking_overlay_without_alpha(base, overlay, color_channels, opacity)
    return base


def hard_masking(base: Image, overlay: Image, opacity: float = 1.0) -> Image:

    if base.size != overlay.size:
        raise ValueError(f"Size mismatch: base {base.size} != overlay {overlay.size}.")

    elif not overlay.format.same_spaces(base.format):
        raise NotImplementedError(
            f"Format mismatch: cannot blend '{overlay.format}' into '{base.format}'."
        )

    if _HAS_CY_BLEND and _cy_hard_masking is not None:
        _cy_hard_masking(base[...], overlay[...], opacity)
        return base

    return _hard_masking_numpy(base, overlay, opacity)


def _blend_clip_numpy(base: Image, overlay: Image, opacity: float = 1.0) -> Image:
    """
    Aplica o recorte de pixels (clip): modula o canal alpha da base onde houver transparência
    e preenche com fundo branco (255) as áreas cortadas de camadas sem canal alpha.
    """
    b_arr = base[...]
    o_arr = overlay[...]

    if overlay.has_alpha:
        factor = (o_arr[..., -1:].astype(np.float32) / 255.0) * opacity
    else:
        if overlay.format in (ImageFormat.GRAY, ImageFormat.GRAY_ALPHA):
            luma = o_arr[..., 0:1].astype(np.float32) / 255.0
        else:
            luma = (0.299 * o_arr[..., 0:1] + 0.587 * o_arr[..., 1:2] + 0.114 * o_arr[..., 2:3]).astype(np.float32) / 255.0
        factor = luma * opacity

    if base.has_alpha:
        b_arr[..., -1:] = np.clip(b_arr[..., -1:] * factor, 0, 255).astype(np.uint8)
        mask_zero = b_arr[..., -1] == 0
        b_arr[mask_zero, :-1] = 255
    else:
        color_channels = 1 if base.format == ImageFormat.GRAY else 3
        b_arr[..., :color_channels] = np.clip(
            b_arr[..., :color_channels] * factor + 255.0 * (1.0 - factor),
            0,
            255,
        ).astype(np.uint8)

    return base


def blend_clip(base: Image, overlay: Image, opacity: float = 1.0) -> Image:
    if base.size != overlay.size:
        raise ValueError(f"Size mismatch: base {base.size} != overlay {overlay.size}.")

    return _blend_clip_numpy(base, overlay, opacity)


BLEND_MODE = {
    BlendMode.NORMAL: blend_normal,
    BlendMode.NORMAL_LINEAR: blend_normal_linear,
    BlendMode.HARD_MASKING: hard_masking,
    BlendMode.CLIP: blend_clip,
}
