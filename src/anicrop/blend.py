from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

import numpy as np


from anicrop.enums import BlendMode
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


def blend_normal_linear(base: Image, edit: Image, opacity: float = 1.0) -> None:
    """
    Realiza o blend de forma fisicamente correta (Linear Blending).
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

    # 1. Criar a máscara (Otimização)
    if e_view.shape[-1] == 4:
        mask = e_view[..., 3] > 0
    else:
        mask = np.ones((h, w), dtype=bool)

    if not np.any(mask):
        return

    b_channels = 1 if b_view.shape[-1] == 1 else 3
    e_channels = 1 if e_view.shape[-1] == 1 else 3

    # =====================================================================
    # 2. EXTRAÇÃO E LINEARIZAÇÃO DO OVERLAY (EDIT)
    # =====================================================================
    rgb_e_srgb = e_view[mask, :e_channels].astype(np.float32) / 255.0

    if b_channels == 3 and e_channels == 1:
        rgb_e_srgb = np.repeat(rgb_e_srgb, 3, axis=-1)
    elif b_channels == 1 and e_channels == 3:
        rgb_e_srgb = (0.299 * rgb_e_srgb[..., 0:1] + 0.587 * rgb_e_srgb[..., 1:2] + 0.114 * rgb_e_srgb[..., 2:3])

    # LINEARIZA: Eleva a 2.2 para remover a curva da tela
    rgb_e_lin = rgb_e_srgb ** 2.2

    if e_view.shape[-1] == 4:
        alpha_e = (e_view[mask, 3:4].astype(np.float32) / 255.0) * opacity
    else:
        alpha_e = np.full((np.count_nonzero(mask), 1), opacity, dtype=np.float32)

    # =====================================================================
    # 3. MATEMÁTICA E DESLINEARIZAÇÃO (BASE)
    # =====================================================================
    rgb_b_srgb = b_view[mask, :b_channels].astype(np.float32) / 255.0
    rgb_b_lin = rgb_b_srgb ** 2.2  # Lineariza o fundo

    if b_view.shape[-1] == 4:
        # Fundo COM transparência
        alpha_b = b_view[mask, 3:4].astype(np.float32) / 255.0

        # Calcula o Alpha resultante da mesclagem (mesma fórmula Porter-Duff)
        out_a = alpha_e + alpha_b * (1.0 - alpha_e)
        out_a_safe = np.where(out_a == 0, 1.0, out_a)

        # A Mágica: Mistura as luzes LINEARES usando os Alphas
        out_rgb_lin = (rgb_e_lin * alpha_e + rgb_b_lin * alpha_b * (1.0 - alpha_e)) / out_a_safe

        # DESLINEARIZA: Eleva a (1 / 2.2) para devolver a curva sRGB que o monitor espera ver
        out_rgb_srgb = out_rgb_lin ** (1.0 / 2.2)

        # Injeta de volta (Cor sRGB e Alpha original)
        b_view[mask, :b_channels] = np.clip(out_rgb_srgb * 255, 0, 255).astype(np.uint8)
        b_view[mask, 3:4] = np.clip(out_a * 255, 0, 255).astype(np.uint8)

    else:
        # Fundo SÓLIDO (Ex: RGB puro ou Grayscale)
        out_rgb_lin = (rgb_e_lin * alpha_e) + (rgb_b_lin * (1.0 - alpha_e))

        # Deslineariza o resultado final
        out_rgb_srgb = out_rgb_lin ** (1.0 / 2.2)

        b_view[mask, :b_channels] = np.clip(out_rgb_srgb * 255, 0, 255).astype(np.uint8)


def blend_normal(base: Image, edit: Image, opacity: float = 1.0) -> None:
    """
    Realiza o blend de forma segura usando a fórmula Porter-Duff 'Over',
    preservando as bordas suaves (anti-aliasing) em fundos transparentes.
    """
    if opacity <= 0.0:
        return

    base_arr = base[...]
    edit_arr = edit[...]
    h, w = min(base_arr.shape[0], edit_arr.shape[0]), min(base_arr.shape[1], edit_arr.shape[1])

    b_view = base_arr[:h, :w]
    e_view = edit_arr[:h, :w]

    b_channels = 1 if b_view.shape[-1] == 1 else 3
    e_channels = 1 if e_view.shape[-1] == 1 else 3

    # Fast-Path: Cópia direta para imagens 100% sólidas com opacidade total (1.0)
    if opacity >= 1.0:
        if e_view.shape[-1] == b_view.shape[-1] and (e_view.shape[-1] in (1, 3) or (e_view.shape[-1] == 4 and np.all(e_view[..., 3] == 255))):
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

    # 1. Criar a máscara (Otimização)
    if e_view.shape[-1] == 4:
        mask = e_view[..., 3] > 0
    else:
        mask = np.ones((h, w), dtype=bool)

    if not np.any(mask):
        return

    # 2. Extrair dados da imagem Edit (Cima)
    rgb_e = e_view[mask, :e_channels].astype(np.float32)
    if b_channels == 3 and e_channels == 1:
        rgb_e = np.repeat(rgb_e, 3, axis=-1)
    elif b_channels == 1 and e_channels == 3:
        rgb_e = (0.299 * rgb_e[..., 0:1] + 0.587 * rgb_e[..., 1:2] + 0.114 * rgb_e[..., 2:3])

    if e_view.shape[-1] == 4:
        alpha_e = (e_view[mask, 3:4].astype(np.float32) / 255.0) * opacity
    else:
        alpha_e = np.full((np.count_nonzero(mask), 1), opacity, dtype=np.float32)

    # 3. Matemática baseada no formato do Fundo (Base)
    rgb_b = b_view[mask, :b_channels].astype(np.float32)

    if b_view.shape[-1] == 4:
        # Fundo COM transparência (Usa Porter-Duff Over)
        alpha_b = b_view[mask, 3:4].astype(np.float32) / 255.0

        # Calcula o Alpha resultante da mesclagem das duas camadas
        out_a = alpha_e + alpha_b * (1.0 - alpha_e)
        out_a_safe = np.where(out_a == 0, 1.0, out_a)

        out_rgb = (rgb_e * alpha_e + rgb_b * alpha_b * (1.0 - alpha_e)) / out_a_safe

        # Injeta de volta (Cor e Alpha novo)
        b_view[mask, :b_channels] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        b_view[mask, 3:4] = np.clip(out_a * 255, 0, 255).astype(np.uint8)

    else:
        # Fundo SÓLIDO (Ex: RGB puro ou Grayscale)
        out_rgb = (rgb_e * alpha_e) + (rgb_b * (1.0 - alpha_e))
        b_view[mask, :b_channels] = np.clip(out_rgb, 0, 255).astype(np.uint8)


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


def hard_masking(base: Image, overlay: Image, opacity: float = 1.0) -> Image:

    if base.size != overlay.size:
        raise ValueError(f"Size mismatch: base {base.size} != overlay {overlay.size}.")

    elif not overlay.format.same_spaces(base.format):
        raise NotImplementedError(
            f"Format mismatch: cannot blend '{overlay.format}' into '{base.format}'."
        )

    color_channels = overlay.channels

    if overlay.has_alpha:
        hard_masking_overlay_with_alpha(base, overlay, color_channels - 1, opacity)
    else:
        hard_masking_overlay_without_alpha(base, overlay, color_channels, opacity)

    return base


BLEND_MODE = {
    BlendMode.NORMAL: blend_normal,
    BlendMode.NORMAL_LINEAR: blend_normal_linear,
    BlendMode.HARD_MASKING: hard_masking,
}
