from anicrop.image import Image
from enum import Enum
import numpy as np


def blend_normal_linear(base: Image, edit: Image, opacity: float = 1.0) -> None:
    """
    Realiza o blend de forma fisicamente correta (Linear Blending).
    Converte as cores sRGB para espaço linear antes da mistura (evitando o escurecimento
    dos tons médios) e converte de volta para sRGB ao final.
    """
    if opacity <= 0.0:
        return

    base = base[...]
    edit = edit[...]
    h, w = min(base.shape[0], edit.shape[0]), min(base.shape[1], edit.shape[1])

    b_view = base[:h, :w]
    e_view = edit[:h, :w]

    # 1. Criar a máscara (Otimização)
    if e_view.shape[-1] == 4:
        mask = e_view[..., 3] > 0
    else:
        mask = np.ones((h, w), dtype=bool)

    if not np.any(mask):
        return

    # =====================================================================
    # 2. EXTRAÇÃO E LINEARIZAÇÃO DO OVERLAY (EDIT)
    # =====================================================================
    # Para a curva Gamma funcionar, precisamos converter a cor para a escala 0.0 a 1.0 primeiro
    rgb_e_srgb = e_view[mask, :3].astype(np.float32) / 255.0

    # LINEARIZA: Eleva a 2.2 para remover a curva da tela
    rgb_e_lin = rgb_e_srgb ** 2.2

    if e_view.shape[-1] == 4:
        # ATENÇÃO: O canal Alpha NUNCA sofre correção de Gamma, ele é sempre linear!
        alpha_e = (e_view[mask, 3:4].astype(np.float32) / 255.0) * opacity
    else:
        alpha_e = np.full((np.count_nonzero(mask), 1), opacity, dtype=np.float32)

    # =====================================================================
    # 3. MATEMÁTICA E DESLINEARIZAÇÃO (BASE)
    # =====================================================================
    if b_view.shape[-1] == 4:
        # Fundo COM transparência
        rgb_b_srgb = b_view[mask, :3].astype(np.float32) / 255.0
        rgb_b_lin = rgb_b_srgb ** 2.2  # Lineariza o fundo
        alpha_b = b_view[mask, 3:4].astype(np.float32) / 255.0

        # Calcula o Alpha resultante da mesclagem (mesma fórmula Porter-Duff)
        out_a = alpha_e + alpha_b * (1.0 - alpha_e)
        out_a_safe = np.where(out_a == 0, 1.0, out_a)

        # A Mágica: Mistura as luzes LINEARES usando os Alphas
        out_rgb_lin = (rgb_e_lin * alpha_e + rgb_b_lin * alpha_b * (1.0 - alpha_e)) / out_a_safe

        # DESLINEARIZA: Eleva a (1 / 2.2) para devolver a curva sRGB que o monitor espera ver
        out_rgb_srgb = out_rgb_lin ** (1.0 / 2.2)

        # Injeta de volta (Cor sRGB e Alpha original)
        b_view[mask, :3] = np.clip(out_rgb_srgb * 255, 0, 255).astype(np.uint8)
        b_view[mask, 3:4] = np.clip(out_a * 255, 0, 255).astype(np.uint8)

    else:
        # Fundo SÓLIDO (Ex: RGB puro)
        rgb_b_srgb = b_view[mask, :3].astype(np.float32) / 255.0
        rgb_b_lin = rgb_b_srgb ** 2.2  # Lineariza o fundo

        # Alpha Blending comum com as cores lineares
        out_rgb_lin = (rgb_e_lin * alpha_e) + (rgb_b_lin * (1.0 - alpha_e))

        # Deslineariza o resultado final
        out_rgb_srgb = out_rgb_lin ** (1.0 / 2.2)

        b_view[mask, :3] = np.clip(out_rgb_srgb * 255, 0, 255).astype(np.uint8)


def blend_normal(base: Image, edit: Image, opacity: float = 1.0) -> None:
    """
    Realiza o blend de forma segura usando a fórmula Porter-Duff 'Over',
    preservando as bordas suaves (anti-aliasing) em fundos transparentes.
    """
    # Otimização suprema: se a camada for invisível, não fazemos nada!
    if opacity <= 0.0:
        return

    base = base[...]
    edit = edit[...]
    h, w = min(base.shape[0], edit.shape[0]), min(base.shape[1], edit.shape[1])

    b_view = base[:h, :w]
    e_view = edit[:h, :w]

    # 1. Criar a máscara (Otimização)
    if e_view.shape[-1] == 4:
        mask = e_view[..., 3] > 0
    else:
        mask = np.ones((h, w), dtype=bool)

    if not np.any(mask):
        return

    # 2. Extrair dados da imagem Edit (Cima)
    rgb_e = e_view[mask, :3].astype(np.float32)
    if e_view.shape[-1] == 4:
        # AQUI ENTRA O OPACITY: Multiplicamos o alfa extraído pela opacidade da camada
        alpha_e = (e_view[mask, 3:4].astype(np.float32) / 255.0) * opacity
    else:
        # Se a imagem não tiver alfa, a opacidade vira o próprio alfa!
        alpha_e = np.full((np.count_nonzero(mask), 1), opacity, dtype=np.float32)

    # 3. Matemática baseada no formato do Fundo (Base)
    if b_view.shape[-1] == 4:
        # Fundo COM transparência (Usa Porter-Duff Over)
        rgb_b = b_view[mask, :3].astype(np.float32)
        alpha_b = b_view[mask, 3:4].astype(np.float32) / 255.0

        # Calcula o Alpha resultante da mesclagem das duas camadas
        out_a = alpha_e + alpha_b * (1.0 - alpha_e)

        # Evita divisão por zero onde o pixel final for 100% transparente
        out_a_safe = np.where(out_a == 0, 1.0, out_a)

        # A Mágica: Multiplica as cores pelos seus respectivos Alphas
        out_rgb = (rgb_e * alpha_e + rgb_b * alpha_b * (1.0 - alpha_e)) / out_a_safe

        # Injeta de volta (Cor e Alpha novo)
        b_view[mask, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        b_view[mask, 3:4] = np.clip(out_a * 255, 0, 255).astype(np.uint8)

    else:
        # Fundo SÓLIDO (Ex: RGB puro)
        # Aqui o fundo é 100% opaco, então o Alpha Blending simples funciona perfeitamente
        out_rgb = (rgb_e * alpha_e) + (b_view[mask, :3] * (1.0 - alpha_e))
        b_view[mask, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)


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


class BlendMode(Enum):
    """Defines how an edit layer blends with the underlying content."""
    NORMAL = 'normal'
    NORMAL_LINEAR = 'normal_linear'
    MULTIPLY = 'multiply'
    HARD_MASKING = 'hard_masking'


BLEND_MODE = {
    BlendMode.NORMAL: blend_normal,
    BlendMode.NORMAL_LINEAR: blend_normal_linear,
    BlendMode.HARD_MASKING: hard_masking,
}
