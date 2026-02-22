from anicrop.image import Image
from anicrop.spatial import Region
from enum import Enum
import numpy as np


def blend_normal(base: Image, edit: Image) -> None:
    """
    Realiza o blend de forma segura, validando a existência do canal Alfa.
    """
    base = base[...]
    edit = edit[...]
    h, w = min(base.shape[0], edit.shape[0]), min(base.shape[1], edit.shape[1])

    b_view = base[:h, :w]
    e_view = edit[:h, :w]

    # 1. Criar a máscara
    if e_view.shape[-1] == 4:
        mask = e_view[..., 3] > 0
    else:
        mask = np.ones((h, w), dtype=bool)

    # Otimização: Se a máscara for toda falsa (Edit invisível), encerra aqui
    if not np.any(mask):
        return

    # 2. Extrair o Alpha APENAS se o canal existir
    if e_view.shape[-1] == 4:
        # e_view[mask] retorna (N, 4). O slice [:, 3:4] deixa no formato (N, 1) para o broadcast
        alpha = (e_view[mask, 3:4].astype(np.float32) / 255.0)
    else:
        # Se for RGB, é 100% sólido. O float 1.0 funciona perfeitamente no broadcast.
        alpha = 1.0

    # 3. Matemática do Blend restrita aos pixels da máscara
    pixel_blend = (e_view[mask, :3] * alpha) + (b_view[mask, :3] * (1.0 - alpha))

    # 4. Injeta de volta
    b_view[mask, :3] = pixel_blend.astype(np.uint8)

    # 5. Tratamento de transparência da base
    if b_view.shape[-1] == 4:
        b_view[mask, 3] = 255


def hard_masking(base: Image, overlay: Image) -> Image:
    if base.size != overlay.size:
        raise ValueError(f"Size mismatch: base {base.size} != overlay {overlay.size}.")

    elif not overlay.format.same_spaces(base.format):
        raise NotImplementedError(f"Format mismatch: cannot blend '{overlay.format}' into '{base.format}'.")

    elif base.shape == overlay.shape:
        if overlay.has_alpha:
            mask = overlay[..., -1:] > 0
            np.copyto(base[...], overlay[...], where=mask)

        else:
            base[...] = overlay[...]

    elif overlay.has_alpha:
        ch = overlay.channels - 1
        mask = overlay[..., -1:] > 0
        np.copyto(base[...], overlay[..., :ch], where=mask)

    else:
        ch = base.channels - 1
        base[..., :ch] = overlay[...]
        base[..., ch] = 255

    return base


class BlendMode(Enum):
    """Defines how an edit layer blends with the underlying content."""
    NORMAL = 'normal'
    MULTIPLY = 'multiply'
    HARD_MASKING = 'hard_masking'


BLEND_MODE = {
    BlendMode.NORMAL: blend_normal,
    BlendMode.HARD_MASKING: hard_masking,
}
