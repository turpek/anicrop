from __future__ import annotations
from typing import Protocol, runtime_checkable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from anicrop.frame import BaseFrame
    from anicrop.image import Image
    from anicrop.mask import Mask


@runtime_checkable
class Effect(Protocol):
    """Protocolo formal para qualquer efeito ou filtro de processamento de pixels."""

    matrix: np.ndarray

    def prepare(self, frame: BaseFrame) -> None:
        """Etapa preliminar para preparar texturas, pré-cálculos ou métricas espaciais."""
        ...

    def get_padding(self) -> tuple[int, int, int, int]:
        """Retorna a margem extra (top, right, bottom, left) necessária para efeitos de expansão."""
        ...

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        """Processa e transforma o buffer de imagem recebendo a matriz espacial ativa."""
        ...

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        """Tenta combinar este efeito com outro, retornando o efeito unificado ou None."""
        ...


class MaskedEffect(Effect):
    """Decorador que restringe a aplicação de qualquer efeito à área de uma máscara."""

    def __init__(self, effect: Effect, mask: Mask, matrix: np.ndarray | None = None):
        self.effect = effect
        self.mask = mask
        self.matrix = matrix if matrix is not None else getattr(effect, "matrix", np.identity(3, dtype=np.float32))

    def prepare(self, frame: BaseFrame) -> None:
        """Prepara o efeito interno e a máscara."""
        self.effect.prepare(frame)
        self.mask.prepare(frame)

    def get_padding(self) -> tuple[int, int, int, int]:
        """Retorna o padding do efeito interno."""
        return self.effect.get_padding()

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        """Aplica o efeito interno e interpola os pixels utilizando a máscara."""
        filtered = self.effect.apply(image, matrix)
        return self.mask.modulate_blend(image, filtered)

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        """Combina dois MaskedEffects que compartilhem da mesma máscara subjacente."""
        if isinstance(other, MaskedEffect) and self.mask == other.mask:
            merged_effect = self.effect.merge(other.effect, matrix)
            if merged_effect is not None:
                return MaskedEffect(merged_effect, self.mask, matrix=matrix)
        return None
