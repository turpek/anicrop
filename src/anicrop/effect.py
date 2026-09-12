from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anicrop.image import Image
    from anicrop.mask import Mask


class Effect(ABC):
    """Classe base abstrata para qualquer efeito ou filtro puro de processamento de pixels."""

    def __init__(self, visible: bool = True, name: str = "Effect"):
        self.visible = visible
        self.name = name

    @abstractmethod
    def get_padding(self) -> tuple[int, int, int, int]:
        """Retorna a margem extra (top, right, bottom, left) necessária para efeitos de expansão."""
        pass

    @abstractmethod
    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        """Processa e transforma o buffer de imagem recebendo a matriz espacial ativa."""
        pass

    @abstractmethod
    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        """Tenta combinar este efeito com outro, retornando o efeito unificado ou None."""
        pass


class BoundEffect(Effect):
    """Envelope explícito que ancora um Effect à geometria da camada e opcionalmente modula por máscara."""

    def __init__(
        self,
        effect: Effect,
        matrix: np.ndarray,
        mask: Mask | None = None,
        visible: bool = True,
        name: str = "BoundEffect",
    ):
        super().__init__(visible=visible, name=name)
        self.effect = effect
        self.matrix = matrix
        self.mask = mask

    def get_padding(self) -> tuple[int, int, int, int]:
        """Retorna o padding do efeito interno se visível."""
        if not self.visible:
            return (0, 0, 0, 0)
        return self.effect.get_padding()

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        """Aplica o efeito calculando a matriz delta combinada e aplicando modulação por máscara."""
        if not self.visible:
            return image

        delta_matrix = matrix @ self.matrix
        filtered = self.effect.apply(image, delta_matrix)

        if self.mask is not None and self.mask.visible:
            return self.mask.modulate_blend(image, filtered)

        return filtered

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        """Combina dois BoundEffects compativeis com a mesma máscara e visibilidade."""
        if not isinstance(other, BoundEffect):
            return None
        if self.visible != other.visible or self.mask != other.mask:
            return None

        merged_inner = self.effect.merge(other.effect, matrix)
        if merged_inner is not None:
            return BoundEffect(
                merged_inner, self.matrix, mask=self.mask, visible=self.visible
            )
        return None


# Alias para retrocompatibilidade
MaskedEffect = BoundEffect
