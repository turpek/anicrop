from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from anicrop.stack import NamedStack
from anicrop.transform import mat_global, mat_inverse

if TYPE_CHECKING:
    from anicrop.container import BaseLayer
    from anicrop.image import Image
    from anicrop.mask import Mask


class Effect(ABC):
    """Classe base abstrata para qualquer efeito ou filtro puro de processamento de pixels."""

    def __init__(self, visible: bool = True, name: str = "Effect"):
        self.visible = visible
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, visible={self.visible})"

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


class DynamicEffect(Effect):
    """Classe base abstrata para efeitos dinâmicos/relacionais cujo resultado não deve ser cacheado."""

    def __init__(self, visible: bool = True, name: str = "DynamicEffect"):
        super().__init__(visible=visible, name=name)

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        """Efeitos dinâmicos por padrão não são combinados estaticamente."""
        return None


class BoundEffect(Effect):
    """Envelope explícito que ancora um Effect à geometria da camada e opcionalmente modula por máscara."""

    def __init__(
        self,
        effect: Effect,
        matrix: np.ndarray,
        mask: Mask | None = None,
        visible: bool = True,
        name: str | None = None,
    ):
        super().__init__(visible=visible, name=name or effect.name)
        self.effect = effect
        self.matrix = matrix
        self.mask = mask

    def __repr__(self) -> str:
        return f"BoundEffect({self.effect!r}, visible={self.visible})"

    @classmethod
    def from_layer(
        cls,
        layer: BaseLayer,
        effect: Effect,
        mask: Mask | None = None,
        visible: bool = True,
        name: str | None = None,
    ) -> BoundEffect:
        """Cria e ancora um BoundEffect à matriz inversa global da camada."""
        inv_matrix = mat_inverse(mat_global(layer))
        return cls(effect, matrix=inv_matrix, mask=mask, visible=visible, name=name)

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


class EffectStack(NamedStack[Effect]):
    """Contêiner especializado para o pipeline sequencial de efeitos de pós-processamento."""

    _item_type_name: str = "Effect"

    def _validate_item(self, item: Effect) -> None:
        if not isinstance(item, Effect):
            raise TypeError(f"Expected Effect, got {type(item).__name__}")

    def _contains_item(self, item: Any) -> bool:
        return item in self._items or any(
            isinstance(e, BoundEffect) and e.effect is item for e in self._items
        )

    def _matches_item(self, candidate: Effect, target: Any) -> bool:
        return candidate is target or (
            isinstance(candidate, BoundEffect) and candidate.effect is target
        )

    def get_padding(self) -> tuple[int, int, int, int]:
        """Calcula a margem agregada máxima (top, right, bottom, left) dos efeitos visíveis."""
        top, right, bottom, left = 0, 0, 0, 0
        for effect in self._items:
            pt, pr, pb, pl = effect.get_padding()
            top = max(top, pt)
            right = max(right, pr)
            bottom = max(bottom, pb)
            left = max(left, pl)
        return top, right, bottom, left
