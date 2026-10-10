from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import numpy as np

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


class EffectStack:
    """Contêiner especializado para o pipeline sequencial de efeitos de pós-processamento."""

    def __init__(self, effects: Sequence[Effect] | None = None) -> None:
        self._effects: list[Effect] = []
        if effects is not None:
            self.extend(effects)

    def __len__(self) -> int:
        return len(self._effects)

    def __iter__(self) -> Iterator[Effect]:
        return iter(self._effects)

    def __reversed__(self) -> Iterator[Effect]:
        return reversed(self._effects)

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return any(e.name == item for e in self._effects)
        return item in self._effects or any(
            isinstance(e, BoundEffect) and e.effect is item for e in self._effects
        )

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, (int, slice)):
            return self._effects[key]
        if isinstance(key, str):
            for e in self._effects:
                if e.name == key:
                    return e
            raise KeyError(f"Effect '{key}' not found in EffectStack")
        raise TypeError(
            f"EffectStack indices must be integers, slices or strings, not {type(key).__name__}"
        )

    def __delitem__(self, key: int | str) -> None:
        if isinstance(key, int):
            del self._effects[key]
        elif isinstance(key, str):
            target = self[key]
            self._effects.remove(target)
        else:
            raise TypeError(
                f"EffectStack indices must be integers or strings, not {type(key).__name__}"
            )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, EffectStack):
            return self._effects == other._effects
        if isinstance(other, (list, tuple)):
            return self._effects == list(other)
        return False

    def __repr__(self) -> str:
        return f"EffectStack({self._effects!r})"

    def add(self, effect: Effect) -> Effect:
        """Adiciona um efeito ao topo do pipeline de pós-processamento."""
        if not isinstance(effect, Effect):
            raise TypeError(f"Expected Effect, got {type(effect).__name__}")
        self._effects.append(effect)
        return effect

    def extend(self, effects: Sequence[Effect] | EffectStack) -> None:
        """Adiciona múltiplos efeitos à pilha."""
        for e in effects:
            self.add(e)

    def remove(self, effect: Effect | str) -> None:
        """Remove um efeito da pilha por instância ou por nome."""
        if isinstance(effect, str):
            try:
                target = self[effect]
            except KeyError as err:
                raise ValueError(f"Effect '{effect}' not found in EffectStack") from err
            self._effects.remove(target)
            return

        for e in self._effects:
            if e is effect or (isinstance(e, BoundEffect) and e.effect is effect):
                self._effects.remove(e)
                return
        raise ValueError(f"Effect {effect} not found in EffectStack")

    def pop(self, index: int = -1) -> Effect:
        """Remove e retorna o efeito no índice especificado (padrão: topo)."""
        return self._effects.pop(index)

    def clear(self) -> None:
        """Remove todos os efeitos da pilha."""
        self._effects.clear()

    def insert(self, index: int, effect: Effect) -> Effect:
        """Insere um efeito em uma posição específica do pipeline."""
        if not isinstance(effect, Effect):
            raise TypeError(f"Expected Effect, got {type(effect).__name__}")
        self._effects.insert(index, effect)
        return effect

    def move(self, effect: Effect | str, new_index: int) -> None:
        """Move um efeito para uma nova posição na ordem de execução."""
        idx = self.index(effect)
        item = self._effects.pop(idx)
        self._effects.insert(new_index, item)

    def swap(self, a: Effect | int, b: Effect | int) -> None:
        """Troca a posição de dois efeitos no pipeline."""
        idx_a = a if isinstance(a, int) else self.index(a)
        idx_b = b if isinstance(b, int) else self.index(b)
        self._effects[idx_a], self._effects[idx_b] = (
            self._effects[idx_b],
            self._effects[idx_a],
        )

    def index(self, effect: Effect | str) -> int:
        """Retorna o índice de um efeito por instância ou nome."""
        if isinstance(effect, str):
            for i, e in enumerate(self._effects):
                if e.name == effect:
                    return i
            raise ValueError(f"Effect '{effect}' not found in EffectStack")

        for i, e in enumerate(self._effects):
            if e is effect or (isinstance(e, BoundEffect) and e.effect is effect):
                return i
        raise ValueError(f"Effect {effect} not found in EffectStack")

    def get_padding(self) -> tuple[int, int, int, int]:
        """Calcula a margem agregada máxima (top, right, bottom, left) dos efeitos visíveis."""
        top, right, bottom, left = 0, 0, 0, 0
        for effect in self._effects:
            pt, pr, pb, pl = effect.get_padding()
            top = max(top, pt)
            right = max(right, pr)
            bottom = max(bottom, pb)
            left = max(left, pl)
        return top, right, bottom, left
