from __future__ import annotations
from abc import ABC, abstractmethod
from anicrop.layer import Layer
from typing import Any
import numpy as np
from collections import deque


class Command(ABC):

    def __init__(self, name: str, layer: Layer):
        self._sealed = False
        self._layer = layer
        self._name = name

    @abstractmethod
    def execute(self) -> None:
        ...

    @abstractmethod
    def undo(self) -> None:
        ...

    @abstractmethod
    def has_changes(self) -> bool:
        """Determina se o comando gerou alguma mutação de estado."""
        ...

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    def seal(self) -> None:
        self._sealed = True

    def can_merge(self, name: str, layer: object) -> bool:
        if self.is_sealed:
            return False
        return self._name == name and self._layer == layer


class SnapshotLayerCommand(Command):

    def __init__(self, name: str, layer: Layer):
        super().__init__(name, layer)
        self._snapshot_before = self.capture_state(layer)
        self._new_state = None

    def execute(self) -> None:
        if self._new_state is not None:
            self.restore_state(self._layer, self._new_state)

    def undo(self) -> None:
        if not self._sealed:
            self.seal()
        self.restore_state(self._layer, self._snapshot_before)

    def seal(self) -> None:
        if not self._sealed:
            self._new_state = self.capture_state(self._layer)
            self._sealed = True

    def has_changes(self) -> bool:
        if not self._sealed:
            return True
        return self._snapshot_before != self._new_state

    @staticmethod
    def capture_state(layer: Layer) -> dict[str, Any]:
        """Extrai o estado mutável do Layer sem expor métodos de infraestrutura nele."""
        # 1. Copia o estado do Composer de transformações se existir
        transform_state = layer._transform.copy() if layer._transform is not None else None

        # 2. Copia as edições destrutivas da lista
        edits_state = []
        for edit in layer._edits:
            edits_state.append({
                "image": edit.image,
                "region": edit.region,
                "matrix": np.copy(edit.matrix),
                "blend_mode": edit.blend_mode,
                "name": edit.name
            })

        return {
            "name": layer.name,
            "opacity": layer.opacity,
            "blend_mode": layer.blend_mode,
            "region": layer._region,
            "transform": transform_state,
            "edits": edits_state,
            "visible": layer.visible,
            "opacity_mask": np.copy(layer._opacity_mask) if layer._opacity_mask is not None else None
        }

    @staticmethod
    def restore_state(layer: Layer, state: dict[str, Any]) -> None:
        """Aplica o estado diretamente nos atributos do Layer."""
        layer.name = state["name"]
        layer.opacity = state["opacity"]
        layer.blend_mode = state["blend_mode"]
        layer._region = state["region"]
        layer.visible = state["visible"]
        layer._opacity_mask = state["opacity_mask"]

        # Restaura o Composer transform clonando o estado salvo
        t_state = state["transform"]
        layer._transform = t_state.copy() if t_state is not None else None

        # Restaura a fila de edições (EditLayer)
        from anicrop.layer import EditLayer
        restored_edits = deque()
        for e_state in state["edits"]:
            edit = EditLayer(
                image=e_state["image"],
                region=e_state["region"],
                matrix=e_state["matrix"],
                blend_mode=e_state["blend_mode"],
                name=e_state["name"]
            )
            restored_edits.append(edit)
        layer._edits = restored_edits

    def __repr__(self):
        return f'{type(self).__name__}(name="{self._name}")'
