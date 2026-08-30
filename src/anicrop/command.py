from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Iterable

import numpy as np

from anicrop.container import (
    BaseLayer,
    Container,
    LayerStack,
    NodeContainerProtocol,
    NullContainer,
)
from anicrop.geometry import GeometryController
from anicrop.layer import Layer
from anicrop.mask import Mask


def _create_snapshot(
    obj: Any, registry: Iterable[tuple[type, type[StateSnapshot]]]
) -> StateSnapshot:
    if hasattr(obj, "_target"):
        obj = getattr(obj, "_target")
    elif not isinstance(obj, NullContainer):
        raise TypeError("Expected a Proxy or neutral object (NullContainer).")

    for target_type, snapshot_cls in registry:
        if isinstance(obj, target_type):
            return snapshot_cls(obj)

    raise TypeError(f"No snapshot registered for type {type(obj).__name__}")


class StateSnapshot(ABC):
    @abstractmethod
    def __init__(self, item: Any):
        pass

    @abstractmethod
    def restore(self) -> None:
        pass

    @abstractmethod
    def has_change(self, other: Any) -> bool:
        pass


class ContainerSnapshot(StateSnapshot):
    def __init__(self, item: Container):

        self._parent_inverse = np.copy(item._parent_inverse)
        self._children = list(item._children)
        self._parent_children = list(item.parent._children)
        self._parent = item.parent
        self._item = item

    def restore(self) -> None:
        self._item._parent_inverse = np.copy(self._parent_inverse)
        self._item._children = list(self._children)

        self._item.parent = self._parent
        self._parent._children = list(self._parent_children)

    def has_change(self, other: ContainerSnapshot) -> bool:
        return self._children != other._children


class NodeContainerSnapshot(StateSnapshot):
    def __init__(self, item: NodeContainerProtocol):
        self._parent_inverse = np.copy(item._parent_inverse)
        self._parent = item.parent
        self._item = item

    def restore(self) -> None:
        self._item._parent_inverse = np.copy(self._parent_inverse)
        self._item.parent = self._parent

    def has_change(self, other: NodeContainerSnapshot) -> bool:
        return self._parent is not other._parent


class NullContainerSnapshot(StateSnapshot):
    def __init__(self, item: Any):
        self._item = item

    def restore(self) -> None:
        pass

    def has_change(self, other: StateSnapshot) -> bool:
        return False


class GeometryControllerSnapshot(StateSnapshot):
    def __init__(self, controller: GeometryController):
        self._frame_region = controller.frame.region
        self._frame_strategy = controller.frame
        self._offset = controller._offset
        self._controller = controller

    def restore(self) -> None:
        self._controller._frame = self._frame_strategy
        self._controller._offset = self._offset
        self._controller.sync(self._frame_region)

    def _region_changed(self, other: GeometryControllerSnapshot) -> bool:
        return self._frame_region != other._frame_region

    def _instance_changed(self, other: GeometryControllerSnapshot) -> bool:
        return self._frame_strategy is not other._frame_strategy

    def _type_changed(self, other: GeometryControllerSnapshot) -> bool:
        return type(self._frame_strategy) is not type(other._frame_strategy)

    def has_change(self, other: GeometryControllerSnapshot) -> bool:
        return (
            self._region_changed(other)
            or self._instance_changed(other)
            or self._type_changed(other)
        )


class BaseLayerSnapshot(StateSnapshot):
    def __init__(self, item: BaseLayer):
        self._name = item.name
        self._opacity = item.opacity
        self._blend_mode = item.blend_mode
        self._visible = item.visible
        self._format = item.format
        self._transform = item._transform.copy()
        self._control = GeometryControllerSnapshot(item.control)
        self._mask = copy.copy(item._mask) if item._mask is not None else None
        self._effects = list(item._effects)
        self._item = item

    def restore(self) -> None:
        self._item.name = self._name
        self._item.opacity = self._opacity
        self._item.blend_mode = self._blend_mode
        self._item.visible = self._visible
        self._item.format = self._format
        self._item._transform = self._transform.copy()
        self._control.restore()
        self._item._mask = copy.copy(self._mask) if self._mask is not None else None
        self._item._effects = list(self._effects)

    def has_change(self, other: BaseLayerSnapshot) -> bool:
        return (
            self._name != other._name
            or self._opacity != other._opacity
            or self._blend_mode != other._blend_mode
            or self._visible != other._visible
            or self._format != other._format
            or self._transform != other._transform
            or self._control.has_change(other._control)
            or self._mask != other._mask
            or self._effects != other._effects
        )


class Command(ABC):
    def __init__(self, name: str, item: Any, value: Any):
        self._sealed = False
        self._item = item
        self._value = value
        self._name = name

    @property
    def item(self) -> Any:
        return self._item

    @property
    def value(self) -> Any:
        return self._value

    @abstractmethod
    def execute(self) -> None:
        pass

    @abstractmethod
    def undo(self) -> None:
        pass

    @abstractmethod
    def has_changes(self) -> bool:
        """Determina se o comando gerou alguma mutação de estado."""
        pass

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    def seal(self) -> None:
        self._sealed = True

    def can_merge(self, name: str, target: object) -> bool:
        if self.is_sealed:
            return False
        return self._name == name and self._item == target


class ReparentCommand(Command):
    """Comando relacional O(1) para mutações de hierarquia em Containers."""

    SNAPSHOT_REGISTRY = (
        (Container, ContainerSnapshot),
        (NodeContainerProtocol, NodeContainerSnapshot),
        (NullContainer, NullContainerSnapshot),
    )

    def __init__(
        self, name: str, item: Container, value: NodeContainerProtocol | Container
    ):

        if isinstance(value, LayerStack):
            raise ValueError(
                "A LayerStack is a Root object and cannot be added as a child."
            )

        super().__init__(name, item, value)
        self._old_item = _create_snapshot(item, self.SNAPSHOT_REGISTRY)
        self._old_value = _create_snapshot(value, self.SNAPSHOT_REGISTRY)
        self._old_pvalue = _create_snapshot(value.parent, self.SNAPSHOT_REGISTRY)
        self.pvalue = value.parent

    def seal(self) -> None:
        if not self._sealed:
            self._new_item = _create_snapshot(self.item, self.SNAPSHOT_REGISTRY)
            self._new_value = _create_snapshot(self.value, self.SNAPSHOT_REGISTRY)
            self._new_pvalue = _create_snapshot(self.pvalue, self.SNAPSHOT_REGISTRY)
            self._sealed = True

    def execute(self) -> None:
        if not self._sealed:
            return

        self._new_item.restore()
        self._new_value.restore()
        self._new_pvalue.restore()

    def undo(self) -> None:
        if not self._sealed:
            self.seal()

        self._old_item.restore()
        self._old_value.restore()
        self._old_pvalue.restore()

    def has_changes(self) -> bool:
        if not self._sealed:
            return True
        return self._old_item.has_change(self._new_item) or self._old_value.has_change(
            self._new_value
        )


class BaseLayerCommand(Command):
    """Gerencia mutações em propriedades da classe BaseLayer."""

    SNAPSHOT_REGISTRY = ((BaseLayer, BaseLayerSnapshot),)

    def __init__(self, name: str, item: BaseLayer, value: Any = None):
        super().__init__(name, item, value)
        self._old_item = _create_snapshot(item, self.SNAPSHOT_REGISTRY)

    def seal(self) -> None:
        if not self._sealed:
            self._new_item = _create_snapshot(self.item, self.SNAPSHOT_REGISTRY)
            self._sealed = True

    def execute(self) -> None:
        if not self._sealed:
            return
        self._new_item.restore()

    def undo(self) -> None:
        if not self._sealed:
            self.seal()
        self._old_item.restore()

    def has_changes(self) -> bool:
        if not self._sealed:
            return True
        return self._old_item.has_change(self._new_item)


class LayerImageSnapshot(StateSnapshot):
    def __init__(self, item: Layer):
        self._edits = list(item.edits)
        self._opacity_mask = (
            np.copy(item._opacity_mask) if item._opacity_mask is not None else None
        )
        self._item = item

    def restore(self) -> None:
        self._item._edits = deque(self._edits)
        self._item._opacity_mask = (
            np.copy(self._opacity_mask) if self._opacity_mask is not None else None
        )

    def has_change(self, other: LayerImageSnapshot) -> bool:
        if self._edits != other._edits:
            return True

        if self._opacity_mask is None and other._opacity_mask is not None:
            return True
        if self._opacity_mask is not None and other._opacity_mask is None:
            return True
        if self._opacity_mask is not None and other._opacity_mask is not None:
            if not np.array_equal(self._opacity_mask, other._opacity_mask):
                return True

        return False


class LayerImageCommand(Command):
    """Gerencia mutações exclusivas na camada de pixels (Edits e Masks)."""

    SNAPSHOT_REGISTRY = ((Layer, LayerImageSnapshot),)

    def __init__(self, name: str, item: Layer, value: Any = None):
        super().__init__(name, item, value)
        self._old_item = _create_snapshot(item, self.SNAPSHOT_REGISTRY)

    def seal(self) -> None:
        if not self._sealed:
            self._new_item = _create_snapshot(self.item, self.SNAPSHOT_REGISTRY)
            self._sealed = True

    def execute(self) -> None:
        if not self._sealed:
            return
        self._new_item.restore()

    def undo(self) -> None:
        if not self._sealed:
            self.seal()
        self._old_item.restore()

    def has_changes(self) -> bool:
        if not self._sealed:
            return True
        return self._old_item.has_change(self._new_item)


class MaskStateSnapshot(StateSnapshot):
    """Snapshot para atributos escalares de estado da Máscara."""

    def __init__(self, item: Mask, value: Any = None):
        self._item = item
        self._visible = item.visible
        self._invert = item.invert
        self._matrix = np.copy(item.matrix)

    def restore(self) -> None:
        self._item.visible = self._visible
        self._item.invert = self._invert
        self._item._matrix = np.copy(self._matrix)

    def has_change(self, other: Any) -> bool:
        if not isinstance(other, MaskStateSnapshot):
            return True
        return (
            self._visible != other._visible
            or self._invert != other._invert
            or not np.array_equal(self._matrix, other._matrix)
        )


class MaskImageSnapshot(StateSnapshot):
    """Snapshot atômico de pixels da Máscara (usa value como a chave/slice do ndarray)."""

    def __init__(self, item: Mask, value: Any = None):
        self._item = item
        self._key = value
        self._data = np.copy(item[value])

    def restore(self) -> None:
        self._item[self._key] = self._data

    def has_change(self, other: Any) -> bool:
        if not isinstance(other, MaskImageSnapshot):
            return True
        return not np.array_equal(self._data, other._data)


class MaskCommand(Command):
    """Gerencia mutações em instâncias de Mask (atributos e pixels)."""

    SNAPSHOT_MAP = {
        "__setitem__": MaskImageSnapshot,
    }
    DEFAULT_SNAPSHOT = MaskStateSnapshot

    def __init__(self, name: str, item: Mask, value: Any = None):
        super().__init__(name, item, value)
        snapshot_cls = self.SNAPSHOT_MAP.get(name, self.DEFAULT_SNAPSHOT)
        self._snapshot_cls = snapshot_cls
        self._old_item = snapshot_cls(item, value)

    def seal(self) -> None:
        if not self._sealed:
            self._new_item = self._snapshot_cls(self.item, self.value)
            self._sealed = True

    def execute(self) -> None:
        if not self._sealed:
            return
        self._new_item.restore()

    def undo(self) -> None:
        if not self._sealed:
            self.seal()
        self._old_item.restore()

    def has_changes(self) -> bool:
        if not self._sealed:
            return True
        return self._old_item.has_change(self._new_item)


class MacroCommand(Command):
    """Comando composto que encapsula múltiplos sub-comandos como uma única unidade atômica."""

    def __init__(self, name: str = "Macro", item: Any = None, value: Any = None):
        super().__init__(name, item, value)
        self._commands: list[Command] = []

    def add_command(self, cmd: Command) -> None:
        """Adiciona e sela o sub-comando anterior se houver."""
        if self._commands and not self._commands[-1]._sealed:
            self._commands[-1].seal()
        self._commands.append(cmd)

    def can_merge(self, name: str, target: Any) -> bool:
        return not self._sealed

    def seal(self) -> None:
        for cmd in self._commands:
            if not cmd._sealed:
                cmd.seal()
        self._commands = [c for c in self._commands if c.has_changes()]
        self._sealed = True

    def execute(self) -> None:
        for cmd in self._commands:
            cmd.execute()

    def undo(self) -> None:
        for cmd in reversed(self._commands):
            cmd.undo()

    def has_changes(self) -> bool:
        return len(self._commands) > 0
