from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anicrop.command import BaseLayerCommand, Command, LayerImageCommand
from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.container import BaseContainerProxy
from anicrop.reactive.fluent import ProxyComposer
from anicrop.reactive.strategy import (
    GroupContentProxy,
    GroupLayoutProxy,
    LayerContentProxy,
    LayerLayoutProxy,
)

if TYPE_CHECKING:
    pass


class ProxyLayer(BaseHistoryProxy["Layer"]):
    """Proxy dedicado à classe Layer (manipula propriedades escalares, transformações e edições)."""

    _ACTION_ROUTER: dict[str, type[Command]] = {
        "name": BaseLayerCommand,
        "opacity": BaseLayerCommand,
        "blend_mode": BaseLayerCommand,
        "visible": BaseLayerCommand,
        "transform": BaseLayerCommand,
        "x": BaseLayerCommand,
        "y": BaseLayerCommand,
        "region": BaseLayerCommand,
        "frame": BaseLayerCommand,
        "set_mask": BaseLayerCommand,
        "remove_mask": BaseLayerCommand,
        "clear_mask": BaseLayerCommand,
        "add_effect": BaseLayerCommand,
        "bind_effect": BaseLayerCommand,
        "remove_effect": BaseLayerCommand,
        "clear_effects": BaseLayerCommand,
        "add_edit": LayerImageCommand,
        "opacity_mask": LayerImageCommand,
    }
    _SPECIAL_WRAPPERS: dict[str, type] = {
        "transform": ProxyComposer,
        "layout": LayerLayoutProxy,
        "content": LayerContentProxy,
    }

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "parent":
            raise AttributeError(
                "Direct assignment to 'parent' is not supported. "
                "Use container methods like 'parent.append(child)' or 'parent.remove(child)' instead."
            )
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        name = getattr(self, "name", "Layer")
        return f'ProxyLayer(name="{name}")'


class GroupProxy(BaseContainerProxy, ProxyLayer):
    """Proxy dedicado a GroupLayer (combina operações de contêiner e camada composta)."""

    _ACTION_ROUTER: dict[str, type[Command]] = {
        **BaseContainerProxy._ACTION_ROUTER,
        **ProxyLayer._ACTION_ROUTER,
    }
    _SPECIAL_WRAPPERS: dict[str, type] = {
        "transform": ProxyComposer,
        "layout": GroupLayoutProxy,
        "content": GroupContentProxy,
    }

    def __repr__(self) -> str:
        name = getattr(self, "name", "Group")
        return f'GroupProxy(name="{name}")'
