from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from anicrop.command import BaseLayerCommand, Command, LayerImageCommand
from anicrop.content import GroupContentStrategy, LayerContentStrategy
from anicrop.layout import GroupLayoutStrategy, LayerLayoutStrategy
from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.container import BaseContainerProxy
from anicrop.reactive.fluent import ProxyComposer
from anicrop.reactive.strategy import StrategyProxy

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
    }

    def _create_layout_strategy(self) -> Any:
        return LayerLayoutStrategy(cast(Any, self))

    def _create_content_strategy(self) -> Any:
        return LayerContentStrategy(cast(Any, self))

    def __getattribute__(self, name: str) -> Any:
        if name == "layout":
            history = object.__getattribute__(self, "_history")
            registry = object.__getattribute__(self, "_registry")
            strategy = self._create_layout_strategy()
            return StrategyProxy(strategy, history, registry=registry)

        if name == "content":
            history = object.__getattribute__(self, "_history")
            registry = object.__getattribute__(self, "_registry")
            strategy = self._create_content_strategy()
            return StrategyProxy(strategy, history, registry=registry)

        return super().__getattribute__(name)

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
    }

    def _create_layout_strategy(self) -> Any:
        return GroupLayoutStrategy(cast(Any, self))

    def _create_content_strategy(self) -> Any:
        return GroupContentStrategy(cast(Any, self))

    def __repr__(self) -> str:
        name = getattr(self, "name", "Group")
        return f'GroupProxy(name="{name}")'
