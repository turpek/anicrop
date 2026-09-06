from __future__ import annotations

from anicrop.container import BaseLayer, Container, GroupLayer, LayerStack
from anicrop.layer import Layer
from anicrop.mask import Mask
from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.container import BaseContainerProxy, LayerStackProxy
from anicrop.reactive.fluent import BaseFluentProxy, ProxyComposer
from anicrop.reactive.layer import GroupProxy, ProxyLayer
from anicrop.reactive.mask import ProxyMask
from anicrop.reactive.registry import (
    ProxyRegistry,
    get_registry_for_history,
    is_property_with_setter,
    unwrap_call_args,
    unwrap_target,
    wrap_domain_result,
)
from anicrop.reactive.strategy import StrategyProxy

# Registro oficial dos proxies das entidades de domínio anicrop
ProxyRegistry.register(GroupLayer, GroupProxy)
ProxyRegistry.register(LayerStack, LayerStackProxy)
ProxyRegistry.register(Layer, ProxyLayer)
ProxyRegistry.register(BaseLayer, ProxyLayer)
ProxyRegistry.register(Mask, ProxyMask)
ProxyRegistry.register(Container, BaseContainerProxy)

__all__ = [
    "BaseContainerProxy",
    "BaseFluentProxy",
    "BaseHistoryProxy",
    "GroupProxy",
    "LayerStackProxy",
    "ProxyComposer",
    "ProxyLayer",
    "ProxyMask",
    "ProxyRegistry",
    "StrategyProxy",
    "get_registry_for_history",
    "is_property_with_setter",
    "unwrap_call_args",
    "unwrap_target",
    "wrap_domain_result",
]
