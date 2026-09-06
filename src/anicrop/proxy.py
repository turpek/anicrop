from __future__ import annotations

from anicrop.reactive import (
    BaseContainerProxy,
    BaseFluentProxy,
    BaseHistoryProxy,
    CanvasLayoutProxy,
    GroupContentProxy,
    GroupLayoutProxy,
    GroupProxy,
    LayerContentProxy,
    LayerLayoutProxy,
    LayerStackProxy,
    ProxyCanvas,
    ProxyComposer,
    ProxyLayer,
    ProxyMask,
    ProxyRegistry,
    StrategyProxy,
    get_registry_for_history,
    is_property_with_setter,
    unwrap_call_args,
    unwrap_target,
    wrap_domain_result,
)

# Aliases para compatibilidade retroativa
ProxyContent = StrategyProxy
ProxyLayout = StrategyProxy

__all__ = [
    "BaseContainerProxy",
    "BaseFluentProxy",
    "BaseHistoryProxy",
    "CanvasLayoutProxy",
    "GroupContentProxy",
    "GroupLayoutProxy",
    "GroupProxy",
    "LayerContentProxy",
    "LayerLayoutProxy",
    "LayerStackProxy",
    "ProxyCanvas",
    "ProxyComposer",
    "ProxyContent",
    "ProxyLayer",
    "ProxyLayout",
    "ProxyMask",
    "ProxyRegistry",
    "StrategyProxy",
    "get_registry_for_history",
    "is_property_with_setter",
    "unwrap_call_args",
    "unwrap_target",
    "wrap_domain_result",
]
