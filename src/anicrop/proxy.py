from __future__ import annotations

from anicrop.reactive import (
    BaseContainerProxy,
    BaseFluentProxy,
    BaseHistoryProxy,
    GroupProxy,
    LayerStackProxy,
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

# Alias para compatibilidade retroativa
ProxyContent = StrategyProxy

__all__ = [
    "BaseContainerProxy",
    "BaseFluentProxy",
    "BaseHistoryProxy",
    "GroupProxy",
    "LayerStackProxy",
    "ProxyComposer",
    "ProxyContent",
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
