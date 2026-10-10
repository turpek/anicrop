from __future__ import annotations

from anicrop.canvas import Canvas
from anicrop.composition import Combine
from anicrop.container import BaseLayer, Container, GroupLayer, LayerStack
from anicrop.effect import Effect, EffectStack
from anicrop.interfaces.canvas import AbstractCanvas
from anicrop.layer import Layer
from anicrop.mask import Mask
from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.canvas import ProxyCanvas
from anicrop.reactive.container import BaseContainerProxy, LayerStackProxy
from anicrop.reactive.effect import ProxyEffect, ProxyEffectStack
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
from anicrop.reactive.strategy import (
    CanvasLayoutProxy,
    CombineProxy,
    GroupContentProxy,
    GroupLayoutProxy,
    LayerContentProxy,
    LayerLayoutProxy,
    StrategyProxy,
)
from anicrop.transform import Composer

# Registro oficial dos proxies das entidades de domínio anicrop
ProxyRegistry.register(Canvas, ProxyCanvas)
ProxyRegistry.register(GroupLayer, GroupProxy)
ProxyRegistry.register(LayerStack, LayerStackProxy)
ProxyRegistry.register(Layer, ProxyLayer)
ProxyRegistry.register(BaseLayer, ProxyLayer)
ProxyRegistry.register(Mask, ProxyMask)
ProxyRegistry.register(Container, BaseContainerProxy)
ProxyRegistry.register(Combine, CombineProxy)
ProxyRegistry.register(EffectStack, ProxyEffectStack)
ProxyRegistry.register(Effect, ProxyEffect)

# Registro ABC virtual para suporte a isinstance(proxy, DomainType)
AbstractCanvas.register(ProxyCanvas)
Canvas.register(ProxyCanvas)
Container.register(BaseContainerProxy)
LayerStack.register(LayerStackProxy)
GroupLayer.register(GroupProxy)
BaseLayer.register(GroupProxy)
BaseLayer.register(ProxyLayer)
Layer.register(ProxyLayer)
Mask.register(ProxyMask)
Composer.register(ProxyComposer)
EffectStack.register(ProxyEffectStack)
Effect.register(ProxyEffect)

__all__ = [
    "BaseContainerProxy",
    "BaseFluentProxy",
    "BaseHistoryProxy",
    "CanvasLayoutProxy",
    "CombineProxy",
    "GroupContentProxy",
    "GroupLayoutProxy",
    "GroupProxy",
    "LayerContentProxy",
    "LayerLayoutProxy",
    "LayerStackProxy",
    "ProxyCanvas",
    "ProxyComposer",
    "ProxyEffect",
    "ProxyEffectStack",
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
