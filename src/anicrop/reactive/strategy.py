from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anicrop.content import GroupContentStrategy, LayerContentStrategy
from anicrop.layout import CanvasLayoutStrategy, GroupLayoutStrategy, LayerLayoutStrategy
from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.registry import unwrap_call_args, wrap_domain_result

if TYPE_CHECKING:
    from anicrop.history import GlobalHistory
    from anicrop.reactive.registry import ProxyRegistry


class StrategyProxy(BaseHistoryProxy):
    """Proxy genérico para estratégias: envolve métodos públicos em with history.atomic(name)."""

    def _history_context(self, action_name: str) -> Any:
        """Hook de contexto de histórico (padrão: transação atômica agrupada)."""
        history = object.__getattribute__(self, "_history")
        return history.atomic(action_name)

    def __getattribute__(self, name: str) -> Any:
        if (
            name in (
                "_target",
                "_history",
                "_registry",
                "_history_context",
                "__dict__",
                "__class__",
            )
            or (name.startswith("_") and hasattr(type(self), name))
        ):
            return object.__getattribute__(self, name)

        target = object.__getattribute__(self, "_target")
        attr = getattr(target, name)

        if callable(attr) and not name.startswith("_"):
            history = object.__getattribute__(self, "_history")
            registry = object.__getattribute__(self, "_registry")

            def method_wrapper(*args: Any, **kwargs: Any) -> Any:
                clean_args, clean_kwargs = unwrap_call_args(args, kwargs)

                if not history.is_active:
                    res = attr(*clean_args, **clean_kwargs)
                    return (
                        self
                        if res is target
                        else wrap_domain_result(res, history, registry)
                    )

                context = object.__getattribute__(self, "_history_context")(name)
                with context:
                    res = attr(*clean_args, **clean_kwargs)
                    return (
                        self
                        if res is target
                        else wrap_domain_result(res, history, registry)
                    )

            return method_wrapper

        return attr

    def __repr__(self) -> str:
        target = object.__getattribute__(self, "_target")
        return f"StrategyProxy({type(target).__name__})"


class OwnerBoundStrategyProxy(StrategyProxy):
    """Proxy de estratégia vinculado ao objeto de domínio proprietário (owner)."""

    _STRATEGY_CLS: type

    def __new__(
        cls, owner: Any, history: GlobalHistory, registry: ProxyRegistry | None = None
    ):
        return object.__new__(cls)

    def __init__(
        self, owner: Any, history: GlobalHistory, registry: ProxyRegistry | None = None
    ) -> None:
        strategy_cls = getattr(type(self), "_STRATEGY_CLS")
        strategy = strategy_cls(owner)
        super().__init__(strategy, history, registry=registry)


class LayerLayoutProxy(OwnerBoundStrategyProxy):
    """Proxy especialista para a estratégia de layout de uma camada folha (Layer)."""

    _STRATEGY_CLS = LayerLayoutStrategy


class LayerContentProxy(OwnerBoundStrategyProxy):
    """Proxy especialista para a estratégia de manipulação de conteúdo de uma camada (Layer)."""

    _STRATEGY_CLS = LayerContentStrategy


class GroupLayoutProxy(OwnerBoundStrategyProxy):
    """Proxy especialista para a estratégia de layout de um grupo (GroupLayer)."""

    _STRATEGY_CLS = GroupLayoutStrategy


class GroupContentProxy(OwnerBoundStrategyProxy):
    """Proxy especialista para a estratégia de conteúdo de um grupo (GroupLayer)."""

    _STRATEGY_CLS = GroupContentStrategy


class CanvasLayoutProxy(OwnerBoundStrategyProxy):
    """Proxy especialista para a estratégia de layout do Canvas."""

    _STRATEGY_CLS = CanvasLayoutStrategy
