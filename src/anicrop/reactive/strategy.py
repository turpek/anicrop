from __future__ import annotations

from typing import Any

from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.registry import unwrap_call_args, wrap_domain_result


class StrategyProxy(BaseHistoryProxy):
    """Proxy genérico para estratégias: envolve métodos públicos em with history.atomic(name)."""

    def _history_context(self, action_name: str) -> Any:
        """Hook de contexto de histórico (padrão: transação atômica agrupada)."""
        history = object.__getattribute__(self, "_history")
        return history.atomic(action_name)

    def __getattribute__(self, name: str) -> Any:
        if name in (
            "_target",
            "_history",
            "_registry",
            "_history_context",
            "__dict__",
            "__class__",
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
