from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anicrop.command import BaseLayerCommand, Command
from anicrop.reactive.registry import unwrap_call_args

if TYPE_CHECKING:
    from anicrop.history import GlobalHistory


class BaseFluentProxy:
    """Proxy genérico para objetos com interface fluente (method chaining)."""

    _MUTATING_METHODS: frozenset[str] = frozenset()
    _COMMAND_CLASS: type[Command] = BaseLayerCommand
    _COMMAND_NAME: str = "fluent_action"

    def __init__(self, target: Any, history: GlobalHistory, owner: Any = None):
        self._target = target
        self._history = history
        self._owner = owner if owner is not None else self
        self._has_mutated = False

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)

        if name in self._MUTATING_METHODS and callable(attr):

            def mutating_wrapper(*args: Any, **kwargs: Any) -> Any:
                clean_args, clean_kwargs = unwrap_call_args(args, kwargs)

                if self._history.is_active:
                    if not self._has_mutated:
                        self._history.start_action(
                            self._COMMAND_CLASS, self._COMMAND_NAME, self._owner
                        )
                        self._has_mutated = True
                    with self._history.disabled():
                        attr(*clean_args, **clean_kwargs)
                else:
                    attr(*clean_args, **clean_kwargs)

                return self

            return mutating_wrapper

        # Leitura passiva pura (não gera histórico)
        return attr

    def __eq__(self, other: Any) -> bool:
        target = getattr(self, "_target", self)
        other_target = getattr(other, "_target", other)
        return target is other_target or target == other_target

    def __del__(self) -> None:
        if getattr(self, "_has_mutated", False):
            history = getattr(self, "_history", None)
            if history is not None and history.is_active:
                history.commit()

    def __dir__(self):
        return dir(self._target)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self._target)})"


class ProxyComposer(BaseFluentProxy):
    """Proxy especialista para transformações afins fluentes (ComposerRel)."""

    _MUTATING_METHODS = frozenset({"rotate", "scale", "translate", "add_transform"})
    _COMMAND_CLASS = BaseLayerCommand
    _COMMAND_NAME = "transform"
