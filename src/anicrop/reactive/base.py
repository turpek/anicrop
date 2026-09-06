from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from anicrop.command import AdaptiveCommand, Command
from anicrop.reactive.fluent import BaseFluentProxy
from anicrop.reactive.registry import (
    get_registry_for_history,
    is_property_with_setter,
    unwrap_call_args,
    unwrap_target,
    wrap_domain_result,
)

if TYPE_CHECKING:
    from anicrop.history import GlobalHistory
    from anicrop.reactive.registry import ProxyRegistry

TargetT = TypeVar("TargetT")

_INTERNAL_PROXY_ATTRS = frozenset({
    "_target",
    "_history",
    "_registry",
    "_IGNORED_ATTRIBUTES",
    "_SPECIAL_WRAPPERS",
    "_ACTION_ROUTER",
    "_DEFAULT_COMMAND",
    "_resolve_command",
    "_extract_command_value",
    "__dict__",
    "__class__",
})


def execute_tracked_setattr(
    proxy: Any,
    target: Any,
    history: GlobalHistory,
    name: str,
    value: Any,
    cmd_cls: type[Command],
) -> None:
    """Executa atribuição no target gravando ação atômica e selada no histórico."""
    history.start_action(cmd_cls, name, proxy, value)
    with history.disabled():
        setattr(target, name, value)
    history.commit()


def resolve_setattr_command(
    target: Any,
    name: str,
    action_router: dict[str, type[Command]],
    default_cmd: type[Command],
) -> type[Command] | None:
    """Identifica a classe de comando para o atributo ou None se não for rastreável."""
    if name in action_router:
        return action_router[name]
    if is_property_with_setter(type(target), name) or hasattr(target, name):
        return default_cmd
    return None


def build_action_wrapper(
    proxy: Any,
    target: Any,
    history: GlobalHistory,
    registry: ProxyRegistry,
    name: str,
    func: Callable,
    cmd_cls: type[Command],
    extract_val_fn: Callable | None = None,
) -> Callable:
    """Constrói o wrapper para métodos mutadores mapeados no _ACTION_ROUTER."""

    def method_wrapper(*args: Any, **kwargs: Any) -> Any:
        clean_args, clean_kwargs = unwrap_call_args(args, kwargs)

        if not history.is_active:
            res = func(*clean_args, **clean_kwargs)
            return (
                proxy
                if res is target
                else wrap_domain_result(res, history, registry)
            )

        val = (
            extract_val_fn(name, cmd_cls, target, clean_args)
            if extract_val_fn
            else None
        )
        history.start_action(cmd_cls, name, proxy, val)

        with history.disabled():
            res = func(*clean_args, **clean_kwargs)
        history.commit()

        return (
            proxy
            if res is target
            else wrap_domain_result(res, history, registry)
        )

    return method_wrapper


def resolve_special_wrapper(
    wrapper_cls: type,
    attr: Any,
    history: GlobalHistory,
    registry: ProxyRegistry,
    owner: Any,
) -> Any:
    """Instancia o wrapper especialista correto (ProxyComposer ou StrategyProxy)."""
    if issubclass(wrapper_cls, BaseFluentProxy):
        return wrapper_cls(attr, history, owner=owner)
    return wrapper_cls(attr, history, registry=registry)


class BaseHistoryProxy(Generic[TargetT]):
    """Proxy 100% genérico e agnóstico: rastreia qualquer objeto Python."""

    _IGNORED_ATTRIBUTES: frozenset[str] = frozenset()
    _SPECIAL_WRAPPERS: dict[str, type] = {}
    _ACTION_ROUTER: dict[str, type[Command]] = {}
    _DEFAULT_COMMAND: type[Command] = AdaptiveCommand

    def __new__(
        cls, target: Any, history: GlobalHistory, registry: ProxyRegistry | None = None
    ):
        if isinstance(target, BaseHistoryProxy):
            return target

        reg = registry or get_registry_for_history(history)
        target_id = id(target)
        if target_id in reg._cache:
            return reg._cache[target_id]

        return super().__new__(cls)

    def __init__(
        self, target: TargetT, history: GlobalHistory, registry: ProxyRegistry | None = None
    ):
        if hasattr(self, "_target"):
            return

        if registry is None:
            registry = get_registry_for_history(history)

        super().__setattr__("_target", target)
        super().__setattr__("_history", history)
        super().__setattr__("_registry", registry)
        registry._cache[id(target)] = self

    def __eq__(self, other: Any) -> bool:
        target = object.__getattribute__(self, "_target")
        other_target = getattr(other, "_target", other)
        return target is other_target or target == other_target

    def __hash__(self) -> int:
        return hash(object.__getattribute__(self, "_target"))

    def _resolve_command(self, name: str) -> type[Command]:
        action_router = object.__getattribute__(self, "_ACTION_ROUTER")
        if name in action_router:
            return action_router[name]
        return object.__getattribute__(self, "_DEFAULT_COMMAND")

    def _extract_command_value(
        self, name: str, cmd_cls: type, target: Any, args: tuple
    ) -> Any:
        if name == "__setitem__" and args:
            return args[0]
        return None

    def __dir__(self):
        target = object.__getattribute__(self, "_target")
        return dir(target)

    def __getattribute__(self, name: str) -> Any:
        if name in _INTERNAL_PROXY_ATTRS or (
            name.startswith("_") and hasattr(type(self), name)
        ):
            return object.__getattribute__(self, name)

        target = object.__getattribute__(self, "_target")
        history = object.__getattribute__(self, "_history")
        registry = object.__getattribute__(self, "_registry")

        special_wrappers = object.__getattribute__(self, "_SPECIAL_WRAPPERS")
        if name in special_wrappers:
            attr = getattr(target, name)
            return resolve_special_wrapper(
                special_wrappers[name], attr, history, registry, owner=self
            )

        attr = getattr(target, name)
        action_router = object.__getattribute__(self, "_ACTION_ROUTER")

        if callable(attr) and name in action_router:
            cmd_cls = self._resolve_command(name)
            return build_action_wrapper(
                self,
                target,
                history,
                registry,
                name,
                attr,
                cmd_cls,
                self._extract_command_value,
            )

        if hasattr(self, "_registry") and not isinstance(attr, BaseHistoryProxy):
            return wrap_domain_result(attr, history, registry)

        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _INTERNAL_PROXY_ATTRS:
            super().__setattr__(name, value)
            return

        target = object.__getattribute__(self, "_target")
        history = object.__getattribute__(self, "_history")
        ignored = object.__getattribute__(self, "_IGNORED_ATTRIBUTES")
        clean_val = unwrap_target(value)

        if name in ignored or not history.is_active:
            setattr(target, name, clean_val)
            return

        action_router = object.__getattribute__(self, "_ACTION_ROUTER")
        default_cmd = object.__getattribute__(self, "_DEFAULT_COMMAND")
        cmd_cls = resolve_setattr_command(target, name, action_router, default_cmd)

        if cmd_cls is None:
            setattr(target, name, clean_val)
            return

        execute_tracked_setattr(self, target, history, name, clean_val, cmd_cls)

    def __getitem__(self, item: Any) -> Any:
        target = object.__getattribute__(self, "_target")
        return target[item]

    def __setitem__(self, item: Any, value: Any) -> None:
        target = object.__getattribute__(self, "_target")
        history = object.__getattribute__(self, "_history")
        ignored = object.__getattribute__(self, "_IGNORED_ATTRIBUTES")
        clean_val = unwrap_target(value)

        if "__setitem__" in ignored or not history.is_active:
            target[item] = clean_val
            return

        action_router = object.__getattribute__(self, "_ACTION_ROUTER")

        if "__setitem__" in action_router:
            cmd_cls = self._resolve_command("__setitem__")
            val_arg = self._extract_command_value(
                "__setitem__", cmd_cls, target, (item, clean_val)
            )
            history.start_action(cmd_cls, "__setitem__", self, val_arg)

            with history.disabled():
                target[item] = clean_val
            history.commit()
        else:
            target[item] = clean_val
