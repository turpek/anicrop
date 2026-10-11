from __future__ import annotations

from typing import Any, Iterator

from anicrop.command import Command, StackCommand
from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.registry import unwrap_target
from anicrop.stack import NamedStack


class ProxyNamedStack(BaseHistoryProxy[NamedStack]):
    """Proxy reativo genérico para NamedStack (intercepta operações de coleção via StackCommand)."""

    _ACTION_ROUTER: dict[str, type[Command]] = {
        "add": StackCommand,
        "append": StackCommand,
        "extend": StackCommand,
        "insert": StackCommand,
        "remove": StackCommand,
        "pop": StackCommand,
        "clear": StackCommand,
        "move": StackCommand,
        "swap": StackCommand,
        "__delitem__": StackCommand,
        "__setitem__": StackCommand,
    }

    def _extract_command_value(
        self, name: str, cmd_cls: type, target: Any, args: tuple
    ) -> Any:
        registry = object.__getattribute__(self, "_registry")
        if issubclass(cmd_cls, StackCommand):
            if name in ("add", "append", "remove", "move", "swap"):
                return registry.get_or_create(args[0])
            elif name in ("insert", "__setitem__"):
                return registry.get_or_create(args[1])
            elif name == "extend":
                return args[0]
            elif name in ("pop", "__delitem__"):
                idx = args[0] if args else -1
                return registry.get_or_create(target[idx])
        return None

    def __iter__(self) -> Iterator[Any]:
        registry = object.__getattribute__(self, "_registry")
        for item in object.__getattribute__(self, "_target"):
            yield registry.get_or_create(item)

    def __reversed__(self) -> Iterator[Any]:
        registry = object.__getattribute__(self, "_registry")
        for item in reversed(object.__getattribute__(self, "_target")):
            yield registry.get_or_create(item)

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_target"))

    def __contains__(self, item: Any) -> bool:
        target = object.__getattribute__(self, "_target")
        clean_item = unwrap_target(item)
        return clean_item in target

    def __getitem__(self, item: Any) -> Any:
        registry = object.__getattribute__(self, "_registry")
        raw_item = object.__getattribute__(self, "_target")[item]
        if isinstance(raw_item, list):
            return [registry.get_or_create(e) for e in raw_item]
        return registry.get_or_create(raw_item)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(count={len(self)})"
