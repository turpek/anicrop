from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from anicrop.command import Command, ReparentCommand
from anicrop.container import Container
from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.registry import unwrap_target

if TYPE_CHECKING:
    pass


class BaseContainerProxy(BaseHistoryProxy[Container]):
    """Proxy base para contêineres de camadas com suporte a comandos relacionais."""

    _ACTION_ROUTER: dict[str, type[Command]] = {
        "append": ReparentCommand,
        "insert": ReparentCommand,
        "remove": ReparentCommand,
        "move": ReparentCommand,
        "pop": ReparentCommand,
    }

    def _extract_command_value(
        self, name: str, cmd_cls: type, target: Any, args: tuple
    ) -> Any:
        registry = object.__getattribute__(self, "_registry")
        if cmd_cls is ReparentCommand:
            if name in ("append", "remove", "move"):
                return registry.get_or_create(args[0])
            elif name == "insert":
                return registry.get_or_create(args[1])
            elif name == "pop":
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
        return registry.get_or_create(raw_item)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "parent":
            raise AttributeError(
                "Direct assignment to 'parent' is not supported. "
                "Use container methods like 'parent.append(child)' or 'parent.remove(child)' instead."
            )
        super().__setattr__(name, value)


class LayerStackProxy(BaseContainerProxy):
    """Proxy dedicado a LayerStack (Root Container)."""

    def __repr__(self) -> str:
        return f"LayerStackProxy(count={len(self)})"
