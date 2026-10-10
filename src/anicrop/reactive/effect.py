from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from anicrop.command import AdaptiveCommand, Command, EffectStackCommand
from anicrop.effect import Effect, EffectStack
from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.registry import unwrap_target

if TYPE_CHECKING:
    pass


class ProxyEffect(BaseHistoryProxy[Effect]):
    """Proxy reativo para instâncias de Effect e BoundEffect (rastreia mutações escalares via AdaptiveCommand)."""

    _DEFAULT_COMMAND: type[Command] = AdaptiveCommand

    def __repr__(self) -> str:
        name = getattr(self, "name", "Effect")
        return f'ProxyEffect(name="{name}")'


class ProxyEffectStack(BaseHistoryProxy[EffectStack]):
    """Proxy reativo para EffectStack (intercepta operações de coleção via EffectStackCommand)."""

    _ACTION_ROUTER: dict[str, type[Command]] = {
        "add": EffectStackCommand,
        "extend": EffectStackCommand,
        "insert": EffectStackCommand,
        "remove": EffectStackCommand,
        "pop": EffectStackCommand,
        "clear": EffectStackCommand,
        "move": EffectStackCommand,
        "swap": EffectStackCommand,
        "__delitem__": EffectStackCommand,
    }

    def _extract_command_value(
        self, name: str, cmd_cls: type, target: Any, args: tuple
    ) -> Any:
        registry = object.__getattribute__(self, "_registry")
        if cmd_cls is EffectStackCommand:
            if name in ("add", "remove", "move", "swap"):
                return registry.get_or_create(args[0])
            elif name == "insert":
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
        return f"ProxyEffectStack(count={len(self)})"
