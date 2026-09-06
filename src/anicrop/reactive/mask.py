from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anicrop.command import Command, MaskCommand
from anicrop.mask import Mask
from anicrop.reactive.base import BaseHistoryProxy

if TYPE_CHECKING:
    pass


class ProxyMask(BaseHistoryProxy[Mask]):
    """Proxy reativo para instâncias de Mask (gerencia micro-snapshots de fatias e estado)."""

    _ACTION_ROUTER: dict[str, type[Command]] = {
        "__setitem__": MaskCommand,
        "visible": MaskCommand,
        "invert": MaskCommand,
        "offset": MaskCommand,
    }
    _DEFAULT_COMMAND: type[Command] = MaskCommand

    def _extract_command_value(
        self, name: str, cmd_cls: type, target: Any, args: tuple
    ) -> Any:
        if name == "__setitem__" and args:
            return args[0]
        return None

    def __repr__(self) -> str:
        name = getattr(self, "name", "Mask")
        return f'ProxyMask(name="{name}")'
