from __future__ import annotations

from anicrop.command import AdaptiveCommand, Command
from anicrop.edit_layer import EditLayer
from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.stack import ProxyNamedStack


class ProxyEdit(BaseHistoryProxy[EditLayer]):
    """Proxy reativo para EditLayer (rastreia mutações escalares via AdaptiveCommand)."""

    _DEFAULT_COMMAND: type[Command] = AdaptiveCommand

    def __repr__(self) -> str:
        return f'ProxyEdit(name="{self.name}")'


class ProxyEditStack(ProxyNamedStack):
    """Proxy reativo para EditStack (intercepta operações de coleção via StackCommand)."""

    pass
