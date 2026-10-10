from __future__ import annotations

from typing import TYPE_CHECKING

from anicrop.command import AdaptiveCommand, Command
from anicrop.effect import Effect
from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.stack import ProxyNamedStack

if TYPE_CHECKING:
    pass


class ProxyEffect(BaseHistoryProxy[Effect]):
    """Proxy reativo para instâncias de Effect e BoundEffect (rastreia mutações escalares via AdaptiveCommand)."""

    _DEFAULT_COMMAND: type[Command] = AdaptiveCommand

    def __repr__(self) -> str:
        return f'ProxyEffect(name="{self.name}")'


class ProxyEffectStack(ProxyNamedStack):
    """Proxy reativo para EffectStack (intercepta operações de coleção via StackCommand)."""

    pass
