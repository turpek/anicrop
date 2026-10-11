from __future__ import annotations

from anicrop.canvas import Canvas
from anicrop.reactive.base import BaseHistoryProxy
from anicrop.reactive.strategy import CanvasLayoutProxy


class ProxyCanvas(BaseHistoryProxy[Canvas]):
    """Proxy reativo para Canvas (gerencia enquadramento e moldura com histórico)."""

    _SPECIAL_WRAPPERS: dict[str, type] = {
        "layout": CanvasLayoutProxy,
    }

    def __repr__(self) -> str:
        region = getattr(self, "region", None)
        return f"ProxyCanvas(region={region})"
