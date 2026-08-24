from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class LayerProtocol(Protocol):
    """Protocolo estrutural simples para camadas com edicoes e matrizes."""

    @property
    def _edits(self) -> Any: ...

    @property
    def matrix(self) -> np.ndarray: ...

    @property
    def base(self) -> Any: ...

    @property
    def parent(self) -> Any: ...

    @property
    def transform(self) -> Any: ...
