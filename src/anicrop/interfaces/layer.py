from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class LayerProtocol(Protocol):
    """Protocolo estrutural simples para camadas com edicoes e matrizes."""
    _edits: Any
    matrix: np.ndarray
    base: Any
    parent: Any
    transform: Any
