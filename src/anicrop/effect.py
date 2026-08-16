from __future__ import annotations
from typing import Protocol, runtime_checkable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from anicrop.frame import BaseFrame
    from anicrop.image import Image


@runtime_checkable
class Effect(Protocol):
    """Protocolo formal para qualquer efeito ou filtro de processamento de pixels."""

    def prepare(self, frame: BaseFrame) -> None:
        """Etapa preliminar para preparar texturas, pré-cálculos ou métricas espaciais."""
        ...

    def get_padding(self) -> tuple[int, int, int, int]:
        """Retorna a margem extra (top, right, bottom, left) necessária para efeitos de expansão."""
        ...

    def apply(self, image: Image, matrix: np.ndarray | None = None) -> Image:
        """Processa e transforma o buffer de imagem recebendo opcionalmente a matriz espacial ativa."""
        ...

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        """Tenta combinar este efeito com outro, retornando o efeito unificado ou None."""
        ...
