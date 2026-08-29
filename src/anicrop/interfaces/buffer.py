from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from anicrop.enums import ImageFormat
    from anicrop.spatial import Region


class AbstractScratchBuffer(ABC):
    """Classe base abstrata para buffers temporários reutilizáveis com alocação preguiçosa."""

    @abstractmethod
    def configure(
        self,
        size: tuple[int, int],
        fmt: ImageFormat = ...,
    ) -> AbstractScratchBuffer:
        """Configura a intenção de dimensões e formato para a próxima operação."""
        ...

    @abstractmethod
    def __getitem__(self, region: Region) -> np.ndarray:
        """Retorna uma fatia do buffer subjacente, alocando sob demanda se necessário."""
        ...
