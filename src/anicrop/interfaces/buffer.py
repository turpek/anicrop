from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from anicrop.enums import ImageFormat
    from anicrop.spatial import Region


class AbstractScratchBuffer(ABC):
    """Classe base abstrata para buffers temporários reutilizáveis com alocação preguiçosa."""

    @property
    @abstractmethod
    def was_used(self) -> bool:
        """Indica se o buffer foi acessado desde a última chamada a configure."""
        pass

    @abstractmethod
    def configure(
        self,
        size: tuple[float, float],
        fmt: ImageFormat = ...,
    ) -> AbstractScratchBuffer:
        """Configura a intenção de dimensões e formato para a próxima operação."""
        pass

    @abstractmethod
    def __getitem__(self, region: Region) -> np.ndarray:
        """Retorna uma fatia do buffer subjacente, alocando sob demanda se necessário."""
        pass


class AbstractImageBuffer(ABC):
    """Contrato abstrato base para backends de armazenamento de dados de imagem."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Dimensões do buffer (altura, largura, canais) ou (altura, largura)."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Tipo de dado dos elementos (ex: np.uint8)."""
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Número de dimensões (2 ou 3)."""
        pass

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def channels(self) -> int:
        return self.shape[2] if self.ndim == 3 else 1

    @abstractmethod
    def __getitem__(self, key: Any) -> np.ndarray:
        """Fatia e retorna um recorte como array NumPy."""
        pass

    def __setitem__(self, key: Any, value: Any) -> None:
        """Modifica uma fatia do buffer. Backends somente-leitura devem lançar TypeError."""
        raise TypeError(f"{type(self).__name__} does not support item assignment")

    @abstractmethod
    def get_lod(
        self, level: int, threshold_pixels: int | None = None
    ) -> AbstractImageBuffer:
        """Gera ou extrai o nível de resolução (LOD = 1/2^level)."""
        pass
