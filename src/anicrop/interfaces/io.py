from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from anicrop.enums import ImageFormat
    from anicrop.interfaces.buffer import AbstractImageBuffer
    from anicrop.spatial import Region


@dataclass(frozen=True)
class SaveOptions:
    """Opções de codificação para salvar imagens no disco."""

    quality: int = 90
    lossless: bool = False
    compression_level: int = 6
    bg_color: tuple[int, ...] = (255, 255, 255)
    strip_metadata: bool = False
    dpi: tuple[int, int] | None = None


class AbstractImageIO(ABC):
    """Contrato formal para decodificadores e codificadores de imagens."""

    @abstractmethod
    def read(
        self,
        file_path: str | Path,
        format: ImageFormat | None = None,
        shrink: int = 1,
        roi: Region | None = None,
    ) -> tuple[np.ndarray, ImageFormat, tuple[int, int]]:
        """Lê e decodifica uma imagem a partir do disco.

        Args:
            file_path: Caminho do arquivo de imagem.
            format: Formato desejado. Se None, auto-detecta o formato nativo do arquivo.
            shrink: Fator de redução direta no decoder (ex: 2 para metade de largura/altura).
            roi: Recorte espacial opcional para ler apenas uma área sem decodificar o todo.

        Returns:
            Tupla contendo (array de dados, formato resolvido, tamanho original (largura, altura)).
        """
        pass

    @abstractmethod
    def write(
        self,
        file_path: str | Path,
        data: AbstractImageBuffer | np.ndarray,
        format: ImageFormat,
        options: SaveOptions | None = None,
    ) -> None:
        """Codifica e grava a imagem no disco.

        Args:
            file_path: Caminho de destino no disco.
            data: Matriz de pixels em memória.
            format: Formato dos canais em memória (RGB, RGBA, GRAY, etc.).
            options: Configurações de qualidade e compressão.
        """
        pass

    @abstractmethod
    def get_size(self, file_path: str | Path) -> tuple[int, int]:
        """Extrai as dimensões (largura, altura) da imagem lendo apenas o cabeçalho."""
        pass

    @abstractmethod
    def read_large(
        self,
        file_path: str | Path,
        format: ImageFormat | None = None,
    ) -> tuple[Any, ImageFormat]:
        """Abre imagens de altíssima resolução (>=8192px) utilizando a estratégia especializada do backend.

        Args:
            file_path: Caminho do arquivo de imagem.
            format: Formato de cor desejado ou None para auto-detecção.

        Returns:
            Tupla contendo (buffer de dados ou array, formato resolvido).
        """
        pass
