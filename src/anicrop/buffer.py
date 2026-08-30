from __future__ import annotations
import numpy as np

from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.interfaces.buffer import AbstractScratchBuffer
from anicrop.spatial import Region


class ScratchBuffer(AbstractScratchBuffer):
    """Buffer temporário reutilizável com alocação preguiçosa sob demanda."""

    def __init__(self) -> None:
        self._image: Image | None = None
        self._size: tuple[int, int] = (0, 0)
        self._format: ImageFormat = ImageFormat.RGBA
        self._used: bool = False

    @property
    def was_used(self) -> bool:
        """Indica se o buffer foi acessado desde a última chamada a configure."""
        return self._used

    def configure(
        self,
        size: tuple[int, int],
        fmt: ImageFormat = ImageFormat.RGBA,
    ) -> ScratchBuffer:
        """Configura as dimensões mínimas e o formato desejado para o próximo acesso."""
        self._size = size
        self._format = fmt
        self._used = False
        return self

    def _ensure_allocated(self) -> Image:
        """Aloca ou reaproveita o array de imagem subjacente, expandindo por fator 1.5x."""
        w, h = self._size
        if (
            self._image is None or
            self._image.height < h or
            self._image.width < w or
            self._image.format != self._format
        ):
            current_h = self._image.height if self._image is not None else 0
            current_w = self._image.width if self._image is not None else 0
            new_h = max(h, int(current_h * 1.5))
            new_w = max(w, int(current_w * 1.5))
            self._image = Image.new((new_w, new_h), self._format)

        return self._image.view(Region.from_size(w, h))

    def __getitem__(self, region: Region) -> np.ndarray:
        """Retorna o slice NumPy da região requisitada após garantir a alocação."""
        self._used = True
        view = self._ensure_allocated()
        return view[region]
