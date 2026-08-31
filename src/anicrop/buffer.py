from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import zarr

from anicrop.enums import ImageFormat
from anicrop.interfaces.buffer import AbstractImageBuffer, AbstractScratchBuffer
from anicrop.persistence.manager import manager_global
from anicrop.spatial import Region

if TYPE_CHECKING:
    from anicrop.image import Image


class ArrayBuffer(AbstractImageBuffer):
    """Adaptador de buffer para matrizes NumPy em memória RAM."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return self._array.ndim

    @property
    def __array_interface__(self) -> dict[str, Any]:
        return self._array.__array_interface__

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self._array, dtype=dtype)

    def __eq__(self, other: Any) -> Any:
        other_arr = other._array if isinstance(other, ArrayBuffer) else other
        return self._array == other_arr

    def __getitem__(self, key: Any) -> np.ndarray:
        return self._array[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._array[key] = value

    def get_lod(self, level: int) -> ArrayBuffer:
        """Retorna uma versão reduzida em memória RAM usando cv2.INTER_AREA."""
        if level <= 0:
            return self
        factor = 2.0 ** (-level)
        new_w = max(1, int(self.width * factor))
        new_h = max(1, int(self.height * factor))
        resized = cv2.resize(self._array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2 and self.ndim == 3:
            resized = resized[..., np.newaxis]
        return ArrayBuffer(resized)


class ZarrBuffer(AbstractImageBuffer):
    """Adaptador de buffer para arrays particionados Zarr out-of-core em disco."""

    def __init__(self, zarr_array: zarr.Array) -> None:
        self._zarr = zarr_array

    @property
    def shape(self) -> tuple[int, ...]:
        return self._zarr.shape

    @property
    def dtype(self) -> np.dtype:
        return self._zarr.dtype

    @property
    def ndim(self) -> int:
        return self._zarr.ndim

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self._zarr[...], dtype=dtype)

    def __getitem__(self, key: Any) -> np.ndarray:
        return self._zarr[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._zarr[key] = value

    def get_lod(self, level: int) -> AbstractImageBuffer:
        """Gera um novo nível de resolução (LOD) reduzido sob demanda."""
        if level <= 0:
            return self
        factor = 2.0 ** (-level)
        new_w = max(1, int(self.width * factor))
        new_h = max(1, int(self.height * factor))

        raw = self._zarr[...]
        resized = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2 and self.ndim == 3:
            resized = resized[..., np.newaxis]

        if new_w * new_h > 4096 * 4096:
            zarr_dir = manager_global.workspace_path / f"{uuid.uuid4().hex}.zarr"
            channels = resized.shape[2] if resized.ndim == 3 else 1
            z_arr = zarr.open_array(
                str(zarr_dir),
                mode="w",
                shape=resized.shape,
                chunks=(min(512, new_h), min(512, new_w), channels),
                dtype=np.uint8,
            )
            z_arr[...] = resized
            return ZarrBuffer(z_arr)

        return ArrayBuffer(resized)


class MMapBuffer(AbstractImageBuffer):
    """Adaptador de buffer baseado em np.memmap (mapeamento de memória / tmpfs / /dev/shm)."""

    def __init__(
        self,
        mmap_array: np.memmap,
        file_path: str | Path | None = None,
    ) -> None:
        self._mmap = mmap_array
        self._file_path = Path(file_path) if file_path is not None else None

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        file_path: str | Path | None = None,
    ) -> MMapBuffer:
        """Cria um MMapBuffer a partir de uma matriz NumPy gravando no workspace temporário."""
        if file_path is None:
            file_path = manager_global.workspace_path / f"mmap_{uuid.uuid4().hex}.raw"
        else:
            file_path = Path(file_path)

        mm = np.memmap(
            str(file_path),
            dtype=array.dtype,
            mode="w+",
            shape=array.shape,
        )
        mm[...] = array
        mm.flush()
        return cls(mm, file_path=file_path)

    @classmethod
    def create_empty(
        cls,
        shape: tuple[int, ...],
        dtype: Any = np.uint8,
        file_path: str | Path | None = None,
    ) -> MMapBuffer:
        """Aloca um buffer de memória mapeada com formato e dimensões predefinidos."""
        if file_path is None:
            file_path = manager_global.workspace_path / f"mmap_{uuid.uuid4().hex}.raw"
        else:
            file_path = Path(file_path)

        mm = np.memmap(
            str(file_path),
            dtype=dtype,
            mode="w+",
            shape=shape,
        )
        return cls(mm, file_path=file_path)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._mmap.shape

    @property
    def dtype(self) -> np.dtype:
        return self._mmap.dtype

    @property
    def ndim(self) -> int:
        return self._mmap.ndim

    @property
    def __array_interface__(self) -> dict[str, Any]:
        return self._mmap.__array_interface__

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self._mmap, dtype=dtype)

    def __eq__(self, other: Any) -> Any:
        other_arr = other._mmap if isinstance(other, MMapBuffer) else other
        return self._mmap == other_arr

    def __getitem__(self, key: Any) -> np.ndarray:
        return self._mmap[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._mmap[key] = value

    def flush(self) -> None:
        """Sincroniza as alterações de memória para o arquivo subjacente."""
        self._mmap.flush()

    def close(self) -> None:
        """Fecha o descritor de memória mapeada se aberto."""
        if hasattr(self._mmap, "_mmap") and self._mmap._mmap is not None:
            self._mmap._mmap.close()

    def get_lod(self, level: int) -> AbstractImageBuffer:
        """Gera um nível de resolução reduzido sob demanda."""
        if level <= 0:
            return self
        factor = 2.0 ** (-level)
        new_w = max(1, int(self.width * factor))
        new_h = max(1, int(self.height * factor))

        raw = np.asarray(self._mmap)
        resized = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2 and self.ndim == 3:
            resized = resized[..., np.newaxis]

        if new_w * new_h > 4096 * 4096:
            return MMapBuffer.from_array(resized)

        return ArrayBuffer(resized)


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
        size: tuple[float, float],
        fmt: ImageFormat = ImageFormat.RGBA,
    ) -> ScratchBuffer:
        """Configura as dimensões mínimas e o formato desejado para o próximo acesso."""
        self._size = (int(round(size[0])), int(round(size[1])))
        self._format = fmt
        self._used = False
        return self

    def _ensure_allocated(self) -> Image:
        """Aloca ou reaproveita o array de imagem subjacente, expandindo por fator 1.5x."""
        from anicrop.image import Image

        w, h = self._size
        if (
            self._image is None
            or self._image.height < h
            or self._image.width < w
            or self._image.format != self._format
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
