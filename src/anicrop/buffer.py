from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from anicrop.interfaces.buffer import AbstractImageBuffer
from anicrop.persistence.manager import manager_global

MMapMode = Literal["readonly", "r", "copyonwrite", "c", "readwrite", "r+", "write", "w+"]


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

    def get_lod(self, level: int, threshold_pixels: int | None = None) -> ArrayBuffer:
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

    @classmethod
    def open_existing(
        cls,
        file_path: str | Path,
        shape: tuple[int, ...],
        dtype: Any = np.uint8,
        mode: MMapMode = "r+",
    ) -> MMapBuffer:
        """Abre um arquivo binário memmap existente no disco."""
        path = Path(file_path)
        mm = np.memmap(str(path), dtype=dtype, mode=mode, shape=shape)
        return cls(mm, file_path=path)

    @property
    def file_path(self) -> Path | None:
        """Caminho do arquivo temporário no disco, se aplicável."""
        return self._file_path

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
        if (
            hasattr(self, "_mmap")
            and hasattr(self._mmap, "_mmap")
            and self._mmap._mmap is not None
        ):
            try:
                self._mmap._mmap.close()
            except Exception:
                pass

    def get_lod(
        self, level: int, threshold_pixels: int | None = None
    ) -> AbstractImageBuffer:
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

        if threshold_pixels is not None and (new_w * new_h > threshold_pixels):
            return MMapBuffer.from_array(resized)

        return ArrayBuffer(resized)
