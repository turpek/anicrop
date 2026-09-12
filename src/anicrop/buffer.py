from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from anicrop.interfaces.buffer import AbstractImageBuffer
from anicrop.persistence.manager import manager_global

MMapMode = Literal["readonly", "r", "copyonwrite", "c", "readwrite", "r+", "write", "w+"]


def _cleanup_mmap_file(file_path: Path | None, auto_remove: bool) -> None:
    """Função desacoplada de finalização para remoção do arquivo em disco."""
    if auto_remove and file_path is not None:
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass


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
        auto_remove: bool = False,
    ) -> None:
        self._mmap = mmap_array
        self._file_path = Path(file_path) if file_path is not None else None
        self._auto_remove = auto_remove
        self._finalizer = (
            weakref.finalize(self, _cleanup_mmap_file, self._file_path, self._auto_remove)
            if self._file_path is not None and self._auto_remove
            else None
        )

    @property
    def auto_remove(self) -> bool:
        """Indica se o arquivo temporário será automaticamente removido ao ser desalocado."""
        return self._auto_remove

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        file_path: str | Path | None = None,
        auto_remove: bool | None = None,
    ) -> MMapBuffer:
        """Cria um MMapBuffer a partir de uma matriz NumPy gravando no workspace temporário."""
        required_bytes = array.nbytes
        is_temp = file_path is None
        if file_path is None:
            file_path = manager_global.get_temp_file_path(required_bytes=required_bytes)
        else:
            file_path = Path(file_path)
            manager_global.check_disk_space(file_path.parent, required_bytes)

        resolved_auto_remove = is_temp if auto_remove is None else auto_remove

        try:
            mm = np.memmap(
                str(file_path),
                dtype=array.dtype,
                mode="w+",
                shape=array.shape,
            )
            mm[...] = array
            mm.flush()
            return cls(mm, file_path=file_path, auto_remove=resolved_auto_remove)
        except BaseException:
            if resolved_auto_remove and file_path.exists():
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise

    @classmethod
    def create_empty(
        cls,
        shape: tuple[int, ...],
        dtype: Any = np.uint8,
        file_path: str | Path | None = None,
        auto_remove: bool | None = None,
    ) -> MMapBuffer:
        """Aloca um buffer de memória mapeada com formato e dimensões predefinidos."""
        resolved_dtype = np.dtype(dtype)
        required_bytes = int(np.prod(shape)) * resolved_dtype.itemsize
        is_temp = file_path is None
        if file_path is None:
            file_path = manager_global.get_temp_file_path(required_bytes=required_bytes)
        else:
            file_path = Path(file_path)
            manager_global.check_disk_space(file_path.parent, required_bytes)

        resolved_auto_remove = is_temp if auto_remove is None else auto_remove

        try:
            mm = np.memmap(
                str(file_path),
                dtype=resolved_dtype,
                mode="w+",
                shape=shape,
            )
            return cls(mm, file_path=file_path, auto_remove=resolved_auto_remove)
        except BaseException:
            if resolved_auto_remove and file_path.exists():
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise

    @classmethod
    def open_existing(
        cls,
        file_path: str | Path,
        shape: tuple[int, ...],
        dtype: Any = np.uint8,
        mode: MMapMode = "r+",
        auto_remove: bool = False,
    ) -> MMapBuffer:
        """Abre um arquivo binário memmap existente no disco."""
        path = Path(file_path)
        mm = np.memmap(str(path), dtype=dtype, mode=mode, shape=shape)
        return cls(mm, file_path=path, auto_remove=auto_remove)

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
        """Fecha o descritor de memória mapeada se aberto e desaloca arquivo temporário."""
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer()
        elif self._auto_remove and self._file_path is not None:
            try:
                self._file_path.unlink(missing_ok=True)
            except Exception:
                pass

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
