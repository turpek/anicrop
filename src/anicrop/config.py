from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anicrop.interfaces.io import AbstractImageIO

import numpy as np

from anicrop.io.registry import get_default_backend_name, set_default_backend
from anicrop.persistence.manager import DEFAULT_MIN_DISK_HEADROOM, manager_global

DEFAULT_MEMORY_THRESHOLD: int = 8192 * 8192  # 64 MP (8K x 8K)
DEFAULT_BACKEND: str = "opencv"
DEFAULT_DTYPE: np.dtype = np.dtype(np.uint8)
DEFAULT_HARD_MASK_THRESHOLD: int = 128
DEFAULT_SOLID_FILL_THRESHOLD: int = 200
SUPPORTED_DTYPES: tuple[np.dtype, ...] = (
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.float32),
)


class Config:
    """Configuração global centralizada do anicrop."""

    def __init__(self) -> None:
        self._backend: str = DEFAULT_BACKEND
        self._memory_threshold: int | None = DEFAULT_MEMORY_THRESHOLD
        self._dtype: np.dtype = DEFAULT_DTYPE
        self._hard_mask_threshold: int = DEFAULT_HARD_MASK_THRESHOLD
        self._solid_fill_threshold: int = DEFAULT_SOLID_FILL_THRESHOLD

    @property
    def backend(self) -> str:
        """Nome do backend de I/O padrão ('opencv' ou 'vips')."""
        try:
            return get_default_backend_name()
        except (RuntimeError, KeyError):
            return self._backend

    @backend.setter
    def backend(self, value: str | AbstractImageIO) -> None:
        set_default_backend(value)
        if isinstance(value, str):
            val_lower = value.lower()
            self._backend = "vips" if val_lower == "pyvips" else val_lower
        else:
            name = value.__class__.__name__.lower().replace("backend", "")
            self._backend = "vips" if name == "pyvips" else name

    @property
    def memory_threshold(self) -> int | None:
        """Threshold global de pixels para alocação em RAM antes de usar paginação em disco.

        Retorna None se a paginação em disco estiver desativada (100% RAM).
        """
        return self._memory_threshold

    @memory_threshold.setter
    def memory_threshold(self, value: int | None) -> None:
        if value is not None and value <= 0:
            raise ValueError(
                f"memory_threshold deve ser positivo ou None, recebeu: {value}"
            )
        self._memory_threshold = value

    @property
    def dtype(self) -> np.dtype:
        """Tipo de dado padrão (dtype) para criação e decodificação de imagens."""
        return self._dtype

    @dtype.setter
    def dtype(self, value: Any) -> None:
        try:
            dt = np.dtype(value)
        except Exception as e:
            raise ValueError(f"dtype inválido: {value}") from e
        if dt not in SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype não suportado: '{value}'. Valores suportados: {SUPPORTED_DTYPES}"
            )
        self._dtype = dt

    @property
    def hard_mask_threshold(self) -> int:
        """Limiar de opacidade [0, 255] para corte binário em BlendMode.HARD_MASKING."""
        return self._hard_mask_threshold

    @hard_mask_threshold.setter
    def hard_mask_threshold(self, value: int) -> None:
        if not isinstance(value, (int, np.integer)) or not (0 <= int(value) <= 255):
            raise ValueError(
                f"hard_mask_threshold deve ser um inteiro entre 0 e 255, recebeu: {value}"
            )
        self._hard_mask_threshold = int(value)

    @property
    def solid_fill_threshold(self) -> int:
        """Limiar de opacidade [0, 255] mínimo do overlay para BlendMode.SOLID_FILL."""
        return self._solid_fill_threshold

    @solid_fill_threshold.setter
    def solid_fill_threshold(self, value: int) -> None:
        if not isinstance(value, (int, np.integer)) or not (0 <= int(value) <= 255):
            raise ValueError(
                f"solid_fill_threshold deve ser um inteiro entre 0 e 255, recebeu: {value}"
            )
        self._solid_fill_threshold = int(value)

    @property
    def min_disk_headroom(self) -> int:
        """Margem de segurança mínima em bytes no disco antes de rejeitar novas alocações."""
        return manager_global.min_disk_headroom

    @min_disk_headroom.setter
    def min_disk_headroom(self, value: int) -> None:
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(
                f"min_disk_headroom deve ser um inteiro não-negativo, recebeu: {value}"
            )
        manager_global.min_disk_headroom = int(value)

    def reset(self) -> None:
        """Restaura todas as configurações para seus valores padrão de fábrica."""
        self.backend = DEFAULT_BACKEND
        self.memory_threshold = DEFAULT_MEMORY_THRESHOLD
        self._dtype = DEFAULT_DTYPE
        self._hard_mask_threshold = DEFAULT_HARD_MASK_THRESHOLD
        self._solid_fill_threshold = DEFAULT_SOLID_FILL_THRESHOLD
        self.min_disk_headroom = DEFAULT_MIN_DISK_HEADROOM

    @contextmanager
    def __call__(self, **kwargs: Any) -> Generator[Config, None, None]:
        """Aplica configurações temporárias dentro de um bloco 'with'."""
        old_state: dict[str, Any] = {}
        for key in kwargs:
            if not hasattr(self, key) or key.startswith("_"):
                raise AttributeError(f"Opção de configuração inválida: '{key}'")
            old_state[key] = getattr(self, key)

        try:
            for key, value in kwargs.items():
                setattr(self, key, value)
            yield self
        finally:
            for key, old_value in old_state.items():
                setattr(self, key, old_value)


config = Config()
