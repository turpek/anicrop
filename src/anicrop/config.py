from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anicrop.interfaces.io import AbstractImageIO

from anicrop.io.registry import get_default_backend_name, set_default_backend

DEFAULT_MEMORY_THRESHOLD: int = 8192 * 8192  # 64 MP (8K x 8K)
DEFAULT_BACKEND: str = "opencv"


class Config:
    """Configuração global centralizada do anicrop."""

    def __init__(self) -> None:
        self._backend: str = DEFAULT_BACKEND
        self._memory_threshold: int | None = DEFAULT_MEMORY_THRESHOLD

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

    def reset(self) -> None:
        """Restaura todas as configurações para seus valores padrão de fábrica."""
        self.backend = DEFAULT_BACKEND
        self.memory_threshold = DEFAULT_MEMORY_THRESHOLD

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
