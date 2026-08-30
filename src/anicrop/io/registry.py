from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anicrop.interfaces.io import AbstractImageIO

_BACKENDS: dict[str, AbstractImageIO] = {}
_DEFAULT_BACKEND_NAME: str | None = None


def register_backend(name: str, backend: AbstractImageIO) -> None:
    """Registra um backend de I/O de imagens no sistema."""
    key = name.lower()
    _BACKENDS[key] = backend
    global _DEFAULT_BACKEND_NAME
    if _DEFAULT_BACKEND_NAME is None:
        _DEFAULT_BACKEND_NAME = key


def set_default_backend(backend: str | AbstractImageIO) -> None:
    """Define o backend padrão para leitura e gravação de imagens."""
    global _DEFAULT_BACKEND_NAME
    if isinstance(backend, str):
        key = backend.lower()
        if key not in _BACKENDS:
            raise KeyError(
                f"Backend '{backend}' não está registrado. Disponíveis: {list(_BACKENDS.keys())}"
            )
        _DEFAULT_BACKEND_NAME = key
    else:
        # Se foi passado um objeto direto
        name = backend.__class__.__name__.lower().replace("backend", "")
        _BACKENDS[name] = backend
        _DEFAULT_BACKEND_NAME = name


def get_backend(
    name_or_instance: str | AbstractImageIO | None = None,
) -> AbstractImageIO:
    """Retorna a instância do backend solicitado ou o backend padrão."""
    if name_or_instance is None:
        if _DEFAULT_BACKEND_NAME is None or _DEFAULT_BACKEND_NAME not in _BACKENDS:
            raise RuntimeError("Nenhum backend de I/O registrado no sistema.")
        return _BACKENDS[_DEFAULT_BACKEND_NAME]

    if isinstance(name_or_instance, str):
        key = name_or_instance.lower()
        if key not in _BACKENDS:
            raise KeyError(
                f"Backend '{name_or_instance}' não encontrado. Disponíveis: {list(_BACKENDS.keys())}"
            )
        return _BACKENDS[key]

    return name_or_instance


def get_default_backend() -> AbstractImageIO:
    """Retorna o backend padrão ativo."""
    return get_backend(None)
