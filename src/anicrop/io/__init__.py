from anicrop.interfaces.io import AbstractImageIO, SaveOptions
from anicrop.io.opencv import OpenCVBackend
from anicrop.io.registry import (
    get_backend,
    get_default_backend,
    register_backend,
    set_default_backend,
)

# Registra o backend OpenCV inicial
register_backend("opencv", OpenCVBackend())

__all__ = [
    "AbstractImageIO",
    "OpenCVBackend",
    "SaveOptions",
    "get_backend",
    "get_default_backend",
    "register_backend",
    "set_default_backend",
]
