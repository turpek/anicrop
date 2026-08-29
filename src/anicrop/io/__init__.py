from anicrop.interfaces.io import AbstractImageIO, SaveOptions
from anicrop.io.opencv import OpenCVBackend
from anicrop.io.registry import (
    get_backend,
    get_default_backend,
    register_backend,
    set_default_backend,
)
from anicrop.io.vips import PyvipsBackend, is_vips_available

# Registra os backends disponíveis
register_backend("opencv", OpenCVBackend())

if is_vips_available():
    vips_backend = PyvipsBackend()
    register_backend("vips", vips_backend)
    set_default_backend("vips")

__all__ = [
    "AbstractImageIO",
    "OpenCVBackend",
    "PyvipsBackend",
    "SaveOptions",
    "get_backend",
    "get_default_backend",
    "is_vips_available",
    "register_backend",
    "set_default_backend",
]
