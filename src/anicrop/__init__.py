from importlib.metadata import PackageNotFoundError, version

from .config import config
from .container import GroupLayer
from .document import Document
from .enums import BlendMode, ImageFormat
from .frame import BaseFrame, CanvasFrame, ViewportFrame
from .image import Image, get_memory_threshold, set_memory_threshold
from .layer import Layer
from .render import transform_image
from .scratch import ScratchBuffer
from .spatial import Region, Span
from .viewer import Viewer
from .viewport import Viewport

try:
    __version__ = version("anicrop")
except PackageNotFoundError:
    __version__ = "0.6.3"

__all__ = [
    "__version__",
    "transform_image",
    "config",
    "Document",
    "Viewport",
    "BlendMode",
    "ImageFormat",
    "Region",
    "Span",
    "Layer",
    "GroupLayer",
    "ScratchBuffer",
    "Image",
    "Viewer",
    "BaseFrame",
    "CanvasFrame",
    "ViewportFrame",
    "set_memory_threshold",
    "get_memory_threshold",
]
