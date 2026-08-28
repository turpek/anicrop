from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from anicrop.enums import ImageFormat
from anicrop.spatial import Region

if TYPE_CHECKING:
    from anicrop.interfaces.layout import LayoutStrategy


class AbstractCanvas(ABC):
    """Classe base abstrata para superfícies de composição Canvas."""

    bg_color: tuple[int, ...]

    @property
    @abstractmethod
    def format(self) -> ImageFormat:
        ...

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        ...

    @property
    @abstractmethod
    def width(self) -> int:
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        ...

    @property
    @abstractmethod
    def region(self) -> Region:
        ...

    @region.setter
    @abstractmethod
    def region(self, value: Region) -> None:
        ...

    @property
    @abstractmethod
    def layout(self) -> LayoutStrategy:
        ...
