from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from anicrop.spatial import Point, Region

if TYPE_CHECKING:
    from anicrop.interfaces.layout import LayoutStrategy


class AbstractCanvas(ABC):
    """Classe base abstrata para superfícies de composição Canvas."""

    bg_color: tuple[int, ...]

    @property
    @abstractmethod
    def size(self) -> Point:
        pass

    @property
    @abstractmethod
    def width(self) -> float:
        pass

    @property
    @abstractmethod
    def height(self) -> float:
        pass

    @property
    @abstractmethod
    def region(self) -> Region:
        pass

    @region.setter
    @abstractmethod
    def region(self, value: Region) -> None:
        pass

    @property
    @abstractmethod
    def layout(self) -> LayoutStrategy:
        pass
