from __future__ import annotations
from abc import ABC, abstractmethod
from functools import reduce
from numpy import ndarray
from operator import or_
from typing import Optional, TYPE_CHECKING

from anicrop.spatial import Region, Span
from anicrop.transform import calculate_new_rect, mat_global, mat_inverse, mat_position

import numpy as np

if TYPE_CHECKING:
    from anicrop.container import BaseLayer, GroupLayer
    from anicrop.layer import Layer


class GeometryController:
    """Controls and synchronizes geometry strategies for a layer.

    Acts as an intermediary controller managing state transitions between
    the underlying base geometry and active layout strategies.
    """

    def __init__(self, base: GeometryStrategy, layout: GeometryStrategy):
        self._base: GeometryStrategy = base
        self._layout: GeometryStrategy = layout
        self._offset = layout.region - base.region

    @property
    def base(self) -> GeometryStrategy:
        return self._base

    @property
    def layout(self) -> GeometryStrategy:
        return self._layout

    def sync(self, value: Region):
        self._base._region = value
        self._layout._region = self._offset + value

    def set_x(self, value: int | Span) -> None:
        self.sync(self._base._region.replace(x=value))

    def set_y(self, value: int | Span) -> None:
        self.sync(self._base._region.replace(y=value))


class GeometryStrategy(ABC):

    @property
    @abstractmethod
    def matrix(self) -> Region:
        ...

    @property
    @abstractmethod
    def region(self) -> Region:
        ...

    @property
    @abstractmethod
    def global_region(self) -> Region:
        ...


class LayerGeometry(GeometryStrategy):

    def __init__(self, base: Layer, region: Region):
        self._base = base
        self._region = region

    @property
    def matrix(self) -> ndarray:
        base = self._base
        return base.parent.matrix @ mat_position(self.region) @ base.transform.matrix

    @property
    def region(self) -> Region:
        return self._region

    @property
    def global_region(self) -> Region:
        rect = calculate_new_rect(self.matrix, self.region.size)
        return Region.from_rect(*rect)


class GroupGeometry(GeometryStrategy):

    def __init__(self, base: GroupLayer, region: Region):
        self._base = base
        self._region = region

    @property
    def matrix(self) -> np.ndarray:
        base = self._base
        return base.parent.matrix @ base._parent_inverse @ base.transform.matrix

    def _calculate_region(self, attr_name: str) -> Region:
        base = self._base
        if len(base):
            return reduce(or_, [getattr(c, attr_name) for c in base])
        return self._region

    @property
    def region(self) -> Region:
        return self._calculate_region('region')

    @property
    def global_region(self) -> Region:
        return self._calculate_region('global_region')


class FitGeometry(GeometryStrategy):

    def __init__(
        self,
        base: BaseLayer,
        region: Optional[Region] = None,
    ):
        self._base = base
        self._region = region
        self._local_matrix = mat_inverse(base.base.matrix)

    @property
    def matrix(self) -> ndarray:
        return mat_global(self._base) @ self.mask.local_matrix

    @property
    def region(self) -> Region:
        return self.mask.region

    @property
    def global_region(self) -> Region:
        rect = calculate_new_rect(self.matrix, self.mask.region.size)
        return Region.from_rect(*rect)
