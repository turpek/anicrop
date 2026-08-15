from __future__ import annotations
from abc import ABC, abstractmethod
from functools import reduce
from numpy import ndarray
from operator import or_
from typing import TYPE_CHECKING

from anicrop.spatial import Region, Span
from anicrop.transform import calculate_new_rect, calculate_region_rect, mat_inverse, mat_position, mat_global

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
        self._offset = base.region - layout.region

    @property
    def base(self) -> GeometryStrategy:
        return self._base

    @property
    def layout(self) -> GeometryStrategy:
        return self._layout

    def sync(self, value: Region) -> None:
        self._layout._region = value
        self._base._region = value + self._offset

    def set_x(self, value: int | Span) -> None:
        self.sync(self._layout.region.replace(x=value))

    def set_y(self, value: int | Span) -> None:
        self.sync(self._layout.region.replace(y=value))

    def set_strategy(self, layout_strategy: GeometryStrategy) -> None:
        self._layout = layout_strategy
        self._offset = self._base.region - self._layout.region


class GeometryStrategy(ABC):
    _region: Region

    def __init__(self) -> None:
        self._cached_matrix: ndarray | None = None
        self._resolve_matrix = self._direct_matrix

    @abstractmethod
    def _compute_matrix(self) -> ndarray:
        """Calcula a matriz dinamicamente."""
        ...

    def _direct_matrix(self) -> ndarray:
        """Executa o cálculo dinâmico diretamente."""
        return self._compute_matrix()

    def _lazy_matrix(self) -> ndarray:
        """Retorna a matriz em snapshot, calculando apenas na 1ª consulta."""
        if self._cached_matrix is None:
            self._cached_matrix = self._compute_matrix()
        return self._cached_matrix

    @property
    def matrix(self) -> ndarray:
        return self._resolve_matrix()

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
        super().__init__()
        self._base = base
        self._region = region

    def _compute_matrix(self) -> ndarray:
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
        super().__init__()
        self._base = base
        self._region = region

    def _compute_matrix(self) -> np.ndarray:
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
        region: Region,
    ):
        super().__init__()
        self._base = base
        parent_mat = base.parent.matrix
        rect = calculate_region_rect(mat_inverse(parent_mat), region)
        self._region = Region.from_rect(*rect)

    def _compute_matrix(self) -> ndarray:
        return self._base.parent.matrix @ self._base.transform.matrix

    @property
    def region(self) -> Region:
        return self._region

    @property
    def global_region(self) -> Region:
        rect = calculate_region_rect(self.matrix, self._region)
        return Region.from_rect(*rect)


class FitGroupGeometry(GeometryStrategy):

    def __init__(
        self,
        base: GroupLayer,
        region: Region,
    ):
        super().__init__()
        self._base = base
        rect = calculate_region_rect(mat_inverse(base.matrix), region)
        self._region = Region.from_rect(*rect)

    def _compute_matrix(self) -> ndarray:
        return self._base.matrix

    @property
    def region(self) -> Region:
        return self._region

    @property
    def global_region(self) -> Region:
        rect = calculate_region_rect(self.matrix, self._region)
        return Region.from_rect(*rect)
