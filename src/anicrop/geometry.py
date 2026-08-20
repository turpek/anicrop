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

    @property
    def layout_matrix(self) -> ndarray:
        """Matriz geométrica da moldura/janela ativa (sem offset físico)."""
        return self._layout.matrix

    @property
    def content_matrix(self) -> ndarray:
        """Matriz do conteúdo físico (a moldura + o deslocamento físico da foto)."""
        return self._layout.matrix @ mat_position(self._offset)

    @property
    def matrix(self) -> ndarray:
        """Alias para content_matrix."""
        return self.content_matrix

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
        self._cached_region: Region | None = None
        self._resolve_region = self._direct_region
        self._cached_global_region: Region | None = None
        self._resolve_global_region = self._direct_global_region

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

    @abstractmethod
    def _compute_region(self) -> Region:
        """Calcula a região local dinamicamente."""
        ...

    def _direct_region(self) -> Region:
        """Executa o cálculo de região local dinamicamente."""
        return self._compute_region()

    def _lazy_region(self) -> Region:
        """Retorna a região local em snapshot, calculando apenas na 1ª consulta."""
        if self._cached_region is None:
            self._cached_region = self._compute_region()
        return self._cached_region

    @property
    def region(self) -> Region:
        return self._resolve_region()

    @abstractmethod
    def _compute_global_region(self) -> Region:
        """Calcula a região global dinamicamente."""
        ...

    def _direct_global_region(self) -> Region:
        """Executa o cálculo de região global dinamicamente."""
        return self._compute_global_region()

    def _lazy_global_region(self) -> Region:
        """Retorna a região global em snapshot, calculando apenas na 1ª consulta."""
        if self._cached_global_region is None:
            self._cached_global_region = self._compute_global_region()
        return self._cached_global_region

    @property
    def global_region(self) -> Region:
        return self._resolve_global_region()


class LayerGeometry(GeometryStrategy):

    def __init__(self, base: Layer, region: Region):
        super().__init__()
        self._base = base
        self._region = region

    def _compute_matrix(self) -> ndarray:
        base = self._base
        return base.parent.matrix @ mat_position(self.region) @ base.transform.matrix

    def _compute_region(self) -> Region:
        return self._region

    def _compute_global_region(self) -> Region:
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

    def _compute_region(self) -> Region:
        base = self._base
        if len(base):
            return reduce(or_, [c.region for c in base])
        return self._region

    def _compute_global_region(self) -> Region:
        base = self._base
        if len(base):
            return reduce(or_, [c.global_region for c in base])
        return self._region


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
        base = self._base
        return base.parent.matrix @ mat_position(self.region) @ base.transform.matrix

    def _compute_region(self) -> Region:
        return self._region

    def _compute_global_region(self) -> Region:
        rect = calculate_new_rect(self.matrix, self.region.size)
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

    def _compute_region(self) -> Region:
        return self._region

    def _compute_global_region(self) -> Region:
        rect = calculate_region_rect(self.matrix, self._region)
        return Region.from_rect(*rect)
