from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from anicrop.spatial import Region, Span
from anicrop.transform import (
    calculate_new_rect,
    calculate_region_rect,
    mat_inverse,
    mat_position,
)

if TYPE_CHECKING:
    from anicrop.container import BaseLayer, GroupLayer
    from anicrop.layer import Layer


class GeometryController:
    """Controls and synchronizes geometry strategies for a layer.

    Acts as an intermediary controller managing state transitions between
    the underlying base geometry and active frame strategies.
    """

    def __init__(self, base: GeometryStrategy, frame: GeometryStrategy):
        self._base: GeometryStrategy = base
        self._frame: GeometryStrategy = frame
        self._offset = base.region - frame.region

    @property
    def base(self) -> GeometryStrategy:
        return self._base

    @property
    def frame(self) -> GeometryStrategy:
        return self._frame

    @property
    def frame_matrix(self) -> ndarray:
        """Matriz geométrica da moldura/janela ativa (sem offset físico)."""
        return self._frame.matrix

    @property
    def content_matrix(self) -> ndarray:
        """Matriz do conteúdo físico (a moldura + o deslocamento físico da foto)."""
        return self._frame.matrix @ mat_position(self._offset)

    @property
    def matrix(self) -> ndarray:
        """Alias para content_matrix."""
        return self.content_matrix

    def _resolve_base_position(self, value: Region) -> tuple[int, int]:
        """Calcula a posição (x, y) da base compensando o offset pela rotação/escala da camada."""
        if self._offset.top_left == (0, 0):
            return value.x.start, value.y.start

        ox, oy = self._offset.top_left
        m = self._base.matrix

        rx = m[0, 0] * ox + m[0, 1] * oy
        ry = m[1, 0] * ox + m[1, 1] * oy

        x = int(round(value.x.start + rx))
        y = int(round(value.y.start + ry))

        return x, y

    def sync(self, value: Region) -> None:
        self._frame._region = value
        x, y = self._resolve_base_position(value)
        self._base._region = self._base._region.replace(x=x, y=y)

    def set_x(self, value: int | Span) -> None:
        self.sync(self._frame.region.replace(x=value))

    def set_y(self, value: int | Span) -> None:
        self.sync(self._frame.region.replace(y=value))

    def set_strategy(self, frame_strategy: GeometryStrategy) -> None:
        self._frame = frame_strategy
        self._offset = self._base.region - self._frame.region


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
        pass

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
        pass

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
        pass

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
        self._ref_region = region
        self._initial_matrix = np.copy(base.matrix)
        rect = calculate_region_rect(mat_inverse(base.matrix), region)
        self._region = Region.from_rect(*rect)

    def _compute_matrix(self) -> ndarray:
        return self._base.matrix

    def _compute_region(self) -> Region:
        return self._region

    def _compute_global_region(self) -> Region:
        if np.allclose(self._base.matrix, self._initial_matrix):
            return self._ref_region
        delta_m = self._base.matrix @ mat_inverse(self._initial_matrix)
        rect = calculate_region_rect(delta_m, self._ref_region)
        return Region.from_rect(*rect)
