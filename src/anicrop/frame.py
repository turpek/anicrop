from __future__ import annotations
from abc import ABC
from typing import Protocol, runtime_checkable, TYPE_CHECKING
import numpy as np

from anicrop.canvas import Canvas
from anicrop.spatial import Region, rect_to_region

from anicrop.transform import (
    calculate_new_rect,
    calculate_region_rect,
    mat_global,
    mat_inverse,
)


if TYPE_CHECKING:
    from anicrop.layer import EditLayer
    from anicrop.viewport import Viewport
    from anicrop.container import BaseLayer


@runtime_checkable
class SurfaceProtocol(Protocol):
    bg_color: tuple[int, int, int, int]

    @property
    def region(self) -> Region:
        ...

    @property
    def size(self) -> tuple[int, int]:
        ...


class BaseFrame(ABC):

    def __init__(
        self,
        bounds: Region,
        view_region: None | Region,
        matrix: np.ndarray = np.identity(3, dtype=np.float32),
        surface_size: tuple[int, int] | None = None,
    ):
        self._bounds = bounds
        self._matrix = matrix
        self._view_region = view_region
        self._dst_region = self._render_region(self.bounds, view_region)
        self._src_region = self._source_region(self.bounds, self.dst_region)
        self.surface_size = surface_size if surface_size is not None else bounds.size

    def _render_region(
        self, final_region: Region, view_region: Region | None,
    ) -> Region | None:

        if view_region is None:
            return None
        if view_region.overlaps(final_region):
            return view_region & final_region
        return None

    def _source_region(self, bounds: Region, dst_region: None | Region) -> None | Region:
        if dst_region and bounds.overlaps(dst_region):
            return bounds.overlap_with(dst_region)
        return None

    def screen_scale(self, edit_layer: EditLayer) -> float:
        m_edit_local = edit_layer.local_matrix
        m_total = self.matrix @ m_edit_local

        # SVD na submatriz 2x2 para extrair a escala real na tela
        submatrix_2x2 = m_total[:2, :2]
        _, s, _ = np.linalg.svd(submatrix_2x2)

        # Retorna a escala final exata combinada de tudo!
        return float(s[0])

    def _effective_view(self, surface_region: Region, view_region: Region | None) -> Region | None:
        if view_region is not None:
            if surface_region.overlaps(view_region):
                return view_region & surface_region
            return None
        return surface_region

    @property
    def bounds(self) -> Region:
        return self._bounds

    @property
    def dst_region(self) -> None | Region:
        return self._dst_region

    @property
    def targ_region(self) -> None | Region:
        if self._dst_region is not None and self._view_region is not None:
            return self._dst_region - self._view_region
        return self._dst_region

    @property
    def src_region(self) -> None | Region:
        return self._src_region

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class ViewportFrame(BaseFrame):
    def __init__(
        self,
        base: BaseLayer,
        viewport: Viewport,
        view_region: Region | None = None,
        local: bool = False,
    ):
        self.base = base
        self.viewport = viewport
        self.local = local

        effective_view = self._effective_view(viewport.region, view_region)
        m_view = viewport.roi_matrix @ viewport.fit_matrix(base.canvas_size)
        matrix = m_view if local else m_view @ mat_global(base)
        bounds = rect_to_region(calculate_new_rect(matrix, base.region.size))

        super().__init__(bounds, effective_view, matrix=matrix, surface_size=viewport.size)


class CanvasFrame(BaseFrame):
    def __init__(
        self,
        base: BaseLayer,
        canvas: Canvas,
        view_region: Region | None = None,
        local: bool = False,
    ):
        self.base = base
        self.local = local

        m_global = mat_global(base)
        effective_view = self._effective_view(canvas.region, view_region)

        if local:
            matrix = np.identity(3, dtype=np.float32)
            if effective_view is not None:
                inv_matrix = mat_inverse(m_global)
                rect = calculate_region_rect(inv_matrix, effective_view)
                view_target = rect_to_region(rect)
            else:
                view_target = None
            bounds = base.region
        else:
            matrix = m_global
            view_target = effective_view
            bounds = base.global_region

        super().__init__(bounds, view_target, matrix=matrix, surface_size=canvas.size)
