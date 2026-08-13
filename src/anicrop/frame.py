from __future__ import annotations
from abc import ABC
from typing import Optional, TYPE_CHECKING
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
    from anicrop.layer import Layer, EditLayer
    from anicrop.viewport import Viewport


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
        self._dst_region = self._render_region(self.bounds, view_region)
        self._src_region = self._view_region(self.bounds, self.dst_region)
        self.surface_size = surface_size if surface_size is not None else bounds.size

    def _render_region(
        self, final_region: Region, view_region: Region | None,
    ) -> Region | None:

        if view_region is None:
            return final_region
        if view_region.overlaps(final_region):
            return view_region & final_region
        return None

    def _view_region(self, bounds: Region, dst_region: None | Region) -> None | Region:
        if dst_region and bounds.overlaps(dst_region):
            return bounds.overlap_with(dst_region)

    def screen_scale(self, edit_layer: EditLayer) -> float:
        m_edit_local = edit_layer.local_matrix
        m_total = self.matrix @ m_edit_local

        # SVD na submatriz 2x2 para extrair a escala real na tela
        submatrix_2x2 = m_total[:2, :2]
        _, s, _ = np.linalg.svd(submatrix_2x2)

        # Retorna a escala final exata combinada de tudo!
        return float(s[0])

    @property
    def bounds(self) -> Region:
        return self._bounds

    @property
    def dst_region(self) -> None | Region:
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
        layer: Layer,
        viewport: Viewport,
        local: bool = False,
    ):
        self.layer = layer
        self.viewport = viewport
        self.local = local

        m_view = viewport.roi_matrix @ viewport.fit_matrix(layer.canvas_size)
        matrix = m_view if local else m_view @ mat_global(layer)
        bounds = rect_to_region(calculate_new_rect(matrix, layer.region.size))

        super().__init__(bounds, viewport.region, matrix=matrix, surface_size=viewport.size)


class CanvasFrame(BaseFrame):
    def __init__(
        self,
        layer: Layer,
        view_region: Optional[Region] = None,
        local: bool = False,
    ):
        self.layer = layer
        self.local = local

        view_region = view_region.region if isinstance(view_region, Canvas) else view_region
        m_global = mat_global(layer)

        if local:
            matrix = np.identity(3, dtype=np.float32)
            if view_region is not None:
                inv_matrix = mat_inverse(m_global)
                rect = calculate_region_rect(inv_matrix, view_region)
                view_target = rect_to_region(rect)
            else:
                view_target = None
        else:
            matrix = m_global
            view_target = view_region

        bounds = layer.global_region
        super().__init__(bounds, view_target, matrix=matrix, surface_size=bounds.size)
