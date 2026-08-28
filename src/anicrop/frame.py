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
    from anicrop.mask import Mask


@runtime_checkable
class SurfaceProtocol(Protocol):
    bg_color: tuple[int, ...]

    @property
    def region(self) -> Region:
        ...

    @property
    def size(self) -> tuple[int, int]:
        ...


def calculate_mask_rect(mask: Mask | None, matrix: np.ndarray) -> Region | None:
    """Calcula a Bounding Box projetada (rect) da máscara combinando sua matriz local com a matriz global fornecida."""
    if mask is not None:
        return mask.projected_region(matrix)
    return None


class BaseFrame(ABC):

    def __init__(
        self,
        base: BaseLayer,
        bounds: Region,
        view_region: None | Region,
        effective_view: None | Region,
        matrix: np.ndarray,
        surface: SurfaceProtocol,
    ):
        self.base = base
        self._bounds = bounds
        self._matrix = matrix
        self._view_region = view_region
        self.surface = surface
        self._dst_region = self._render_region(self.bounds, effective_view)
        self._src_region = self._source_region(self.bounds, self.dst_region)

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

    def _effective_view(
        self,
        surface_region: Region,
        view_region: Region | None,
        mask: Mask | None,
        matrix: np.ndarray,
    ) -> Region | None:
        """Calcula a janela efetiva de visualização respeitando a prioridade de view_region, máscara e superfície."""
        if view_region is not None:
            if surface_region.overlaps(view_region):
                return view_region & surface_region
            return None

        if mask is not None and mask.visible:
            projected_box = mask.projected_region(matrix)
            if surface_region.overlaps(projected_box):
                return projected_box & surface_region
            return None

        return surface_region

    def _expand_bounds(self, bounds: Region, base: BaseLayer) -> Region:
        """Expande os limites geométricos da camada de acordo com o padding dos efeitos ativos."""
        pad_t, pad_r, pad_b, pad_l = base.get_effects_padding()
        return bounds.expand(
            top=max(0, pad_t),
            right=max(0, pad_r),
            bottom=max(0, pad_b),
            left=max(0, pad_l),
        )

    @property
    def bounds(self) -> Region:
        return self._bounds

    @property
    def dst_region(self) -> None | Region:
        return self._dst_region

    @property
    def surface_size(self) -> tuple[int, int]:
        return self.surface.size

    @property
    def surface_region(self) -> Region:
        return self.surface.region

    @property
    def targ_region(self) -> None | Region:
        if self._dst_region is None:
            return None

        buffer_region = self._view_region if self._view_region is not None else self.surface.region

        if buffer_region.overlaps(self._dst_region):
            return buffer_region.overlap_with(self._dst_region)

        return None

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
        self.viewport = viewport
        self.local = local

        m_view = viewport.roi_matrix @ viewport.fit_matrix(base.region.size)
        matrix = m_view if local else m_view @ mat_global(base)

        effective_view = self._effective_view(
            viewport.region, view_region, base.mask, matrix
        )
        bounds = rect_to_region(calculate_new_rect(matrix, base.region.size))
        bounds = self._expand_bounds(bounds, base)

        super().__init__(base, bounds, view_region, effective_view, matrix=matrix, surface=viewport)


class CanvasFrame(BaseFrame):
    def __init__(
        self,
        base: BaseLayer,
        canvas: Canvas,
        view_region: Region | None = None,
        local: bool = False,
    ):
        self.local = local

        m_global = mat_global(base)
        if local:
            matrix = np.identity(3, dtype=np.float32)
            effective_view = self._effective_view(
                canvas.region, view_region, base.mask, matrix
            )
            if effective_view is not None:
                inv_matrix = mat_inverse(m_global)
                rect = calculate_region_rect(inv_matrix, effective_view)
                view_target = rect_to_region(rect)
            else:
                view_target = None
            bounds = base.region
        else:
            matrix = m_global
            view_target = self._effective_view(
                canvas.region, view_region, base.mask, matrix
            )
            bounds = base.global_region

        bounds = self._expand_bounds(bounds, base)

        super().__init__(base, bounds, view_region, view_target, matrix=matrix, surface=canvas)
