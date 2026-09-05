from numpy import ndarray

from anicrop.canvas import Canvas
from anicrop.interfaces.canvas import AbstractCanvas
from anicrop.layout import ViewportLayoutStrategy
from anicrop.spatial import Point, Region, rect_to_region
from anicrop.transform import calculate_new_rect, mat_inverse, mat_pivot, mat_translation
from anicrop.type import Scale


class Viewport:
    bg_color: tuple[int, ...]
    _canvas: AbstractCanvas
    _layout: ViewportLayoutStrategy

    def __init__(
        self,
        size: tuple[float, float],
        fit_scale: float = 1.0,
        canvas: AbstractCanvas | None = None,
        bg_color: tuple[int, ...] | None = None,
    ):
        self._region = Region.from_size(size[0], size[1])
        self._scale = Scale(1, 1)
        self._fit = Scale(fit_scale, fit_scale)
        if canvas is not None:
            self.set_canvas(canvas)
        else:
            self._canvas = Canvas.from_size(size[0], size[1])
        self.bg_color = bg_color if bg_color is not None else (204, 204, 204)
        self._layout = ViewportLayoutStrategy(self)

    def __repr__(self) -> str:
        return f"Viewport(region={self.region}, scale={self.scale})"

    def set_canvas(self, canvas: AbstractCanvas) -> None:
        """Define ou troca o Canvas observado pela Viewport."""
        if not isinstance(canvas, AbstractCanvas):
            raise TypeError(f"Expected AbstractCanvas, got {type(canvas).__name__}")
        self._canvas = canvas

    @property
    def layout(self) -> ViewportLayoutStrategy:
        """Estratégia de layout para a moldura e câmera da Viewport."""
        return self._layout

    @property
    def canvas_size(self) -> Point:
        """Retorna a dimensão do Canvas atual em tempo real."""
        return self._canvas.size

    @property
    def size(self) -> Point:
        return self._region.size

    @property
    def top_left(self) -> Point:
        return self._region.top_left

    @property
    def scale_factor(self) -> float:
        return self.scale.sx * self._fit.sx

    @property
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, value) -> None:
        self._region = value

    @property
    def scale(self) -> Scale:
        return self._scale

    @scale.setter
    def scale(self, value) -> None:
        self._scale = value

    @property
    def roi_matrix(self) -> ndarray:
        x, y = self._region.top_left
        return mat_pivot(self.scale, self.size) @ mat_translation(-x, -y)

    def fit_matrix(self) -> ndarray:
        s = self._fit.sx
        scaled_w = self._canvas.size[0] * s
        scaled_h = self._canvas.size[1] * s

        view_w, view_h = self.size
        offset_x = (view_w - scaled_w) / 2
        offset_y = (view_h - scaled_h) / 2

        return mat_translation(offset_x, offset_y) @ self._fit.matrix

    def roi(self, region: Region) -> Region:
        mat_tr = mat_inverse(mat_translation(*region.top_left))
        m_roi = mat_tr @ mat_inverse(self.roi_matrix)
        return rect_to_region(calculate_new_rect(m_roi, self.size))
