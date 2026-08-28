from anicrop.enums import ImageFormat
from anicrop.spatial import Region, rect_to_region
from anicrop.type import Scale
from anicrop.transform import calculate_new_rect, mat_inverse, mat_pivot, mat_translation
from numpy import ndarray


class Viewport:
    bg_color: tuple[int, ...]

    def __init__(
        self,
        size: tuple[int, int],
        fit_scale: float = 1.0,
        bg_color: tuple[int, ...] | None = None,
        format: ImageFormat = ImageFormat.RGBA,
    ):
        self._region = Region.from_size(*size)
        self._scale = Scale(1, 1)
        self._fit = Scale(fit_scale, fit_scale)
        self._format = format
        if bg_color is None:
            if format == ImageFormat.RGBA:
                self.bg_color = (204, 204, 204, 255)
            elif format == ImageFormat.RGB:
                self.bg_color = (204, 204, 204)
            elif format == ImageFormat.GRAY:
                self.bg_color = (204,)
            elif format == ImageFormat.GRAY_ALPHA:
                self.bg_color = (204, 255)
            else:
                self.bg_color = (204,) * format.channels
        else:
            self.bg_color = bg_color

    def __repr__(self) -> str:
        return f'Viewport(region={self.region}, scale={self.scale}, format={self.format})'

    @property
    def format(self) -> ImageFormat:
        return self._format

    @property
    def size(self) -> tuple[int, int]:
        return self._region.size

    @property
    def top_left(self) -> tuple[int, int]:
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

    def fit_matrix(self, layer_size: tuple[int, int]) -> ndarray:
        # 2. Qual o tamanho do papel DEPOIS de encolher?
        s = self._fit.sx
        scaled_w = layer_size[0] * s
        scaled_h = layer_size[1] * s

        # 3. A CENTRALIZAÇÃO: Calcula o espaço que sobrou e divide por 2
        view_w, view_h = self.size
        offset_x = (view_w - scaled_w) / 2
        offset_y = (view_h - scaled_h) / 2

        # 4. A Matriz: Encolhe primeiro, depois empurra pro centro
        return mat_translation(offset_x, offset_y) @ self._fit.matrix
        # return self._fit.matrix

    def roi(self, region: Region) -> Region:
        mat_tr = mat_inverse(mat_translation(*region.top_left))
        m_roi = mat_tr @ mat_inverse(self.roi_matrix)
        return rect_to_region(calculate_new_rect(m_roi, self.size))
