from anicrop.spatial import Region, bbox_to_region
from anicrop.type import Scale
from anicrop.transform import calculate_new_bbox, mat_inverse, mat_pivot, mat_translation
from numpy import ndarray


class Viewport:
    def __init__(self, size, fit_scale):
        self._region = Region.from_size(*size)
        self._scale = Scale(1, 1)
        self._fit = Scale(fit_scale, fit_scale)

    def __repr__(self) -> None:
        return f'Viewport(region={self.region}, scale={self.scale})'

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

    @property
    def fit_matrix(self) -> ndarray:
        return self._fit.matrix

    def roi(self, region: Region) -> Region:
        mat_tr = mat_inverse(mat_translation(*region.top_left))
        m_roi = mat_tr @ mat_inverse(self.roi_matrix)
        return bbox_to_region(calculate_new_bbox(m_roi, self.size))
