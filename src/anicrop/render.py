from anicrop.blend import blend_normal
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.spatial import Region, Span
from anicrop.transform import (
    calculate_new_bbox,
    mat_edit_final,
    mat_edit_local,
    mat_final
)

import cv2
import numpy as np


class LayerRender:

    def __flatten_edits(
        self,
        layer: Layer,
        matrix_final: np.ndarray,
        layer_image: Image,
    ) -> np.ndarray:

        for edit_layer in layer._edits:

            matrix = mat_edit_final(edit_layer, matrix_final)
            x, y, w, h = calculate_new_bbox(matrix, edit_layer.image.size)

            matrix_local = mat_edit_local(edit_layer, matrix_final)
            x, y, w, h = calculate_new_bbox(matrix_local, edit_layer.image.size)
            local_region = Region(Span(x, w), Span(y, h))
            size = local_region.size

            x, y, w, h = calculate_new_bbox(matrix_final, layer.region.size)
            layer_bbox = Region(Span(x, w), Span(y, h))
            edit_region = local_region.overlap_with(layer_bbox)
            local_region = layer_bbox.overlap_with(local_region)

            edit_data = cv2.warpPerspective(
                edit_layer.image[...],
                matrix,
                size,
                flags=cv2.INTER_LANCZOS4
            )
            edit_image = Image(edit_data, edit_layer.image.format)
            blend_normal(layer_image[local_region], edit_image[edit_region])

        return layer_image

    def render(self, layer: Layer) -> Image:

        region_final = layer.canvas_region
        matrix = mat_final(layer, *region_final.top_left)
        size = region_final.size

        layer_image = Image.new(size, layer.format)
        return self.__flatten_edits(layer, matrix, layer_image)
