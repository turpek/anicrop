from anicrop.blend import BLEND_MODE
from anicrop.enums import InterpolationOption
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.spatial import Region, bbox_to_region
from anicrop.transform import (
    calculate_new_bbox,
    calculate_region_bbox,
    mat_global,
    mat_inverse,
    mat_translation
)
from typing import Optional

import cv2
import numpy as np


def render_patch(edit_layer, matrix_global, dest_region, interp: InterpolationOption = InterpolationOption.LANCZOS):

    # 1. Projeta a BBox global de volta para a imagem original do Edit
    src_region = bbox_to_region(calculate_region_bbox(mat_inverse(matrix_global), dest_region))
    src_region = src_region.expand(all=interp.padding)
    limit_region = Region.from_size(*edit_layer.image.size)

    if limit_region.overlaps(src_region):
        region_mask = limit_region & src_region

        src_data = np.ascontiguousarray(edit_layer.image[region_mask][...])

        # Matriz que determina a posição local da região de recorte do edit
        M_src_offset = mat_translation(*region_mask.top_left)

        # Matriz que leva o recorte do edit para a origem
        dst_x, dst_y = dest_region.top_left
        M_dst_offset_inv = mat_translation(-dst_x, -dst_y)

        # Da direita para esquerda, pega a posição local do recorte,
        # transforma em global e leva para a origem
        M_cv2 = (M_dst_offset_inv @ matrix_global @ M_src_offset).astype(np.float64)

        return cv2.warpPerspective(src_data, M_cv2, dest_region.size, flags=interp.value)


class LayerRender:

    def __flatten_edits(
        self,
        layer: Layer,
        layer_image: Image,
        render_region: Region,
        interp: InterpolationOption,
    ) -> Image:

        m_layer_global = mat_global(layer)

        for edit_layer in layer._edits:

            # Calcula o bbox global do edit
            m_edit_global = m_layer_global @ edit_layer.local_matrix
            edit_global_bbox = bbox_to_region(
                calculate_new_bbox(m_edit_global, edit_layer.image.size)
            )

            if not edit_global_bbox.overlaps(render_region):
                continue

            dest_region = edit_global_bbox & render_region

            edit_data = render_patch(edit_layer, m_edit_global, dest_region, interp)
            if edit_data is None:
                continue

            edit_image = Image(edit_data, edit_layer.image.format)

            blend_region = dest_region - render_region.top_left
            blend = BLEND_MODE.get(edit_layer.blend_mode)

            blend(layer_image.view(blend_region), edit_image)

        return layer_image

    def __render_region(self, final_region, view_region) -> None:
        if not view_region:
            return final_region
        elif view_region.overlaps(final_region):
            return view_region & final_region

    def render_area(
        self,
        layer: Layer,
        view_region: Optional[Region] = None,
        interp: InterpolationOption = InterpolationOption.LANCZOS
    ) -> Image | None:

        # BBox global do layer
        final_region = layer.canvas_region
        render_region = self.__render_region(final_region, view_region)
        if render_region:
            size = render_region.size
            layer_image = Image.new(size, layer.format)
            return self.__flatten_edits(layer, layer_image, render_region, interp)

    def render(
        self,
        layer: Layer,
        interp: InterpolationOption = InterpolationOption.LANCZOS
    ) -> Image:
        return self.render_area(layer)
