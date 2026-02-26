from anicrop.blend import BLEND_MODE
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.spatial import Region, Span
from anicrop.transform import (
    calculate_new_bbox,
    calculate_region_bbox,
    mat_edit_final,
    mat_edit_local,
    mat_final,
    mat_global,
    mat_inverse,
    mat_translation
)
from typing import Optional

import cv2
import numpy as np


def perf_render(edit_layer, matrix_global, dest_region):
    # 1. Projeta a BBox global de volta para a imagem original do Edit
    src_x, src_y, src_w, src_h = calculate_region_bbox(mat_inverse(matrix_global), dest_region)

    # 2. Clamping Blindado
    img_w, img_h = edit_layer.image.size
    start_x = max(0, int(src_x - 2))
    start_y = max(0, int(src_y - 2))
    end_x = min(img_w, int(src_x + src_w + 2))
    end_y = min(img_h, int(src_y + src_h + 2))

    if start_x >= end_x or start_y >= end_y:
        return None

    region_mask = Region(Span(start_x, end_x - start_x), Span(start_y, end_y - start_y))
    src_data = np.ascontiguousarray(edit_layer.image[region_mask][...])

    # 3. As Matrizes de Compensação
    M_src_offset = mat_translation(*region_mask.top_left)

    # Como dest_region já é uma coordenada global, o offset inverso é direto e óbvio:
    dst_x, dst_y = dest_region.top_left
    M_dst_offset_inv = mat_translation(-dst_x, -dst_y)

    # 4. A Composição Pura (Destino Inverso -> Global -> Origem)
    M_cv2 = (M_dst_offset_inv @ matrix_global @ M_src_offset).astype(np.float64)

    # 5. Render
    return cv2.warpPerspective(
        src_data,
        M_cv2,
        dest_region.size,
        # flags=cv2.INTER_LANCZOS4
        flags=cv2.INTER_NEAREST
    )


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
            blend = BLEND_MODE.get(edit_layer.blend_mode)
            blend(layer_image.view(local_region), edit_image.view(edit_region))

        return layer_image

    def render(self, layer: Layer) -> Image:

        region_final = layer.canvas_region
        matrix = mat_final(layer, *region_final.top_left)
        size = region_final.size

        layer_image = Image.new(size, layer.format)
        return self.__flatten_edits(layer, matrix, layer_image)

    def __flatten_edits_perf(
        self,
        layer: Layer,
        layer_image: Image,
        render_region: Region,
    ) -> Image:

        # 1. A matriz base da camada (Onde o papel está na mesa)
        m_layer_global = mat_global(layer)

        for edit_layer in layer._edits:
            # 2. Matriz Global do Edit (Onde o adesivo está na mesa)
            m_edit_global = m_layer_global @ edit_layer.local_matrix

            # 3. Descobre a BBox Global desse edit (Sem compensações)
            ex, ey, ew, eh = calculate_new_bbox(m_edit_global, edit_layer.image.size)
            edit_global_bbox = Region(Span(ex, ew), Span(ey, eh))
            if not edit_global_bbox.overlaps(render_region):
                continue

            dest_region = edit_global_bbox & render_region

            edit_data = perf_render(edit_layer, m_edit_global, dest_region)
            if edit_data is None:
                continue

            edit_image = Image(edit_data, edit_layer.image.format)

            blend_region = dest_region - render_region.top_left

            blend = BLEND_MODE.get(edit_layer.blend_mode)
            blend(layer_image.view(blend_region), edit_image)

        return layer_image

    def render_perf(self, layer: Layer, view_region: Optional[Region] = None) -> Image | None:

        final_region = layer.canvas_region
        view_region = view_region if view_region else final_region

        if view_region and not view_region.overlaps(final_region):
            return None

        render_region = view_region & final_region
        size = render_region.size

        layer_image = Image.new(size, layer.format)
        return self.__flatten_edits_perf(layer, layer_image, render_region)
