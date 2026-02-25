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
    mat_inverse,
    mat_translation
)
from typing import Optional

import cv2
import numpy as np


def perf_render(edit_layer, matrix, local_region, edit_region):

    src_x, src_y, src_w, src_h = calculate_region_bbox(mat_inverse(matrix), edit_region)

    # 2. Clamping Manual Blindado (Impede o NumPy de ler pixels de trás pra frente)
    # Adicionamos a margem de 2 pixels do Lanczos4
    img_w, img_h = edit_layer.image.size

    start_x = max(0, int(src_x - 2))
    start_y = max(0, int(src_y - 2))
    end_x = min(img_w, int(src_x + src_w + 2))
    end_y = min(img_h, int(src_y + src_h + 2))

    # Se a região rotacionada/movida ficou 100% fora da imagem, aborta
    if start_x >= end_x or start_y >= end_y:
        return None
    region_mask = Region(Span(start_x, end_x - start_x), Span(start_y, end_y - start_y))

    # 3. Fatiamento Seguro
    src_data = np.ascontiguousarray(edit_layer.image[region_mask][...])

    # 4. As Únicas Duas Matrizes que Importam
    # M_src_offset: Avisa o OpenCV que o fatiamento começou deslocado da origem
    M_src_offset = mat_translation(*region_mask.top_left)

    # M_dst_offset_inv: Puxa o resultado usando o EDIT_REGION (que é relativo à matrix), não o local_region!
    M_dst_offset_inv = mat_translation(-edit_region.x.start, -edit_region.y.start)

    # A COMPOSIÇÃO PURA: A própria "matrix" já resolve a rotação, escala e posição do Edit!
    M_cv2 = (M_dst_offset_inv @ matrix @ M_src_offset).astype(np.float64)

    # 4. Processa apenas os pixels estritamente necessários
    edit_data = cv2.warpPerspective(
        src_data,
        M_cv2,
        local_region.size,
        flags=cv2.INTER_LANCZOS4
    )
    return edit_data


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
            print()
            print(matrix)
            print(matrix_local)

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
        matrix_final: np.ndarray,
        layer_image: Image,
        render_region: Region,
    ) -> np.ndarray:

        for edit_layer in layer._edits:

            matrix = mat_edit_final(edit_layer, matrix_final)
            x, y, w, h = calculate_new_bbox(matrix, edit_layer.image.size)

            matrix_local = mat_edit_local(edit_layer, matrix_final)
            x, y, w, h = calculate_new_bbox(matrix_local, edit_layer.image.size)
            local_region = Region(Span(x, w), Span(y, h))

            x, y, w, h = calculate_new_bbox(matrix_final, layer.region.size)
            layer_bbox = Region(Span(x, w), Span(y, h))
            edit_region = local_region.overlap_with(layer_bbox)
            local_region = layer_bbox.overlap_with(local_region)

            edit_data2 = perf_render(edit_layer, matrix, local_region, edit_region)
            edit_image2 = Image(edit_data2, edit_layer.image.format)
            blend = BLEND_MODE.get(edit_layer.blend_mode)
            blend(layer_image.view(local_region), edit_image2)

        return layer_image

    def render_perf(self, layer: Layer, view_region: Optional[Region] = None) -> Image | None:

        view_region = view_region if view_region else view_region

        final_region = layer.canvas_region

        if not view_region.overlaps(final_region):
            return None

        render_region = view_region & final_region
        size = render_region.size

        matrix = mat_final(layer, *final_region.top_left)

        layer_image = Image.new(size, layer.format)
        return self.__flatten_edits_perf(layer, matrix, layer_image, render_region)
