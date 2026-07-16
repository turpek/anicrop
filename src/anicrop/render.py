from anicrop.blend import BLEND_MODE
from anicrop.enums import InterpolationOption, RenderFlags, WarpMode
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer, EditLayer
from anicrop.spatial import Region, bbox_to_region
from anicrop.transform import (
    calculate_new_bbox,
    calculate_region_bbox,
    mat_global,
    mat_inverse,
    mat_scale,
    mat_translation
)
from anicrop.viewport import Viewport
from operator import mul
from typing import Optional, Iterable

import cv2
import numpy as np
import weakref


def warp_affine(
    src_data: np.ndarray,
    m_cv2: np.ndarray,
    dest_size: tuple[int, int],
    interp: InterpolationOption = InterpolationOption.LINEAR
):
    M_affine = m_cv2[:2, :].astype(np.float64)

    return cv2.warpAffine(
        src_data,
        M_affine,
        dest_size,
        flags=interp.value,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )


def warp_perspective(
    src_data: np.ndarray,
    m_cv2: np.ndarray,
    dest_size: tuple[int, int],
    interp: InterpolationOption = InterpolationOption.LINEAR
):
    return cv2.warpPerspective(
        src_data,
        m_cv2,
        dest_size,
        flags=interp.value,
    )


WARP_MODE = {
    WarpMode.AFFINE: warp_affine,
    WarpMode.PERSPECTIVE: warp_perspective,
}


def render_patch(
    src_image: Image,
    matrix_global: np.ndarray,
    dest_region: Region,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
):

    # 1. Projeta a BBox global de volta para a imagem original do Edit
    src_region = bbox_to_region(calculate_region_bbox(mat_inverse(matrix_global), dest_region))
    src_region = src_region.expand(all=interp.padding)
    limit_region = Region.from_size(*src_image.size)

    if limit_region.overlaps(src_region):
        region_mask = limit_region & src_region

        src_data = src_image[region_mask]

        # Matriz que determina a posição local da região de recorte do edit
        M_src_offset = mat_translation(*region_mask.top_left)

        # Matriz que leva o recorte do edit para a origem
        dst_x, dst_y = dest_region.top_left
        M_dst_offset_inv = mat_translation(-dst_x, -dst_y)

        # Da direita para esquerda, pega a posição local do recorte,
        # transforma em global e leva para a origem
        M_cv2 = (M_dst_offset_inv @ matrix_global @ M_src_offset).astype(np.float64)

        warp = WARP_MODE.get(warp_mode, warp_affine)
        return warp(src_data, M_cv2, dest_region.size, interp)


def generate_opacity_mask(
    image: Image,
    render_region: Region,
    viewport_size: tuple[int, int],
    target_size=(32, 32)
) -> np.ndarray:
    """Função usada para gerar miniaturas do layer mapeadas proporcionalmente na tela"""

    eroded_alpha = np.zeros(target_size, dtype=np.uint8)

    scale_x = target_size[0] / viewport_size[0]
    scale_y = target_size[1] / viewport_size[1]

    tw_img = max(1, int(image.width * scale_x))
    th_img = max(1, int(image.height * scale_y))

    if image.has_alpha:
        w, h = image.size
        alpha_origin = image[..., -1:]

        kernel_h = max(1, h // th_img)
        kernel_w = max(1, w // tw_img)
        kernel = np.ones((kernel_h, kernel_w), dtype=np.uint8)

        # A erosão propaga os pixels de menor opacidade (mais escuros)
        eroded = cv2.erode(alpha_origin, kernel)
        mini_mask = cv2.resize(eroded, (tw_img, th_img), interpolation=cv2.INTER_NEAREST)
    else:
        mini_mask = np.full((th_img, tw_img), 255, dtype=np.uint8)

    # Descobre as coordenadas proporcionais na grade target_size
    start_x = int(render_region.top_left[0] * scale_x)
    start_y = int(render_region.top_left[1] * scale_y)

    # Limites seguros na matriz target_size (caso o layer saia da tela)
    sy = max(0, start_y)
    ey = min(target_size[1], start_y + th_img)
    sx = max(0, start_x)
    ex = min(target_size[0], start_x + tw_img)

    # Pedaço da mini_mask que será efetivamente copiado
    my1 = sy - start_y
    my2 = my1 + (ey - sy)
    mx1 = sx - start_x
    mx2 = mx1 + (ex - sx)

    if ey > sy and ex > sx:
        eroded_alpha[sy:ey, sx:ex] = mini_mask[my1:my2, mx1:mx2]

    return eroded_alpha


class LODManager:
    def __init__(self):

        self._l1_cache = weakref.WeakKeyDictionary()
        self._l2_cache = weakref.WeakKeyDictionary()

    def get_source(
            self,
            viewport: Viewport,
            edit_layer: EditLayer,
            layer_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna (pixels, m_adjust) baseada na escala.
        Decide se gera o cache ou usa o original.
        """
        viewport_scale = viewport.scale_factor

        # Só fazemos cache se o layer for MAIOR que a viewport
        if mul(*layer_size) > mul(*viewport.size):
            # Heurística: Zoom muito baixo? Tenta L2
            if viewport_scale <= 0.25:
                return self._resolve_level(edit_layer, self._l2_cache, factor=10.0)

            # Heurística: Zoom baixo? Tenta L1
            if viewport_scale <= 0.5:
                return self._resolve_level(edit_layer, self._l1_cache, factor=5.0)

        # Fallback: Original
        return edit_layer.image[...], np.identity(3, dtype=np.float32)

    def _resolve_level(
        self,
        edit_layer: EditLayer,
        cache_dict: weakref.WeakKeyDictionary,
        factor: float
    ) -> tuple[np.ndarray, np.ndarray]:

        # Se não estiver no cache, regenera. A imutabilidade do EditLayer
        # garante que a identidade do objeto é suficiente para o cache.
        if edit_layer not in cache_dict:
            img = edit_layer.image[...]
            new_size = (int(img.shape[1] // factor), int(img.shape[0] // factor))
            cache_dict[edit_layer] = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        cached_data = cache_dict[edit_layer]
        m_adjust = mat_scale(factor, factor)
        return cached_data, m_adjust


class LayerRender:

    def __init__(self):
        self._cache = weakref.WeakKeyDictionary()

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

            edit_data = render_patch(
                edit_layer.image, m_edit_global, dest_region, layer._warp_mode, interp,
            )
            if edit_data is None:
                continue

            edit_image = Image(edit_data, edit_layer.image.format)

            blend_region = dest_region - render_region.top_left
            blend = BLEND_MODE.get(edit_layer.blend_mode)

            blend(layer_image.view(blend_region), edit_image)

        return layer_image

    def __render_region(self, final_region, view_region) -> None | Region:
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

        flags = layer._resolve_render()
        # BBox global do layer
        final_region = layer.global_region
        render_region = self.__render_region(final_region, view_region)

        if render_region:

            if flags & RenderFlags.PIXELS or layer._id not in self._cache:
                size = render_region.size
                layer_image = Image.new(size, layer.format)
                return self.__flatten_edits(layer, layer_image, render_region, interp)

            elif layer._id in self._cache:
                view = final_region.overlap_with(render_region)
                return self._cache[layer._id].crop(view)

    def render(
        self,
        layer: Layer,
        interp: InterpolationOption = InterpolationOption.LANCZOS
    ) -> Image:

        flags = layer._resolve_render()
        if flags & RenderFlags.PIXELS:
            self._cache[layer._id] = self.render_area(layer)
        layer._commit_render_state()
        return self._cache[layer._id].crop(...)


class ViewportRender:
    def __init__(self, lod_manager: Optional[LODManager] = None):
        self._cache = weakref.WeakKeyDictionary()
        self.lod_manager = lod_manager or LODManager()
        self._target_size = (32, 32)

    def __render_region(self, final_region: Region, view_region: Region) -> None | Region:
        if view_region.overlaps(final_region):
            return view_region & final_region

    def __flatten_edits(
        self,
        layer: Layer,
        viewport: Viewport,
        layer_image: Image,
        render_region: Region,
        interp: InterpolationOption,
    ) -> Image:

        # Matriz base da Viewport (Zoom + Pan + Fit)
        # m_view = viewport.roi_matrix @ viewport.fit_matrix(layer.region.size)
        m_view = viewport.roi_matrix @ viewport.fit_matrix(layer.canvas_size)
        m_layer_global = mat_global(layer)

        for edit_layer in layer._edits:
            # 1. Resolve a fonte de pixels ideal (LOD) para o zoom atual
            # Passamos a escala combinada para a heurística
            pixels, m_adjust = self.lod_manager.get_source(
                    viewport, edit_layer, layer.region.size
            )

            # 2. Calcula a matriz que projeta este edit direto na tela
            m_edit_viewport = m_view @ m_layer_global @ edit_layer.local_matrix

            # 3. BBox do edit no espaço da Viewport para Culling
            edit_screen_bbox = bbox_to_region(
                calculate_new_bbox(m_edit_viewport, edit_layer.image.size)
            )

            if not edit_screen_bbox.overlaps(render_region):
                continue

            # 4. Onde este edit será pintado nos 800x600 da tela
            dest_region = edit_screen_bbox & render_region

            # 5. Aplica o ajuste de escala do LOD na matriz de renderização
            m_render = m_edit_viewport @ m_adjust

            # 6. Renderização cirúrgica
            src_image = Image(pixels, edit_layer.image.format)
            edit_data = render_patch(
                src_image, m_render, dest_region, layer._warp_mode, interp,
            )

            if edit_data is None:
                continue

            edit_image = Image(edit_data, edit_layer.image.format)

            # 7. Blend no buffer da Viewport
            # O render_region representa o pedaço da tela que estamos pintando
            blend_region = dest_region - render_region.top_left
            blend = BLEND_MODE.get(edit_layer.blend_mode)

            blend(layer_image.view(blend_region), edit_image)

        return layer_image

    def _final_region(self, layer: Layer, viewport: Viewport):
        # m_view = viewport.roi_matrix @ viewport.fit_matrix(layer.region.size)
        m_view = viewport.roi_matrix @ viewport.fit_matrix(layer.canvas_size)
        m_global = m_view @ mat_global(layer)
        return bbox_to_region(calculate_new_bbox(m_global, layer.region.size))

    def render_area(
        self,
        layer: Layer,
        viewport: Viewport,
        interp: InterpolationOption = InterpolationOption.LANCZOS
    ) -> Image | None:

        flags = layer._resolve_render()
        final_region = self._final_region(layer, viewport)

        view_region = viewport.region
        render_region = self.__render_region(final_region, view_region)

        if render_region:

            if flags & RenderFlags.PIXELS or layer._id not in self._cache:
                size = render_region.size
                layer_image = Image.new(size, layer.format)
                image = self.__flatten_edits(layer, viewport, layer_image, render_region, interp)
                if viewport.scale.sx == 1.0:
                    self._cache[layer._id] = image

                # Cria a miniatura do layer
                layer._opacity_mask = generate_opacity_mask(image, render_region, viewport.size, self._target_size)

                return image

            elif layer._id in self._cache:
                view = final_region.overlap_with(render_region)
                image = self._cache[layer._id].crop(view)

                # Cria a miniatura do layer
                layer._opacity_mask = generate_opacity_mask(image, render_region, viewport.size, self._target_size)

                return image

    def render_scene(
        self,
        layers: Iterable[Layer],
        viewport: Viewport,
        interp: InterpolationOption = InterpolationOption.LANCZOS
    ) -> Image:

        target = (32, 32)
        images = []
        miniview = np.zeros(target)

        for layer in layers:
            if layer.visible is False:
                continue

            image = self.render_area(layer, viewport, interp)

            if image:
                images.append((layer, image))
                np.maximum(miniview, layer._opacity_mask, out=miniview)

                if np.all(miniview == 255):
                    break

        composition = Image.new(viewport.size, ImageFormat.RGBA)
        for layer, image in reversed(images):
            final_region = self._final_region(layer, viewport)
            render_region = self.__render_region(final_region, viewport.region)
            blend = BLEND_MODE.get(layer.blend_mode)
            print('final_region', final_region)
            print('render_region', render_region)
            blend(composition.view(render_region), image, layer.opacity)
        return composition
