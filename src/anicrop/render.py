from abc import ABC, abstractmethod
from anicrop.blend import blend_rendered_images, BLEND_MODE
from anicrop.canvas import Canvas
from anicrop.enums import InterpolationOption, WarpMode
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
from typing import Optional, Iterable, Any
from anicrop.viewport import Viewport

import cv2
import numpy as np


def warp_affine(
    src_data: np.ndarray,
    m_cv2: np.ndarray,
    dest_size: tuple[int, int],
    interp: InterpolationOption = InterpolationOption.LINEAR
):
    M_affine = m_cv2[:2, :].astype(np.float64)

    # Usa BORDER_REPLICATE para que o kernel de interpolação (Lanczos/Linear) não amoste
    # pixels nulos (0,0,0,0) fora do limite do retalho, eliminando a moldura/franja na borda.
    return cv2.warpAffine(
        src_data,
        M_affine,
        dest_size,
        flags=interp.value,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
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

    # 1. Região ideal com a margem do Lanczos (pode ter start negativo)
    target_region = bbox_to_region(calculate_region_bbox(
        mat_inverse(matrix_global), dest_region
    )).expand(all=interp.padding)

    image_region = Region.from_size(*src_image.size)

    if image_region.overlaps(target_region):
        # 2. Interseção segura recortada da imagem física no NumPy
        valid_region = image_region & target_region
        src_data = src_image[valid_region]

        # 3. Calcula o padding usando offset_to (inicio e fim)
        pad_start = target_region.offset_to(valid_region)
        pad_end = valid_region.offset_to(target_region, anchor_end=True)

        if (pad_start.x | pad_start.y | pad_end.x | pad_end.y) > 0:
            src_data = np.pad(
                src_data,
                ((pad_start.y, pad_end.y), (pad_start.x, pad_end.x), (0, 0)),
                mode='constant'
            )

        # 4. A origem da matriz da sub-imagem é sempre o top_left da target_region!
        M_src_offset = mat_translation(*target_region.top_left)

        dst_x, dst_y = dest_region.top_left
        M_dst_offset_inv = mat_translation(-dst_x, -dst_y)

        M_cv2 = (M_dst_offset_inv @ matrix_global @
                 M_src_offset).astype(np.float64)

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
        mini_mask = cv2.resize(eroded, (tw_img, th_img),
                               interpolation=cv2.INTER_NEAREST)
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


class BaseRenderPlan(ABC):

    def __init__(
            self,
            bounds: Region,
            view_region: None | Region,
            matrix: np.ndarray = np.identity(3, dtype=np.float32)
    ):
        self._bounds = bounds
        self._matrix = matrix
        self._dst_region = self._render_region(self.bounds, view_region)
        self._src_region = self._view_region(self.bounds, self.dst_region)

    @abstractmethod
    def _render_region(self, final_region: Region, view_region: Region) -> None | Region:
        ...

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


class ViewportPlan(BaseRenderPlan):
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

        if local:
            # 1. ESTADO LOCAL NA VIEWPORT (Mexicano Deitado na Tela com Zoom/Pan da Câmera):
            matrix = m_view
        else:
            # 2. ESTADO GLOBAL NA VIEWPORT (Mexicano Em Pé na Tela com Zoom/Pan da Câmera):
            matrix = m_view @ mat_global(layer)

        bounds = bbox_to_region(calculate_new_bbox(matrix, layer.region.size))

        super().__init__(bounds, viewport.region, matrix=matrix)

    def _render_region(self, final_region: Region, view_region: Optional[Region]) -> Optional[Region]:
        if view_region is not None and view_region.overlaps(final_region):
            return view_region & final_region
        return None


class CanvasPlan(BaseRenderPlan):
    def __init__(
        self,
        layer: Layer,
        view_region: Optional[Region] = None,
        local: bool = False,
        scale_factor: float = 1.0,
    ):
        self.layer = layer
        self.local = local

        m_global = mat_global(layer)
        m_lod = mat_scale(scale_factor, scale_factor)

        if local:
            # 1. ESTADO LOCAL (Mexicano Deitado) NO NÍVEL DE LOD:
            matrix = m_lod

            # Levar view_region (Global 1:1) -> Local 1:1 -> Local LOD
            if view_region is not None:
                inv_matrix = m_lod @ mat_inverse(m_global)
                view_target = bbox_to_region(
                    calculate_region_bbox(inv_matrix, view_region)
                )
            else:
                view_target = None
        else:
            # 2. ESTADO GLOBAL (Mexicano Em Pé) NO NÍVEL DE LOD:
            matrix = m_lod @ m_global

            # Levar view_region (Global 1:1) -> Global LOD
            if view_region is not None:
                view_target = bbox_to_region(
                    calculate_region_bbox(m_lod, view_region)
                )
            else:
                view_target = None

        bounds = bbox_to_region(
            calculate_new_bbox(matrix, layer.region.size)
        )

        super().__init__(bounds, view_target, matrix=matrix)

    def _render_region(self, final_region: Region, view_region: Optional[Region]) -> Optional[Region]:
        if not view_region:
            return final_region
        elif view_region.overlaps(final_region):
            return view_region & final_region
        return None

    @property
    def matrix(self):
        return self._matrix


def render_image(
    image: Image,
    plan: BaseRenderPlan,
    m_local: np.ndarray,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
) -> tuple[Image, Region] | None:
    """Núcleo atômico: renderiza cirurgicamente qualquer Image usando o plano e a matriz local m_local."""
    m_render = plan.matrix @ m_local

    # Bounding box projetado no espaço de destino do plano
    edit_bbox = bbox_to_region(
        calculate_new_bbox(m_render, image.size)
    )

    # Culling: Descarta se a edição não colide com a região visível no destino
    if plan.dst_region is None or not edit_bbox.overlaps(plan.dst_region):
        return None

    dest_region = edit_bbox & plan.dst_region

    # Executa o warp da imagem para a dest_region no espaço de destino
    pixel_data = render_patch(
        image,
        m_render,
        dest_region,
        warp_mode,
        interp
    )

    if pixel_data is None:
        return None

    warped_image = Image(pixel_data, image.format)
    return warped_image, dest_region - plan.dst_region


def render_edit(
    edit_layer: EditLayer,
    plan: BaseRenderPlan,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
) -> tuple[Image, Region] | None:
    """Renderiza cirurgicamente um EditLayer ajustando o LOD automaticamente através do plano."""
    scale = plan.screen_scale(edit_layer)
    lod_image, m_local = edit_layer.get_lod(scale)

    return render_image(
        lod_image,
        plan,
        m_local,
        warp_mode=warp_mode,
        interp=interp,
    )


def render_viewport_edit(
    edit_layer: EditLayer,
    plan: BaseRenderPlan,
    scale_factor: float = 1.0,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
) -> tuple[Image, Region] | None:
    """Auxiliar da Viewport: repassa para render_edit (o plano calcula a escala)."""
    return render_edit(edit_layer, plan, warp_mode=warp_mode, interp=interp)


class CanvasRender:

    def __init__(self):
        self._target_size = (32, 32)

    def _flatten_edits(
        self,
        layer: Layer,
        layer_image: Image,
        plan: BaseRenderPlan,
        interp: InterpolationOption,
    ) -> Image:

        for edit_layer in layer._edits:
            result = render_edit(edit_layer, plan, interp=interp)
            if result is None:
                continue
            edit_image, dst_region = result
            blend = BLEND_MODE.get(edit_layer.blend_mode)
            blend(layer_image.view(dst_region), edit_image)

        return layer_image

    def render_area(
        self,
        layer: Layer,
        plan: Optional[CanvasPlan] = None,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
        local: bool = False,
    ) -> Image | None:

        plan = plan if plan else CanvasPlan(layer, None, local)

        dst_region = plan.dst_region
        if dst_region is not None:
            layer_image = Image.new(dst_region.size, layer.format)
            image = self._flatten_edits(layer, layer_image, plan, interp)

            return image

        return None

    def render(
        self,
        layer: Layer,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
        local: bool = False,
    ) -> Image | None:

        return self.render_area(layer, interp=interp, local=local)

    def render_scene(
        self,
        layers: Iterable[Layer],
        canvas: Canvas,
        interp: InterpolationOption = InterpolationOption.LANCZOS
    ) -> Image:

        target = (32, 32)
        images = []
        miniview = np.zeros(target)

        for layer in layers:
            if layer.visible is False:
                continue

            plan = CanvasPlan(layer, canvas.region)
            image = self.render_area(layer, plan, interp=interp)

            if image:
                layer._opacity_mask = generate_opacity_mask(
                    image, plan.dst_region, canvas.size, self._target_size
                )
                images.append((layer, image, plan))
                np.maximum(miniview, layer._opacity_mask, out=miniview)

                if np.all(miniview == 255):
                    break

        composition = Image.new(canvas.size, ImageFormat.RGBA)
        for layer, image, plan in reversed(images):
            dst_region = plan.dst_region
            blend = BLEND_MODE.get(layer.blend_mode)
            blend(composition.view(dst_region), image, layer.opacity)
        composition = Image.new(canvas.size, ImageFormat.RGBA)
        return blend_rendered_images(reversed(images), composition)


class ViewportRender:

    def __init__(self):
        self._target_size = (32, 32)

    def __flatten_edits(
        self,
        layer: Layer,
        plan: ViewportPlan,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
    ) -> Image:
        layer_image = Image.new(plan.dst_region.size, layer.format)
        for edit_layer in layer._edits:
            result = render_edit(edit_layer, plan, interp=interp)
            if result is None:
                continue
            edit_image, dst_region = result
            blend = BLEND_MODE.get(edit_layer.blend_mode)
            blend(layer_image.view(dst_region), edit_image)

        return layer_image

    def render_area(
        self,
        layer: Layer,
        plan: ViewportPlan,
        interp: InterpolationOption = InterpolationOption.LANCZOS
    ) -> Image | None:

        dst_region = plan.dst_region
        if dst_region is not None:
            image = self.__flatten_edits(layer, plan, interp=interp)

            # Cria a miniatura do layer
            layer._opacity_mask = generate_opacity_mask(
                image, dst_region, plan.viewport.size, self._target_size
            )

            return image
        return None

    def render_scene(
        self,
        layers: Any,
        viewport: Viewport,
        interp: InterpolationOption = InterpolationOption.LANCZOS
    ) -> Image:

        target = self._target_size
        images = []
        miniview = np.zeros(target, dtype=np.uint8)

        if hasattr(layers, 'render'):
            occluded, imgs = layers.render(
                renderer=self.render_area,
                plan_cls=ViewportPlan,
                surface=viewport,
                miniview=miniview,
                interp=interp
            )
            images.extend(imgs)
        else:
            for layer in layers:
                if layer.visible is False:
                    continue

                plan = ViewportPlan(layer, viewport)
                image = self.render_area(layer, plan, interp=interp)

                if image:
                    images.append((layer, image, plan))
                    np.maximum(miniview, layer._opacity_mask, out=miniview)

                    if np.all(miniview == 255):
                        break

        composition = Image.new(viewport.size, ImageFormat.RGBA, color=viewport.bg_color)
        return blend_rendered_images(reversed(images), composition)
