from __future__ import annotations
from anicrop.frame import (
    BaseFrame,
    CanvasFrame,
    ViewportFrame,
)
from anicrop.blend import blend_rendered_images, BLEND_MODE
from anicrop.enums import InterpolationOption, WarpMode
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer, EditLayer
from anicrop.container import GroupLayer, Container
from anicrop.spatial import Region, rect_to_region
from anicrop.transform import (
    calculate_new_rect,
    calculate_region_rect,
    mat_inverse,
    mat_translation
)
from anicrop.viewport import Viewport
from abc import ABC
from typing import Iterable, Protocol, runtime_checkable


import cv2
import numpy as np


@runtime_checkable
class SurfaceProtocol(Protocol):
    size: tuple[int, int]
    region: Region
    bg_color: tuple[int, int, int, int] | None


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
    target_region = rect_to_region(calculate_region_rect(
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


def render_image(
    image: Image,
    plan: BaseFrame,
    m_local: np.ndarray,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
) -> tuple[Image, Region] | None:
    """Núcleo atômico: renderiza cirurgicamente qualquer Image usando o plano/frame e a matriz local m_local."""
    m_render = plan.matrix @ m_local

    # Bounding box projetado no espaço de destino do plano/frame
    edit_bbox = rect_to_region(
        calculate_new_rect(m_render, image.size)
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
    plan: BaseFrame,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
) -> tuple[Image, Region] | None:
    """Renderiza cirurgicamente um EditLayer ajustando o LOD automaticamente através do frame."""
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
    plan: BaseFrame,
    scale_factor: float = 1.0,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
) -> tuple[Image, Region] | None:
    """Auxiliar da Viewport: repassa para render_edit (o frame calcula a escala)."""
    return render_edit(edit_layer, plan, warp_mode=warp_mode, interp=interp)


class SceneTraverser:
    """
    Encapsulates state and recursive traversal logic for rendering a 2D scene,
    executing opacity culling and composition of nested GroupLayers.
    """

    def __init__(
        self,
        renderer: BaseRenderer,
        surface: SurfaceProtocol,
        frame_cls: type[BaseFrame],
        interp: InterpolationOption = InterpolationOption.LANCZOS,
        target_size: tuple[int, int] = (32, 32),
    ):
        self.renderer = renderer
        self.surface = surface
        self.frame_cls = frame_cls
        self.interp = interp
        self.target_size = target_size
        self.miniview = np.zeros(target_size, dtype=np.uint8)

    def traverse(
        self,
        container: Iterable[Layer | GroupLayer] | Container
    ) -> list[tuple[Layer | GroupLayer, Image, BaseFrame]]:
        rendered_items = []

        for item in container:
            if not item.visible:
                continue

            if isinstance(item, GroupLayer):
                children_items = self.traverse(item)

                if children_items:
                    buffer = Image.new(self.surface.size, ImageFormat.RGBA)
                    group_image = blend_rendered_images(children_items, buffer)
                    group_frame = self.frame_cls(item, self.surface)

                    rendered_items.append((item, group_image, group_frame))
                    if np.all(self.miniview == 255):
                        break
            else:
                image = self.renderer.render_area(item, self.surface, self.interp)

                if image is not None:
                    frame = self.frame_cls(item, self.surface)
                    rendered_items.append((item, image, frame))
                    if item._opacity_mask is not None:
                        np.maximum(self.miniview, item._opacity_mask, out=self.miniview)
                    if np.all(self.miniview == 255):
                        break

        return rendered_items


class BaseRenderer(ABC):

    def __init__(
            self, frame_cls: type[BaseFrame], target_size: tuple[int, int] = (32, 32),
    ):

        self.frame_cls = frame_cls
        self._target_size = target_size

    def _flatten_edits(
        self,
        layer: Layer,
        layer_image: Image,
        plan: BaseFrame,
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
        view: Viewport | Region | None = None,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
        local: bool = False,
    ) -> Image | None:
        plan = self.frame_cls(layer, view, local=local)

        dst_region = plan.dst_region
        if dst_region is not None:
            layer_image = Image.new(dst_region.size, layer.format)
            image = self._flatten_edits(layer, layer_image, plan, interp)

            layer._opacity_mask = generate_opacity_mask(
                image, dst_region, plan.surface_size, self._target_size
            )

            return image

        return None

    def render_scene(
        self,
        container: Iterable[Layer | GroupLayer] | Container,
        surface: SurfaceProtocol,
        interp: InterpolationOption = InterpolationOption.LANCZOS
    ) -> Image:

        traverser = SceneTraverser(
            self, surface, self.frame_cls, interp=interp, target_size=self._target_size
        )
        images = traverser.traverse(container)

        composition = Image.new(surface.size, ImageFormat.RGBA, color=surface.bg_color)
        return blend_rendered_images(reversed(images), composition)


class CanvasRender(BaseRenderer):

    def __init__(self):
        super().__init__(frame_cls=CanvasFrame)

    def render_area(
        self,
        layer: Layer,
        view: Region | None = None,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
        local: bool = False,
    ) -> Image | None:
        return super().render_area(layer, view=view, interp=interp, local=local)

    def render(
        self,
        layer: Layer,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
        local: bool = False,
    ) -> Image | None:
        return self.render_area(layer, interp=interp, local=local)


class ViewportRender(BaseRenderer):

    def __init__(self):
        super().__init__(frame_cls=ViewportFrame)

    def render_area(
        self,
        layer: Layer,
        view: Viewport,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
        local: bool = False,
    ) -> Image | None:
        return super().render_area(layer, view=view, interp=interp, local=local)
