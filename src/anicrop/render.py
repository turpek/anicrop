from __future__ import annotations
from abc import ABC
from typing import Any, Callable, Iterable


import cv2
import numpy as np

from anicrop.blend import blend_rendered_images, BLEND_MODE
from anicrop.canvas import Canvas
from anicrop.container import BaseLayer, Container, GroupLayer, freeze_geometry
from anicrop.enums import BlendMode, InterpolationOption, WarpMode
from anicrop.frame import (
    BaseFrame,
    CanvasFrame,
    SurfaceProtocol,
    ViewportFrame,
)
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer, EditLayer
from anicrop.spatial import Region, rect_to_region
from anicrop.transform import (
    calculate_new_rect,
    calculate_region_rect,
    has_distortion,
    mat_inverse,
    mat_translation,
)


def warp_affine(
    src_data: np.ndarray,
    m_cv2: np.ndarray,
    dest_size: tuple[int, int],
    interp: InterpolationOption = InterpolationOption.LINEAR,
    dst: np.ndarray | None = None,
) -> np.ndarray:
    M_affine = m_cv2[:2, :].astype(np.float64)

    # Usa BORDER_REPLICATE para que o kernel de interpolação (Lanczos/Linear) não amoste
    # pixels nulos (0,0,0,0) fora do limite do retalho, eliminando a moldura/franja na borda.
    return cv2.warpAffine(
        src_data,
        M_affine,
        dest_size,
        dst=dst,
        flags=interp.value,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def warp_perspective(
    src_data: np.ndarray,
    m_cv2: np.ndarray,
    dest_size: tuple[int, int],
    interp: InterpolationOption = InterpolationOption.LINEAR,
    dst: np.ndarray | None = None,
) -> np.ndarray:
    return cv2.warpPerspective(
        src_data,
        m_cv2,
        dest_size,
        dst=dst,
        flags=interp.value,
    )


WARP_MODE = {
    WarpMode.AFFINE: warp_affine,
    WarpMode.PERSPECTIVE: warp_perspective,
}


def warp_patch(
    src_image: Image,
    matrix_global: np.ndarray,
    dest_region: Region,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
    dst: np.ndarray | None = None,
) -> np.ndarray | None:

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
        pad_x_start, pad_y_start = target_region.offset_to(valid_region)
        pad_x_end, pad_y_end = valid_region.offset_to(target_region, anchor_end=True)

        if (pad_x_start | pad_y_start | pad_x_end | pad_y_end) > 0:
            src_data = np.pad(
                src_data,
                ((pad_y_start, pad_y_end), (pad_x_start, pad_x_end), (0, 0)),
                mode='constant'
            )

        # 4. A origem da matriz da sub-imagem é sempre o top_left da target_region!
        M_src_offset = mat_translation(*target_region.top_left)

        dst_x, dst_y = dest_region.top_left
        M_dst_offset_inv = mat_translation(-dst_x, -dst_y)

        M_cv2 = (M_dst_offset_inv @ matrix_global @
                 M_src_offset).astype(np.float64)

        warp = WARP_MODE.get(warp_mode, warp_affine)
        return warp(src_data, M_cv2, dest_region.size, interp, dst=dst)

    return None


def generate_opacity_mask(
    image: Image,
    render_region: Region,
    viewport_size: tuple[int, int],
    target_size=(32, 32),
    opacity: float = 1.0,
    blend_mode: BlendMode = BlendMode.NORMAL,
) -> np.ndarray:
    """Função usada para gerar miniaturas do layer mapeadas proporcionalmente na tela"""

    eroded_alpha = np.zeros(target_size, dtype=np.uint8)

    if blend_mode != BlendMode.NORMAL or opacity <= 0.0:
        return eroded_alpha

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

    if opacity < 1.0:
        mini_mask = (mini_mask.astype(np.float32) * opacity).astype(np.uint8)

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


def without_distortion(
    image: Image,
    edit_bbox: Region,
    dst_frame: Region,
    dst_local: Region,
    target_dst: np.ndarray | None,
) -> tuple[Image, Region]:

    src_view = edit_bbox.overlap_with(dst_frame)
    if target_dst is not None:
        np.copyto(target_dst, image[src_view])
        return Image(target_dst, image.format), dst_local
    return image.crop(src_view), dst_local


def render_image(
    image: Image,
    plan: BaseFrame,
    m_local: np.ndarray,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
    dst: Image | None = None,
) -> tuple[Image, Region] | None:
    """Núcleo atômico: renderiza cirurgicamente qualquer Image usando o plano/frame e a matriz local m_local."""
    dst_frame = plan.dst_region
    m_render = plan.matrix @ m_local

    # Bounding box projetado no espaço de destino do plano/frame
    edit_bbox = rect_to_region(calculate_new_rect(m_render, image.size))

    # Culling: Descarta se a edição não colide com a região visível no destino
    if dst_frame is None or not edit_bbox.overlaps(dst_frame):
        return None

    dst_patch = edit_bbox & dst_frame
    dst_local = dst_patch - dst_frame
    target_dst = dst[dst_local] if dst else None

    if not has_distortion(m_render):
        return without_distortion(image, edit_bbox, dst_frame, dst_local, target_dst)

    # Executa o warp da imagem para a dest_region no espaço de destino
    pixel_data = warp_patch(image, m_render, dst_patch, warp_mode, interp, dst=target_dst)

    if pixel_data is None:
        return None

    warped_image = Image(pixel_data, image.format)
    return warped_image, dst_local


def render_edit(
    edit_layer: EditLayer,
    plan: BaseFrame,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
    dst: Image | None = None,
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
        dst=dst,
    )


def render_viewport_edit(
    edit_layer: EditLayer,
    plan: BaseFrame,
    scale_factor: float = 1.0,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
    dst: Image | None = None,
) -> tuple[Image, Region] | None:
    """Auxiliar da Viewport: repassa para render_edit (o frame calcula a escala)."""
    return render_edit(edit_layer, plan, warp_mode=warp_mode, interp=interp, dst=dst)


def apply_post_processing(
    target_image: Image,
    base: BaseLayer,
    frame: BaseFrame,
    interp: InterpolationOption = InterpolationOption.LANCZOS,
) -> Image:
    """Aplica a fila de efeitos e a modulação de máscara sobre a imagem rasterizada de uma camada ou grupo."""
    image = target_image

    for effect in base.effects:
        image = effect.apply(image, frame.matrix)

    for mask in base.masks:
        mask_result = render_edit(mask, frame, interp=interp)
        if mask_result is not None:
            mask_image, dst_local = mask_result
            mask.apply_modulation(image.view(dst_local), mask_image)

    return image


class SceneTraverser:
    """
    Encapsulates state and recursive traversal logic for rendering a 2D scene,
    executing opacity culling and composition of nested GroupLayers.
    """

    def __init__(
        self,
        renderer: BaseRenderer,
        surface: SurfaceProtocol,
        frame_cls: Callable[..., BaseFrame],
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
        container: Iterable[Layer | GroupLayer] | Container,
        view_region: Region | None = None,
        local=False,
    ) -> list[tuple[BaseLayer, Image, Region]]:
        rendered_items: list[tuple[BaseLayer, Image, Region]] = []

        for item in reversed(container):
            frame = self.frame_cls(item, self.surface, view_region, local=local)
            if not item.is_renderable or frame.dst_region is None:
                continue

            if isinstance(item, GroupLayer):
                children_items = self.traverse(item, frame.dst_region)

                if children_items:
                    buffer = Image.new(frame.dst_region.size, ImageFormat.RGBA)
                    group_image = blend_rendered_images(reversed(children_items), buffer)
                    group_image = apply_post_processing(group_image, item, frame, self.interp)
                    rendered_items.append((item, group_image, frame.targ_region))
                    if np.all(self.miniview == 255):
                        break
            else:
                image = self.renderer.render_area(item, frame, self.interp)

                if image:
                    rendered_items.append((item, image, frame.targ_region))

                    if item._opacity_mask is not None:
                        np.maximum(self.miniview, item._opacity_mask, out=self.miniview)
                    if np.all(self.miniview == 255):
                        break

        return rendered_items


class BaseRenderer[FrameT: BaseFrame](ABC):

    def __init__(
            self, frame_cls: type[FrameT], target_size: tuple[int, int] = (32, 32),
    ):

        self.frame_cls = frame_cls
        self._target_size = target_size
        self._scratch_image: Image | None = None

    def _get_scratch_buffer(
        self, width: int, height: int, fmt: ImageFormat = ImageFormat.RGBA
    ) -> Image:
        if (
            self._scratch_image is None or
            self._scratch_image.height < height or
            self._scratch_image.width < width or
            self._scratch_image.format != fmt
        ):
            current_h = self._scratch_image.height if self._scratch_image is not None else 0
            current_w = self._scratch_image.width if self._scratch_image is not None else 0
            new_h = max(height, int(current_h * 1.5))
            new_w = max(width, int(current_w * 1.5))
            self._scratch_image = Image.new((new_w, new_h), fmt)

        return self._scratch_image.view(Region.from_size(width, height))

    def _flatten_edits(
        self,
        layer: Layer,
        layer_image: Image,
        plan: BaseFrame,
        interp: InterpolationOption,
    ) -> Image:
        for edit_layer in layer._edits:
            scratch = self._get_scratch_buffer(*layer_image.size, edit_layer.image.format)
            result = render_edit(edit_layer, plan, interp=interp, dst=scratch)
            if result is None:
                continue
            edit_image, dst_region = result
            blend = BLEND_MODE[edit_layer.blend_mode]
            blend(layer_image.view(dst_region), edit_image)

        return layer_image

    def render_area(
        self,
        layer: Layer,
        frame: FrameT,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
    ) -> Image | None:

        dst_region = frame.dst_region
        if dst_region is not None:
            layer_image = Image.new(dst_region.size, layer.format)
            image = self._flatten_edits(layer, layer_image, frame, interp)
            image = apply_post_processing(image, layer, frame, interp)

            layer._opacity_mask = generate_opacity_mask(
                image,
                dst_region,
                frame.surface_size,
                self._target_size,
                opacity=layer.opacity,
                blend_mode=layer.blend_mode,
            )

            return image

        return None

    def render_scene(
        self,
        container: Iterable[Layer | GroupLayer] | Container,
        surface: SurfaceProtocol,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
    ) -> Image:
        with freeze_geometry(container):
            traverser = SceneTraverser(
                self, surface, self.frame_cls, interp=interp, target_size=self._target_size
            )
            images = traverser.traverse(container)

            composition = Image.new(surface.size, ImageFormat.RGBA, color=surface.bg_color)
            return blend_rendered_images(reversed(images), composition)

    def render_patch(
        self,
        container: Iterable[Layer | GroupLayer] | Container,
        surface: SurfaceProtocol,
        view_region: Region,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
    ) -> Image | None:
        if not surface.region.overlaps(view_region):
            return None

        effective_region = surface.region & view_region
        with freeze_geometry(container):
            traverser = SceneTraverser(
                self, surface, self.frame_cls, interp=interp, target_size=self._target_size
            )
            images = traverser.traverse(container, effective_region)

            composition = Image.new(effective_region.size, ImageFormat.RGBA, color=surface.bg_color)
            return blend_rendered_images(reversed(images), composition)


class CanvasRender(BaseRenderer[CanvasFrame]):

    def __init__(self):
        super().__init__(frame_cls=CanvasFrame)

    def render_layer(
        self,
        layer: Layer,
        interp: InterpolationOption = InterpolationOption.LANCZOS,
        local: bool = False,
    ) -> Image | None:
        frame = CanvasFrame(layer, Canvas(layer.global_region), local=local)
        return self.render_area(layer, frame, interp=interp)


class ViewportRender(BaseRenderer[ViewportFrame]):

    def __init__(self):
        super().__init__(frame_cls=ViewportFrame)
