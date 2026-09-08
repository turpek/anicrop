from __future__ import annotations

from abc import ABC
from functools import reduce
from operator import or_
from typing import Callable, Sequence

import cv2
import numpy as np

from anicrop.blend import blend_rendered_images
from anicrop.canvas import Canvas
from anicrop.container import BaseLayer, Container, GroupLayer, freeze_geometry
from anicrop.enums import BlendMode, InterpMode, WarpMode
from anicrop.frame import (
    BaseFrame,
    CanvasFrame,
    SurfaceProtocol,
    ViewportFrame,
)
from anicrop.image import Image, ImageFormat
from anicrop.interfaces.buffer import AbstractScratchBuffer
from anicrop.layer import EditLayer, Layer
from anicrop.scratch import ScratchBuffer
from anicrop.spatial import Point, Region, rect_to_region
from anicrop.transform import (
    calculate_new_rect,
    calculate_region_rect,
    create_pivot_transform_rel,
    has_distortion,
    mat_inverse,
    mat_rotation,
    mat_scale,
    mat_translation,
)

try:
    from anicrop.native.blend import (  # type: ignore[import-untyped,import-not-found]
        min_pool_alpha as _cy_min_pool_alpha,
    )
except ImportError:
    _cy_min_pool_alpha = None


def pad_straight_alpha_into(
    buf: np.ndarray,
    src_data: np.ndarray,
    pad_y_start: int,
    pad_y_end: int,
    pad_x_start: int,
    pad_x_end: int,
) -> None:
    """Preenche as margens de buf replicando cores da borda de src_data com alfa zero."""
    sh, sw = src_data.shape[:2]
    buf[pad_y_start:pad_y_start + sh, pad_x_start:pad_x_start + sw] = src_data
    if pad_y_start > 0:
        buf[:pad_y_start, pad_x_start:pad_x_start + sw, :-1] = src_data[0:1, :, :-1]
        buf[:pad_y_start, :, -1] = 0
    if pad_y_end > 0:
        buf[pad_y_start + sh:, pad_x_start:pad_x_start + sw, :-1] = src_data[-1:, :, :-1]
        buf[pad_y_start + sh:, :, -1] = 0
    if pad_x_start > 0:
        buf[:, :pad_x_start, :-1] = buf[:, pad_x_start:pad_x_start + 1, :-1]
        buf[:, :pad_x_start, -1] = 0
    if pad_x_end > 0:
        buf[:, pad_x_start + sw:, :-1] = buf[:, pad_x_start + sw - 1:pad_x_start + sw, :-1]
        buf[:, pad_x_start + sw:, -1] = 0


def pad_straight_alpha_symmetric(
    src_data: np.ndarray,
    pad: int,
) -> np.ndarray:
    """Cria um array com margem simetrica replicando cores e zerando o canal alfa."""
    sh, sw = src_data.shape[:2]
    buf = np.empty((sh + 2 * pad, sw + 2 * pad, src_data.shape[2]), dtype=src_data.dtype)
    pad_straight_alpha_into(buf, src_data, pad, pad, pad, pad)
    return buf


def warp_affine(
    src_data: np.ndarray,
    m_cv2: np.ndarray,
    dest_size: tuple[int | float, int | float] | Sequence[int | float],
    interp: InterpMode = InterpMode.LINEAR,
    dst: np.ndarray | None = None,
    format: ImageFormat | None = None,
    auto_pad: bool = True,
) -> np.ndarray:
    M_affine = m_cv2[:2, :].astype(np.float64)
    dsize = (int(round(dest_size[0])), int(round(dest_size[1])))

    is_straight = False
    if format is not None:
        is_straight = format.is_straight_alpha
    elif src_data.ndim == 3 and src_data.shape[2] in (2, 4):
        is_straight = True

    if auto_pad and is_straight and interp.padding > 0:
        pad = interp.padding
        src_data = pad_straight_alpha_symmetric(src_data, pad)
        M_affine = M_affine.copy()
        M_affine[0, 2] -= (M_affine[0, 0] + M_affine[0, 1]) * pad
        M_affine[1, 2] -= (M_affine[1, 0] + M_affine[1, 1]) * pad

    border_val = (0,) * src_data.shape[2] if src_data.ndim == 3 else (0,)
    return cv2.warpAffine(
        src_data,
        M_affine,
        dsize,
        dst=dst,
        flags=interp.value,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )


def warp_perspective(
    src_data: np.ndarray,
    m_cv2: np.ndarray,
    dest_size: tuple[int | float, int | float] | Sequence[int | float],
    interp: InterpMode = InterpMode.LINEAR,
    dst: np.ndarray | None = None,
    format: ImageFormat | None = None,
    auto_pad: bool = True,
) -> np.ndarray:
    dsize = (int(round(dest_size[0])), int(round(dest_size[1])))
    M_persp = m_cv2.astype(np.float64)

    is_straight = False
    if format is not None:
        is_straight = format.is_straight_alpha
    elif src_data.ndim == 3 and src_data.shape[2] in (2, 4):
        is_straight = True

    if auto_pad and is_straight and interp.padding > 0:
        pad = interp.padding
        src_data = pad_straight_alpha_symmetric(src_data, pad)
        M_persp = M_persp.copy()
        M_persp[:, 2] -= (M_persp[:, 0] + M_persp[:, 1]) * pad

    border_val = (0,) * src_data.shape[2] if src_data.ndim == 3 else (0,)
    return cv2.warpPerspective(
        src_data,
        M_persp,
        dsize,
        dst=dst,
        flags=interp.value,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )


WARP_MODE = {
    WarpMode.AFFINE: warp_affine,
    WarpMode.PERSPECTIVE: warp_perspective,
}


_WARP_SRC_SCRATCH = ScratchBuffer()


def warp_patch(
    src_image: Image,
    matrix_global: np.ndarray,
    dest_region: Region,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpMode = InterpMode.LANCZOS,
    dst: np.ndarray | None = None,
) -> np.ndarray | None:

    # 1. Região ideal com a margem do Lanczos (pode ter start negativo)
    target_region = rect_to_region(
        calculate_region_rect(mat_inverse(matrix_global), dest_region)
    ).expand(all=interp.padding)

    image_region = Region.from_size(*src_image.size)

    if image_region.overlaps(target_region):
        # 2. Interseção segura recortada da imagem física no NumPy
        valid_region = image_region & target_region
        src_data = src_image[valid_region]

        # 3. Calcula o padding (início e fim) uma única vez
        pad_x_start, pad_y_start = target_region.offset_to(valid_region).to_int(
            mode="round"
        )
        pad_x_end, pad_y_end = valid_region.offset_to(
            target_region, anchor_end=True
        ).to_int(mode="round")

        pad_x_start = max(0, pad_x_start)
        pad_y_start = max(0, pad_y_start)
        pad_x_end = max(0, pad_x_end)
        pad_y_end = max(0, pad_y_end)

        if pad_x_start > 0 or pad_y_start > 0 or pad_x_end > 0 or pad_y_end > 0:
            sh, sw = src_data.shape[:2]
            out_h = pad_y_start + sh + pad_y_end
            out_w = pad_x_start + sw + pad_x_end
            buf = _WARP_SRC_SCRATCH.configure(
                (out_w, out_h), src_image.format, dtype=src_image.dtype
            )[Region.from_size(out_w, out_h)]
            if src_image.format.is_straight_alpha:
                pad_straight_alpha_into(
                    buf, src_data, pad_y_start, pad_y_end, pad_x_start, pad_x_end
                )
            else:
                buf.fill(0)
                buf[pad_y_start:pad_y_start + sh, pad_x_start:pad_x_start + sw] = src_data
            src_data = buf

        # 4. A origem da matriz da sub-imagem é sempre o top_left da target_region!
        M_src_offset = mat_translation(*target_region.top_left)

        dst_x, dst_y = dest_region.top_left
        M_dst_offset_inv = mat_translation(-dst_x, -dst_y)

        M_cv2 = (M_dst_offset_inv @ matrix_global @ M_src_offset).astype(np.float64)

        warp = WARP_MODE.get(warp_mode, warp_affine)
        dest_size = (int(round(dest_region.width)), int(round(dest_region.height)))
        return warp(src_data, M_cv2, dest_size, interp, dst=dst, format=src_image.format, auto_pad=False)

    return None


def transform_image(
    image: Image,
    angle: float = 0.0,
    scale: float | tuple[float, float] = 1.0,
    pivot_angle: tuple[float, float] | Point = (0.5, 0.5),
    pivot_scale: tuple[float, float] | Point = (0.5, 0.5),
    interp: InterpMode = InterpMode.LINEAR,
    dst: AbstractScratchBuffer | None = None,
    auto_pad: bool = True,
) -> Image:
    """Aplica rotacao e escala com pivos independentes, bounding box exata e protecao de borda."""
    width, height = float(image.width), float(image.height)
    sx, sy = (float(scale), float(scale)) if isinstance(scale, (int, float)) else (float(scale[0]), float(scale[1]))

    M_scale = create_pivot_transform_rel(mat_scale(sx, sy), width, height, *pivot_scale)
    M_rot = create_pivot_transform_rel(mat_rotation(angle), width, height, *pivot_angle)
    M_combined = M_rot @ M_scale

    min_x, min_y, bound_w, bound_h = calculate_new_rect(M_combined, (width, height))
    dsize = (max(1, int(round(bound_w))), max(1, int(round(bound_h))))

    M_affine = M_combined[:2, :].copy()
    M_affine[0, 2] -= min_x
    M_affine[1, 2] -= min_y

    if dst is not None:
        dst.configure(dsize, image.format, dtype=image.dtype)
        dst_data = dst[Region.from_size(dsize[0], dsize[1])]
    else:
        dst_data = None

    res_data = warp_affine(
        image[...],
        M_affine,
        dsize,
        interp=interp,
        dst=dst_data,
        format=image.format,
        auto_pad=auto_pad,
    )
    return Image(res_data, image.format)


def generate_opacity_mask(
    image: Image,
    render_region: Region,
    viewport_size: tuple[float, float],
    target_size: tuple[float, float] = (32.0, 32.0),
    opacity: float = 1.0,
    blend_mode: BlendMode = BlendMode.NORMAL,
) -> np.ndarray:
    """Função usada para gerar miniaturas do layer mapeadas proporcionalmente na tela"""

    dsize = (int(round(target_size[0])), int(round(target_size[1])))
    eroded_alpha = np.zeros(dsize, dtype=np.uint8)

    if blend_mode != BlendMode.NORMAL or opacity <= 0.0:
        return eroded_alpha

    scale_x = target_size[0] / viewport_size[0]
    scale_y = target_size[1] / viewport_size[1]

    tw_img = max(1, int(image.width * scale_x))
    th_img = max(1, int(image.height * scale_y))

    if image.has_alpha:
        mini_mask = np.zeros((th_img, tw_img), dtype=np.uint8)
        if _cy_min_pool_alpha is not None and image.dtype == np.uint8:
            _cy_min_pool_alpha(image[...], mini_mask)
        else:
            alpha_origin = image[..., -1]
            if alpha_origin.dtype == np.uint16:
                alpha_origin = (alpha_origin >> 8).astype(np.uint8)
            elif np.issubdtype(alpha_origin.dtype, np.floating):
                alpha_origin = np.clip(np.round(alpha_origin * 255.0), 0, 255).astype(np.uint8)
            kernel_h = max(1, image.height // th_img)
            kernel_w = max(1, image.width // tw_img)
            kernel = np.ones((kernel_h, kernel_w), dtype=np.uint8)
            eroded = cv2.erode(alpha_origin, kernel)
            mini_mask = cv2.resize(
                eroded, (tw_img, th_img), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
    else:
        mini_mask = np.full((th_img, tw_img), 255, dtype=np.uint8)

    if opacity < 1.0:
        mini_mask = (mini_mask.astype(np.float32) * opacity).astype(np.uint8)

    # Descobre as coordenadas proporcionais na grade target_size
    start_x = int(render_region.top_left[0] * scale_x)
    start_y = int(render_region.top_left[1] * scale_y)

    # Limites seguros na matriz target_size (caso o layer saia da tela)
    max_target_h = int(round(target_size[1]))
    max_target_w = int(round(target_size[0]))
    sy = max(0, start_y)
    ey = min(max_target_h, start_y + th_img)
    sx = max(0, start_x)
    ex = min(max_target_w, start_x + tw_img)

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
    return image.view(src_view), dst_local


def render_image(
    image: Image,
    plan: BaseFrame,
    m_local: np.ndarray,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpMode = InterpMode.LANCZOS,
    dst: AbstractScratchBuffer | Image | None = None,
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

    is_distorted = has_distortion(m_render)
    needs_buffer = is_distorted or (dst_local.size != dst_frame.size)
    target_dst = dst[dst_local] if (dst and needs_buffer) else None

    if not is_distorted:
        return without_distortion(image, edit_bbox, dst_frame, dst_local, target_dst)

    # Executa o warp da imagem para a dest_region no espaço de destino
    pixel_data = warp_patch(
        image, m_render, dst_patch, warp_mode, interp, dst=target_dst
    )

    if pixel_data is None:
        return None

    warped_image = Image(pixel_data, image.format)
    return warped_image, dst_local


def render_edit(
    edit_layer: EditLayer,
    plan: BaseFrame,
    warp_mode: WarpMode = WarpMode.AFFINE,
    interp: InterpMode = InterpMode.LANCZOS,
    dst: AbstractScratchBuffer | Image | None = None,
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
    interp: InterpMode = InterpMode.LANCZOS,
    dst: AbstractScratchBuffer | Image | None = None,
) -> tuple[Image, Region] | None:
    """Auxiliar da Viewport: repassa para render_edit (o frame calcula a escala)."""
    return render_edit(edit_layer, plan, warp_mode=warp_mode, interp=interp, dst=dst)


def apply_post_processing(
    target_image: Image,
    base: BaseLayer,
    frame: BaseFrame,
    interp: InterpMode = InterpMode.LANCZOS,
) -> Image:
    """Aplica a fila de efeitos e a modulação de máscara sobre a imagem rasterizada de uma camada ou grupo."""
    image = target_image

    for effect in base.effects:
        image = effect.apply(image, frame.matrix)

    if base.mask is not None and base.mask.visible:
        mask_result = render_edit(base.mask, frame, interp=interp)
        if mask_result is not None:
            mask_image, dst_local = mask_result
            base.mask.apply_modulation(image.view(dst_local), mask_image)

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
        interp: InterpMode = InterpMode.LANCZOS,
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
        container: Sequence[BaseLayer] | Container,
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
                    buffer = Image.new(
                        frame.dst_region.size,
                        ImageFormat.RGBA,
                        dtype=self.surface.dtype,
                    )
                    group_image = blend_rendered_images(reversed(children_items), buffer)
                    group_image = apply_post_processing(
                        group_image, item, frame, self.interp
                    )
                    rendered_items.append((item, group_image, frame.targ_region))  # type: ignore[arg-type]
                    if np.all(self.miniview == 255):
                        break
            else:
                image = self.renderer.render_area(item, frame, self.interp)

                if image:
                    rendered_items.append((item, image, frame.targ_region))  # type: ignore[arg-type]

                    if item._opacity_mask is not None:
                        np.maximum(self.miniview, item._opacity_mask, out=self.miniview)
                    if np.all(self.miniview == 255):
                        break

        return rendered_items


class BaseRenderer[FrameT: BaseFrame](ABC):
    frame_cls: type[FrameT]

    def __init__(
        self,
        frame_cls: type[FrameT],
        target_size: tuple[int, int] = (32, 32),
    ):

        self.frame_cls = frame_cls
        self._target_size = target_size
        self._scratch_buffer = ScratchBuffer()

    def _render_single_edit(
        self,
        edit_layer: EditLayer,
        layer_format: ImageFormat,
        plan: BaseFrame,
        interp: InterpMode,
    ) -> Image | None:
        """Renderiza um único edit exatamente 1 vez com reciclagem de buffer e fast-path zero-copy."""
        dst = self._scratch_buffer.configure(
            plan.dst_region.size,  # type: ignore[union-attr]
            edit_layer.image.format,
            dtype=edit_layer.image.dtype,
        )
        result = render_edit(edit_layer, plan, interp=interp, dst=dst)
        if result is None:
            return None

        edit_image, dst_region = result

        # 1. Fast-Path: Cobre 100% da área da camada sem uso de buffer (Zero-Copy direto de edit.image!)
        if not dst.was_used:
            return edit_image

        # 2. Fast-Path: Se o resultado cobre 100% da área de destino, desvincula do scratch buffer via crop
        if dst_region.size == plan.dst_region.size:  # type: ignore[union-attr]
            return edit_image.crop()

        # 3. Patch com distorção ou parcial: Mescla o resultado já obtido dentro de layer_image
        layer_image = Image.new(
            plan.dst_region.size,  # type: ignore[union-attr]
            layer_format,
            dtype=edit_layer.image.dtype,
        )
        edit_layer.blend_into(layer_image, edit_image, dst_region)
        return layer_image

    def _flatten_edits(
        self,
        visible_edits: list[EditLayer],
        layer_format: ImageFormat,
        plan: BaseFrame,
        interp: InterpMode,
    ) -> Image:
        """Renderiza múltiplos edits compondo sequencialmente no buffer da camada com scratch buffer."""
        target_dtype = visible_edits[0].image.dtype if visible_edits else np.uint8
        layer_image = Image.new(
            plan.dst_region.size,  # type: ignore[union-attr]
            layer_format,
            dtype=target_dtype,
        )
        for edit_layer in visible_edits:
            scratch = self._scratch_buffer.configure(
                layer_image.size, edit_layer.image.format, dtype=edit_layer.image.dtype
            )
            result = render_edit(edit_layer, plan, interp=interp, dst=scratch)
            if result is None:
                continue
            edit_image, dst_region = result
            edit_layer.blend_into(layer_image, edit_image, dst_region)

        return layer_image

    def render_area(
        self,
        layer: Layer,
        frame: FrameT,
        interp: InterpMode = InterpMode.LANCZOS,
    ) -> Image | None:

        dst_region = frame.dst_region
        if dst_region is None:
            return None

        visible_edits = [edit for edit in layer.edits if edit.visible]

        # Roteamento limpo: 1 edit vs múltiplos edits
        if len(visible_edits) == 1:
            image = self._render_single_edit(
                visible_edits[0], layer.format, frame, interp
            )
            if image is None:
                image = Image.new(dst_region.size, layer.format)
        else:
            image = self._flatten_edits(visible_edits, layer.format, frame, interp)

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

    def render_scene(
        self,
        container: Sequence[BaseLayer] | Container,
        surface: SurfaceProtocol,
        format: ImageFormat = ImageFormat.RGBA,
        interp: InterpMode = InterpMode.LANCZOS,
    ) -> Image:
        with freeze_geometry(container):
            traverser = SceneTraverser(
                self,
                surface,
                self.frame_cls,
                interp=interp,
                target_size=self._target_size,
            )
            images = traverser.traverse(container)

            composition = Image.new(
                surface.size,
                format,
                color=surface.bg_color,
                dtype=surface.dtype,
            )
            return blend_rendered_images(reversed(images), composition)

    def render_patch(
        self,
        container: Sequence[BaseLayer] | Container,
        surface: SurfaceProtocol,
        view_region: Region,
        format: ImageFormat = ImageFormat.RGBA,
        interp: InterpMode = InterpMode.LANCZOS,
    ) -> Image | None:
        if not surface.region.overlaps(view_region):
            return None

        effective_region = surface.region & view_region
        with freeze_geometry(container):
            traverser = SceneTraverser(
                self,
                surface,
                self.frame_cls,
                interp=interp,
                target_size=self._target_size,
            )
            images = traverser.traverse(container, effective_region)

            composition = Image.new(
                effective_region.size,
                format,
                color=surface.bg_color,
                dtype=surface.dtype,
            )
            return blend_rendered_images(reversed(images), composition)


class CanvasRender(BaseRenderer[CanvasFrame]):
    def __init__(self):
        super().__init__(frame_cls=CanvasFrame)

    def render_layer(
        self,
        layer: Layer,
        interp: InterpMode = InterpMode.LANCZOS,
        local: bool = False,
    ) -> Image | None:
        frame = CanvasFrame(layer, Canvas(layer.global_region), local=local)
        return self.render_area(layer, frame, interp=interp)

    def render_container(
        self,
        container: Sequence[BaseLayer] | Container,
        format: ImageFormat = ImageFormat.RGBA,
        interp: InterpMode = InterpMode.LANCZOS,
        bg_color: tuple[int, ...] | None = None,
    ) -> Image | None:
        """Renderiza um contêiner ou sequência de nós (camadas ou grupos) instanciando automaticamente um Canvas
        ajustado à união das regiões globais (global_region) de todos os nós renderizáveis.
        """
        regions = [layer.global_region for layer in container if layer.is_renderable]
        if not regions:
            return None

        roi = reduce(or_, regions)
        canvas = Canvas(roi, bg_color=bg_color)
        return self.render_scene(container, canvas, format=format, interp=interp)


class ViewportRender(BaseRenderer[ViewportFrame]):
    def __init__(self):
        super().__init__(frame_cls=ViewportFrame)
