from __future__ import annotations

import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.container import GroupLayer, LayerStack


from anicrop.enums import BlendMode, ImageFormat, InterpolationOption, WarpMode
from anicrop.frame import CanvasFrame, ViewportFrame
from anicrop.image import Image
from anicrop.layer import Layer
import anicrop.render
from anicrop.render import (
    CanvasRender,
    SceneTraverser,
    ViewportRender,
    generate_opacity_mask,
    render_edit,
    render_image,
    warp_affine,
    warp_patch,
    warp_perspective,
)

from anicrop.spatial import Region, Span
from anicrop.transform import TransformRel
from anicrop.viewport import Viewport


def make_img(w: int = 100, h: int = 100, color: tuple[int, int, int, int] = (255, 0, 0, 255), form: ImageFormat = ImageFormat.RGBA) -> Image:
    img_data = np.zeros((h, w, form.channels), dtype=np.uint8)
    img_data[:] = color
    return Image(img_data, form)


def make_solid_image(size: tuple[int, int], fmt: ImageFormat, fill_value: int = 255) -> Image:
    data = np.full((size[1], size[0], fmt.channels), fill_value, dtype=np.uint8)
    return Image(data, fmt)


def make_layer(w: int = 100, h: int = 100, x: int = 0, y: int = 0, color: tuple[int, int, int, int] = (255, 0, 0, 255)) -> Layer:
    img = make_img(w, h, color)
    layer = Layer(img)
    if x != 0:
        layer.x = x
    if y != 0:
        layer.y = y
    return layer


def make_checkerboard_image(w: int = 20, h: int = 20) -> tuple[Image, dict[str, list[int]]]:
    colors = {
        "red": [255, 0, 0, 255],
        "blue": [0, 0, 255, 255],
        "yellow": [255, 255, 0, 255],
        "green": [0, 255, 0, 255],
    }
    data = np.zeros((h, w, 4), dtype=np.uint8)
    hw, hh = w // 2, h // 2
    data[0:hh, 0:hw] = colors["red"]
    data[0:hh, hw:w] = colors["blue"]
    data[hh:h, 0:hw] = colors["yellow"]
    data[hh:h, hw:w] = colors["green"]
    return Image(data, ImageFormat.RGBA), colors


# ==============================================================================
# Motor de Projeção (Warp Dispatch e Fallback)
# ==============================================================================

@pytest.mark.parametrize(
    "warp_mode, expected_target",
    [
        pytest.param(WarpMode.AFFINE, "affine", id="dispatch_affine"),
        pytest.param(WarpMode.PERSPECTIVE, "perspective", id="dispatch_perspective"),
        pytest.param("modo_inexistente", "affine", id="fallback_affine_quando_modo_inexistente"),
    ],
)
def test_warp_patch_dispatch_and_fallback(mocker, warp_mode, expected_target):
    """Valida o despacho correto do WarpMode com fallback para warp_affine em modos não mapeados."""
    mocks = {
        "affine": mocker.patch("anicrop.render.warp_affine"),
        "perspective": mocker.MagicMock(),
    }
    mocker.patch.dict(
        "anicrop.render.WARP_MODE",
        {WarpMode.AFFINE: mocks["affine"], WarpMode.PERSPECTIVE: mocks["perspective"]},
        clear=True,
    )

    img = make_img(10, 10)
    warp_patch(img, np.eye(3), Region.from_size(10, 10), warp_mode=warp_mode)

    assert mocks[expected_target].called


# ==============================================================================
# Geração da Máscara de Oclusão (generate_opacity_mask)
# ==============================================================================

@pytest.mark.parametrize(
    "img_format, fill_value, is_expected_opaque",
    [
        pytest.param(ImageFormat.RGBA, 255, True, id="rgba_totalmente_opaco"),
        pytest.param(ImageFormat.RGBA, 0, False, id="rgba_totalmente_transparente"),
        pytest.param(ImageFormat.RGB, 255, True, id="rgb_sem_alpha_sempre_opaco"),
    ],
)
def test_generate_opacity_mask_formatos_e_preenchimento(img_format, fill_value, is_expected_opaque):
    """Valida se a máscara de oclusão 32x32 identifica corretamente imagens sólidas e transparentes."""
    img = make_solid_image((100, 100), img_format, fill_value=fill_value)
    mask = generate_opacity_mask(img, Region.from_size(100, 100), (100, 100), target_size=(32, 32))

    assert mask.shape == (32, 32)
    assert bool(np.all(mask == 255)) is is_expected_opaque


def test_generate_opacity_mask_pixel_com_transparencia_minima():
    """Valida se um único pixel com alpha 254 invalida a oclusão total da máscara 32x32."""
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    data[50, 50, 3] = 254
    img = Image(data, ImageFormat.RGBA)

    mask = generate_opacity_mask(img, Region.from_size(100, 100), (100, 100), target_size=(32, 32))

    assert mask.shape == (32, 32)
    assert not np.all(mask == 255)


def test_generate_opacity_mask_spatial_mapping():
    """Valida se a miniatura de oclusão é mapeada proporcionalmente nas coordenadas da matriz 32x32."""
    img = make_solid_image((200, 200), ImageFormat.RGBA, fill_value=255)
    region = Region.from_rect(200, 400, 200, 200)
    viewport_size = (800, 800)

    mask = generate_opacity_mask(img, render_region=region, viewport_size=viewport_size, target_size=(32, 32))

    expected_mask = np.zeros((32, 32), dtype=np.uint8)
    expected_mask[16:24, 8:16] = 255

    assert mask.shape == (32, 32)
    np.testing.assert_array_equal(mask, expected_mask)


# ==============================================================================
# Oclusão e Early-Exit na Cena (SceneTraverser / render_scene)
# ==============================================================================

@pytest.mark.parametrize(
    "layer_configs, expected_rendered_count",
    [
        pytest.param([(1.0, 0), (1.0, 0)], 2, id="sem_oclusao_todas_camadas_renderizadas"),
        pytest.param([(1.0, 255), (1.0, 0)], 1, id="oclusao_total_pelo_topo_interrompe_abaixo"),
        pytest.param([(1.0, 0), (1.0, 255), (1.0, 0)], 2, id="oclusao_pelo_meio_renderiza_topo_e_meio"),
        pytest.param([(0.9, 229), (1.0, 0)], 2, id="topo_semi_transparente_nao_interrompe"),
    ],
)
def test_render_scene_culling_por_oclusao(mocker, layer_configs, expected_rendered_count):
    """Valida se o SceneTraverser realiza early-exit conservador ao atingir 100% de oclusão."""
    mocker.patch("anicrop.render.BLEND_MODE")
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()

    layers = [make_layer(w=800, h=600) for _ in layer_configs]
    for layer, (opacity, _) in zip(layers, layer_configs):
        layer.opacity = opacity
        layer.blend_mode = BlendMode.NORMAL

    rendered = []

    def mock_render(layer, *args, **kwargs):
        rendered.append(layer)
        idx = layers.index(layer)
        layer._opacity_mask = np.full((32, 32), layer_configs[idx][1], dtype=np.uint8)
        return make_img(w=800, h=600)

    mocker.patch.object(vr, "render_area", side_effect=mock_render)
    vr.render_scene(layers, viewport)

    assert len(rendered) == expected_rendered_count


# ==============================================================================
# Projeções e Recortes de Edits (render_edit e render_image)
# ==============================================================================

@pytest.mark.parametrize(
    "canvas_rect, is_local, expected_dest_rect, sample_point, expected_color_name",
    [
        pytest.param((0, 0, 100, 100), True, (80, 40, 20, 20), (5, 5), "yellow", id="local_sem_recorte"),
        pytest.param((40, 0, 20, 10), True, (0, 0, 10, 20), (5, 5), "red", id="local_com_recorte"),
        pytest.param((0, 0, 100, 100), False, (40, 0, 20, 20), (5, 5), "red", id="global_sem_recorte"),
        pytest.param((40, 0, 20, 10), False, (0, 0, 20, 10), (5, 5), "red", id="global_com_recorte"),
    ],
)
def test_render_edit_canvas_frame_projecoes_e_recortes(canvas_rect, is_local, expected_dest_rect, sample_point, expected_color_name):
    """Valida a projeção e o recorte espacial de um EditLayer sob coordenadas locais e globais do CanvasFrame."""
    layer = Layer(Image.new((100, 100), ImageFormat.RGBA))
    layer.set_transform(TransformRel().rotate(-90))

    img_hat, colors = make_checkerboard_image(20, 20)
    layer.add_edit(img_hat, Region.from_rect(40, 0, 20, 20))

    canvas = Canvas.from_rect(*canvas_rect)
    frame = CanvasFrame(layer, canvas, local=is_local)

    result = render_edit(layer._edits[1], plan=frame, interp=InterpolationOption.NEAREST)
    assert result is not None

    warped_image, dest_region = result
    assert dest_region == Region.from_rect(*expected_dest_rect)
    np.testing.assert_array_equal(warped_image[sample_point[1], sample_point[0]], colors[expected_color_name])


def test_render_edit_com_viewport_frame():
    """Valida se o render_edit projeta o EditLayer corretamente dentro do ViewportFrame."""
    layer = Layer(Image.new((100, 100), ImageFormat.RGBA))
    layer.set_transform(TransformRel().rotate(-90))

    img_hat, _ = make_checkerboard_image(20, 20)
    layer.add_edit(img_hat, Region.from_rect(40, 0, 20, 20))

    viewport = Viewport((800, 600), 1.0)
    frame = ViewportFrame(layer, viewport)

    result = render_edit(layer._edits[1], plan=frame, interp=InterpolationOption.NEAREST)
    assert result is not None

    warped_image, dest_region = result
    assert dest_region == Region.from_rect(40, 0, 20, 20)


def test_render_image_direto():
    """Valida a projeção atômica de uma Image arbitrária através de render_image com CanvasFrame."""
    img = make_img(w=50, h=50, color=(0, 255, 0, 255))
    layer = make_layer(w=50, h=50, color=(0, 255, 0, 255))
    canvas = Canvas.from_size(50, 50)
    frame = CanvasFrame(layer, canvas)
    m_local = np.identity(3, dtype=np.float32)

    result = render_image(img, frame, m_local, interp=InterpolationOption.NEAREST)
    assert result is not None

    warped_image, dest_region = result
    assert warped_image.size == (50, 50)
    assert dest_region == Region.from_size(50, 50)


# ==============================================================================
# Travessia e Culling de Hierarquia (SceneTraverser)
# ==============================================================================

def test_scene_traverser_recursivo_com_culling(mocker):
    """Valida a travessia recursiva de SceneTraverser em grupos e a interrupção por oclusão."""
    group_raiz = GroupLayer()
    sub_grupo = GroupLayer()

    layer1 = make_layer(100, 100)
    layer2 = make_layer(100, 100)
    layer1._opacity_mask = np.full((32, 32), 255, dtype=np.uint8)
    layer2._opacity_mask = np.full((32, 32), 255, dtype=np.uint8)

    sub_grupo.append(layer1)
    group_raiz.append(sub_grupo)
    group_raiz.append(layer2)

    mock_renderer = mocker.MagicMock()
    mock_renderer.render_area.return_value = Image.new((100, 100), ImageFormat.RGBA)

    mock_frame = mocker.MagicMock()
    mock_frame.dst_region = Region.from_size(100, 100)
    mock_frame.targ_region = Region.from_size(100, 100)
    mock_frame_cls = mocker.MagicMock(return_value=mock_frame)

    mock_surface = Canvas.from_size(100, 100)

    traverser = SceneTraverser(mock_renderer, mock_surface, mock_frame_cls)
    images_gp = traverser.traverse([group_raiz])

    assert len(images_gp) == 1
    assert images_gp[0][0] == group_raiz
    assert mock_renderer.render_area.call_count == 1
    assert np.all(traverser.miniview == 255)


@pytest.mark.parametrize("item_cls", [Layer, GroupLayer])
def test_scene_traverser_ignora_itens_invisiveis(mocker, item_cls):
    """Valida se o SceneTraverser ignora camadas ou grupos que possuem visible=False."""
    item = mocker.MagicMock(spec=item_cls)
    type(item).visible = mocker.PropertyMock(return_value=False)
    item.parent = mocker.Mock()

    group = GroupLayer()
    group.append(item)

    traverser = SceneTraverser(mocker.Mock(), Canvas.from_size(10, 10), mocker.Mock())
    result = traverser.traverse([group])

    assert result == []


def test_scene_traverser_ignora_tudo_se_raiz_for_invisivel(mocker):
    """Valida se uma árvore inteira é sumariamente descartada quando o grupo raiz for invisível."""
    root = GroupLayer()
    root.visible = False

    child = mocker.MagicMock(spec=GroupLayer)
    type(child).visible = mocker.PropertyMock(return_value=True)
    child.parent = mocker.Mock()
    root.append(child)

    traverser = SceneTraverser(mocker.Mock(), Canvas.from_size(10, 10), mocker.Mock())
    result = traverser.traverse([root])

    assert result == []


# ==============================================================================
# Casos de Borda: Culling Total e Recortes de Borda no Renderizador
# ==============================================================================

def test_canvas_render_area_culling_total_retorna_none(mocker):
    """Valida se CanvasRender.render_area descarta camadas fora do Canvas retornando None sem invocar warp_patch."""
    spy_patch = mocker.spy(anicrop.render, "warp_patch")
    canvas = Canvas.from_size(500, 500)
    layer = make_layer(w=100, h=100, x=1000, y=1000)
    frame = CanvasFrame(layer, canvas)

    renderer = CanvasRender()
    result = renderer.render_area(layer, frame)

    assert result is None
    assert not spy_patch.called


@pytest.mark.parametrize(
    "layer_rect, canvas_size, expected_size",
    [
        pytest.param((-40, -30, 100, 100), (500, 500), (60, 70), id="recorte_topo_esquerdo_coords_negativas"),
        pytest.param((450, 470, 100, 100), (500, 500), (50, 30), id="recorte_base_direita_limite_canvas"),
    ],
)
def test_canvas_render_area_recorte_parcial_retorna_dimensao_exata(layer_rect, canvas_size, expected_size):
    """Valida se CanvasRender.render_area recorta e retorna o retalho com a dimensão visível exata."""
    canvas = Canvas.from_size(*canvas_size)
    layer = make_layer(w=layer_rect[2], h=layer_rect[3], x=layer_rect[0], y=layer_rect[1])
    frame = CanvasFrame(layer, canvas)

    renderer = CanvasRender()
    result = renderer.render_area(layer, frame)

    assert result is not None
    assert result.size == expected_size


# ==============================================================================
# Casos de Borda: GroupLayer Vazio e Filhos Fora da Tela (SceneTraverser)
# ==============================================================================

def test_scene_traverser_grupo_vazio_retorna_lista_vazia():
    """Valida se o SceneTraverser descarta imediatamente um GroupLayer sem camadas filhas."""
    group = GroupLayer()
    traverser = SceneTraverser(CanvasRender(), Canvas.from_size(500, 500), CanvasFrame)

    result = traverser.traverse([group])

    assert result == []


def test_scene_traverser_grupo_com_todos_filhos_fora_da_tela_retorna_lista_vazia():
    """Valida se o SceneTraverser descarta o GroupLayer quando todos os seus filhos sofrem culling total."""
    group = GroupLayer()
    group.append(make_layer(100, 100, x=1000, y=1000))
    group.append(make_layer(100, 100, x=-500, y=-500))

    traverser = SceneTraverser(CanvasRender(), Canvas.from_size(500, 500), CanvasFrame)

    result = traverser.traverse([group])

    assert result == []


# ==============================================================================
# Otimização: Fast-Path para Translação Pura (Bypass de warp_patch)
# ==============================================================================

@pytest.mark.parametrize(
    "tx, ty",
    [
        pytest.param(10, 20, id="transladado_inteiro_ativa_fast_path"),
        pytest.param(0, 0, id="identidade_ativa_fast_path"),
    ],
)
def test_render_image_fast_path_translacao_pura(mocker, tx, ty):
    """Valida se render_image realiza bypass completo de warp_patch em matrizes de translação pura."""
    spy_patch = mocker.spy(anicrop.render, "warp_patch")
    img = make_img(w=100, h=100, color=(255, 0, 0, 255))
    layer = make_layer(w=100, h=100, x=tx, y=ty, color=(255, 0, 0, 255))
    canvas = Canvas.from_size(200, 200)
    frame = CanvasFrame(layer, canvas)
    m_local = np.identity(3, dtype=np.float32)

    result = render_image(img, frame, m_local)

    assert result is not None
    warped_image, dest_region = result
    assert dest_region == Region.from_size(100, 100)
    assert not spy_patch.called
    np.testing.assert_array_equal(warped_image[0, 0], [255, 0, 0, 255])


def test_render_image_rotacao_nao_ativa_fast_path(mocker):
    """Valida se render_image continua delegando para warp_patch quando houver rotação ou escala."""
    spy_patch = mocker.spy(anicrop.render, "warp_patch")
    img = make_img(w=100, h=100, color=(255, 0, 0, 255))
    layer = make_layer(w=100, h=100, color=(255, 0, 0, 255))
    layer.transform.rotate(45)
    canvas = Canvas.from_size(200, 200)
    frame = CanvasFrame(layer, canvas)
    m_local = np.identity(3, dtype=np.float32)

    result = render_image(img, frame, m_local)

    assert result is not None
    assert spy_patch.called


def test_canvas_render_area_translacao_pura_com_mock_blend(mocker):
    """Valida se CanvasRender.render_area executa o fast-path sem invocar warp_patch nem onerar blend."""
    spy_patch = mocker.spy(anicrop.render, "warp_patch")
    mocker.patch.dict("anicrop.render.BLEND_MODE", {BlendMode.NORMAL: mocker.MagicMock()})

    layer = make_layer(w=100, h=100, x=10, y=10)
    canvas = Canvas.from_size(200, 200)
    frame = CanvasFrame(layer, canvas)

    renderer = CanvasRender()
    result = renderer.render_area(layer, frame)

    assert result is not None
    assert not spy_patch.called


def test_render_image_fast_path_recorte_negativo_preserva_pixels_corretos():
    """Valida se o fast-path com coordenadas negativas fatia a sub-região correta da imagem fonte."""
    data = np.zeros((100, 100, 4), dtype=np.uint8)
    data[:50, :] = [255, 0, 0, 255]
    data[50:, :] = [0, 0, 255, 255]
    img = Image(data, ImageFormat.RGBA)

    layer = Layer(img)
    layer.region += (0, -50)
    canvas = Canvas.from_size(100, 100)
    frame = CanvasFrame(layer, canvas)
    m_local = np.identity(3, dtype=np.float32)

    result = render_image(img, frame, m_local)

    assert result is not None
    warped_image, dest_region = result
    assert dest_region == Region.from_size(100, 50)
    assert warped_image.size == (100, 50)
    np.testing.assert_array_equal(warped_image[0, 0], [0, 0, 255, 255])


def test_scene_traverser_pre_culling_ignora_camadas_e_grupos_fora_da_superficie(mocker):
    """Valida se o SceneTraverser descarta camadas e grupos fora da superfície sem instanciar frames."""
    stack = LayerStack()
    layer_in = make_layer(w=50, h=50, x=10, y=10)
    layer_out = make_layer(w=50, h=50, x=500, y=500)
    group_out = GroupLayer()
    group_child = make_layer(w=50, h=50, x=1000, y=1000)
    group_out.append(group_child)
    stack.append(layer_in)
    stack.append(layer_out)
    stack.append(group_out)
    canvas = Canvas.from_size(100, 100)
    spy_frame = mocker.spy(CanvasFrame, "__init__")

    renderer = CanvasRender()
    _ = renderer.render_scene(stack, canvas)

    assert spy_frame.call_count == 1
    assert spy_frame.call_args[0][1] == layer_in


def test_scene_traverser_pre_culling_respeita_view_region_em_render_patch(mocker):
    """Valida se o SceneTraverser filtra por view_region durante render_patch."""
    stack = LayerStack()
    layer_in_canvas_but_out_of_patch = make_layer(w=20, h=20, x=10, y=10)
    layer_in_patch = make_layer(w=20, h=20, x=80, y=80)
    stack.append(layer_in_canvas_but_out_of_patch)
    stack.append(layer_in_patch)
    canvas = Canvas.from_size(200, 200)
    patch_region = Region.from_rect(70, 70, 50, 50)
    spy_frame = mocker.spy(CanvasFrame, "__init__")

    renderer = CanvasRender()
    _ = renderer.render_patch(stack, canvas, patch_region)

    assert spy_frame.call_count == 1
    assert spy_frame.call_args[0][1] == layer_in_patch


def test_warp_affine_com_parametro_dst_reutiliza_buffer_prealocado():
    """Valida se warp_affine escreve diretamente no buffer pré-alocado passado em dst."""
    src_data = np.full((50, 50, 4), 255, dtype=np.uint8)
    m_cv2 = np.identity(3, dtype=np.float64)
    dst_buffer = np.zeros((50, 50, 4), dtype=np.uint8)

    result = warp_affine(src_data, m_cv2, (50, 50), dst=dst_buffer)

    assert result is dst_buffer
    np.testing.assert_array_equal(result[0, 0], [255, 255, 255, 255])


def test_warp_patch_com_parametro_dst_escreve_diretamente_no_buffer():
    """Valida se warp_patch reaproveita o array de destino passado em dst."""
    src_img = make_img(w=50, h=50, color=(255, 0, 0, 255))
    m_global = np.identity(3, dtype=np.float32)
    dst_region = Region.from_size(50, 50)
    dst_buffer = np.zeros((50, 50, 4), dtype=np.uint8)

    result = warp_patch(src_img, m_global, dst_region, dst=dst_buffer)

    assert result is dst_buffer
    np.testing.assert_array_equal(result[0, 0], [255, 0, 0, 255])


def test_renderer_scratch_buffer_reutiliza_memoria_e_expande_conforme_necessidade():
    """Valida se o scratch buffer reaproveita fatias do mesmo array e expande sob demanda."""
    renderer = CanvasRender()

    # 1. Primeira alocação (100x100)
    buf1 = renderer._get_scratch_buffer(100, 100, ImageFormat.RGBA)
    assert buf1.size == (100, 100)

    # 2. Segunda requisição menor (50x50) deve reutilizar o mesmo array base
    buf2 = renderer._get_scratch_buffer(50, 50, ImageFormat.RGBA)
    assert buf2.size == (50, 50)
    assert buf2._data.base is buf1._data.base

    # 3. Terceira requisição maior (300x300) deve expandir o buffer
    buf3 = renderer._get_scratch_buffer(300, 300, ImageFormat.RGBA)
    assert buf3.size == (300, 300)
    assert buf3._data.base is not buf1._data.base
