from __future__ import annotations

import numpy as np
import pytest

import anicrop.render
from anicrop.canvas import Canvas
from anicrop.container import GroupLayer, LayerStack
from anicrop.enums import BlendMode, ImageFormat, InterpMode, WarpMode
from anicrop.frame import CanvasFrame, ViewportFrame
from anicrop.image import Image
from anicrop.layer import EditLayer, Layer
from anicrop.render import (
    CanvasRender,
    SceneTraverser,
    ViewportRender,
    generate_opacity_mask,
    render_edit,
    render_image,
    warp_affine,
    warp_patch,
)
from anicrop.spatial import Region
from anicrop.transform import TransformRel, mat_translation
from anicrop.viewport import Viewport


def make_img(
    w: int = 100,
    h: int = 100,
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
    form: ImageFormat = ImageFormat.RGBA,
) -> Image:
    img_data = np.zeros((h, w, form.channels), dtype=np.uint8)
    img_data[:] = color
    return Image(img_data, form)


def make_solid_image(
    size: tuple[int, int], fmt: ImageFormat, fill_value: int = 255
) -> Image:
    data = np.full((size[1], size[0], fmt.channels), fill_value, dtype=np.uint8)
    return Image(data, fmt)


def make_layer(
    w: int = 100,
    h: int = 100,
    x: int = 0,
    y: int = 0,
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
) -> Layer:
    img = make_img(w, h, color)
    layer = Layer(img)
    if x != 0:
        layer.x = x
    if y != 0:
        layer.y = y
    return layer


def make_checkerboard_image(
    w: int = 20, h: int = 20
) -> tuple[Image, dict[str, list[int]]]:
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
        pytest.param(
            "modo_inexistente", "affine", id="fallback_affine_quando_modo_inexistente"
        ),
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
def test_generate_opacity_mask_formatos_e_preenchimento(
    img_format, fill_value, is_expected_opaque
):
    """Valida se a máscara de oclusão 32x32 identifica corretamente imagens sólidas e transparentes."""
    img = make_solid_image((100, 100), img_format, fill_value=fill_value)
    mask = generate_opacity_mask(
        img, Region.from_size(100, 100), (100, 100), target_size=(32, 32)
    )

    assert mask.shape == (32, 32)
    assert bool(np.all(mask == 255)) is is_expected_opaque


def test_generate_opacity_mask_pixel_com_transparencia_minima():
    """Valida se um único pixel com alpha 254 invalida a oclusão total da máscara 32x32."""
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    data[50, 50, 3] = 254
    img = Image(data, ImageFormat.RGBA)

    mask = generate_opacity_mask(
        img, Region.from_size(100, 100), (100, 100), target_size=(32, 32)
    )

    assert mask.shape == (32, 32)
    assert not np.all(mask == 255)


def test_generate_opacity_mask_spatial_mapping():
    """Valida se a miniatura de oclusão é mapeada proporcionalmente nas coordenadas da matriz 32x32."""
    img = make_solid_image((200, 200), ImageFormat.RGBA, fill_value=255)
    region = Region.from_rect(200, 400, 200, 200)
    viewport_size = (800, 800)

    mask = generate_opacity_mask(
        img, render_region=region, viewport_size=viewport_size, target_size=(32, 32)
    )

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
        pytest.param(
            [(1.0, 0), (1.0, 0)], 2, id="sem_oclusao_todas_camadas_renderizadas"
        ),
        pytest.param(
            [(1.0, 0), (1.0, 255)], 1, id="oclusao_total_pelo_topo_interrompe_abaixo"
        ),
        pytest.param(
            [(1.0, 0), (1.0, 255), (1.0, 0)],
            2,
            id="oclusao_pelo_meio_renderiza_topo_e_meio",
        ),
        pytest.param(
            [(1.0, 0), (0.9, 229)], 2, id="topo_semi_transparente_nao_interrompe"
        ),
    ],
)
def test_render_scene_culling_por_oclusao(
    mocker, layer_configs, expected_rendered_count
):
    """Valida se o SceneTraverser realiza early-exit conservador ao atingir 100% de oclusão."""
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
        pytest.param(
            (0, 0, 100, 100),
            True,
            (80, 40, 20, 20),
            (5, 5),
            "yellow",
            id="local_sem_recorte",
        ),
        pytest.param(
            (40, 0, 20, 10), True, (0, 0, 10, 20), (5, 5), "red", id="local_com_recorte"
        ),
        pytest.param(
            (0, 0, 100, 100),
            False,
            (40, 0, 20, 20),
            (5, 5),
            "red",
            id="global_sem_recorte",
        ),
        pytest.param(
            (40, 0, 20, 10),
            False,
            (0, 0, 20, 10),
            (5, 5),
            "red",
            id="global_com_recorte",
        ),
    ],
)
def test_render_edit_canvas_frame_projecoes_e_recortes(
    canvas_rect, is_local, expected_dest_rect, sample_point, expected_color_name
):
    """Valida a projeção e o recorte espacial de um EditLayer sob coordenadas locais e globais do CanvasFrame."""
    layer = Layer(Image.new((100, 100), ImageFormat.RGBA))
    layer.set_transform(TransformRel().rotate(-90))

    img_hat, colors = make_checkerboard_image(20, 20)
    layer.add_edit(img_hat, Region.from_rect(40, 0, 20, 20))

    canvas = Canvas.from_rect(*canvas_rect)
    frame = CanvasFrame(layer, canvas, local=is_local)

    result = render_edit(layer._edits[1], plan=frame, interp=InterpMode.NEAREST)
    assert result is not None

    warped_image, dest_region = result
    assert dest_region == Region.from_rect(*expected_dest_rect)
    np.testing.assert_array_equal(
        warped_image[sample_point[1], sample_point[0]], colors[expected_color_name]
    )


def test_render_edit_com_viewport_frame():
    """Valida se o render_edit projeta o EditLayer corretamente dentro do ViewportFrame."""
    layer = Layer(Image.new((100, 100), ImageFormat.RGBA))
    layer.set_transform(TransformRel().rotate(-90))

    img_hat, _ = make_checkerboard_image(20, 20)
    layer.add_edit(img_hat, Region.from_rect(40, 0, 20, 20))

    viewport = Viewport((800, 600), 1.0)
    frame = ViewportFrame(layer, viewport)

    result = render_edit(layer._edits[1], plan=frame, interp=InterpMode.NEAREST)
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

    result = render_image(img, frame, m_local, interp=InterpMode.NEAREST)
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
    item = make_layer(10, 10) if item_cls is Layer else GroupLayer()
    item.visible = False

    group = GroupLayer()
    group.append(item)

    renderer = CanvasRender()
    canvas = Canvas.from_size(10, 10)
    traverser = SceneTraverser(renderer, canvas, CanvasFrame)
    result = traverser.traverse([group])

    assert result == []


def test_scene_traverser_ignora_tudo_se_raiz_for_invisivel():
    """Valida se uma árvore inteira é sumariamente descartada quando o grupo raiz for invisível."""
    root = GroupLayer()
    root.visible = False

    child = GroupLayer()
    child.visible = True
    child_layer = make_layer(10, 10)
    child.append(child_layer)
    root.append(child)

    renderer = CanvasRender()
    canvas = Canvas.from_size(10, 10)
    traverser = SceneTraverser(renderer, canvas, CanvasFrame)
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
        pytest.param(
            (-40, -30, 100, 100),
            (500, 500),
            (60, 70),
            id="recorte_topo_esquerdo_coords_negativas",
        ),
        pytest.param(
            (450, 470, 100, 100),
            (500, 500),
            (50, 30),
            id="recorte_base_direita_limite_canvas",
        ),
    ],
)
def test_canvas_render_area_recorte_parcial_retorna_dimensao_exata(
    layer_rect, canvas_size, expected_size
):
    """Valida se CanvasRender.render_area recorta e retorna o retalho com a dimensão visível exata."""
    canvas = Canvas.from_size(*canvas_size)
    layer = make_layer(
        w=layer_rect[2], h=layer_rect[3], x=layer_rect[0], y=layer_rect[1]
    )
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
    renderer = CanvasRender()
    spy_render = mocker.spy(renderer, "render_area")
    _ = renderer.render_scene(stack, canvas)

    assert spy_render.call_count == 1
    assert spy_render.call_args[0][0] == layer_in


def test_scene_traverser_pre_culling_respeita_view_region_em_render_patch(mocker):
    """Valida se o SceneTraverser filtra por view_region durante render_patch."""
    stack = LayerStack()
    layer_in_canvas_but_out_of_patch = make_layer(w=20, h=20, x=10, y=10)
    layer_in_patch = make_layer(w=20, h=20, x=80, y=80)
    stack.append(layer_in_canvas_but_out_of_patch)
    stack.append(layer_in_patch)
    canvas = Canvas.from_size(200, 200)
    patch_region = Region.from_rect(70, 70, 50, 50)

    renderer = CanvasRender()
    spy_render = mocker.spy(renderer, "render_area")
    _ = renderer.render_patch(stack, canvas, patch_region)

    assert spy_render.call_count == 1
    assert spy_render.call_args[0][0] == layer_in_patch


def test_scene_traverser_culling_camada_com_opacidade_zero(mocker):
    """Valida se o SceneTraverser descarta camadas com opacidade zero sem chamar render_area."""
    stack = LayerStack()
    layer_visible = make_layer(w=20, h=20, x=10, y=10)
    layer_invisible = make_layer(w=20, h=20, x=40, y=40)
    layer_invisible.opacity = 0.0
    stack.append(layer_visible)
    stack.append(layer_invisible)
    canvas = Canvas.from_size(100, 100)

    renderer = CanvasRender()
    spy_render = mocker.spy(renderer, "render_area")
    _ = renderer.render_scene(stack, canvas)

    assert spy_render.call_count == 1
    assert spy_render.call_args[0][0] == layer_visible


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
    renderer._scratch_buffer.configure((100, 100), ImageFormat.RGBA)
    _ = renderer._scratch_buffer[Region.from_size(100, 100)]
    buf1 = renderer._scratch_buffer._image
    assert buf1 is not None
    assert buf1.width >= 100
    assert buf1.height >= 100

    # 2. Segunda requisição menor (50x50) deve reutilizar o mesmo array base
    renderer._scratch_buffer.configure((50, 50), ImageFormat.RGBA)
    _ = renderer._scratch_buffer[Region.from_size(50, 50)]
    buf2 = renderer._scratch_buffer._image
    assert buf2 is buf1

    # 3. Terceira requisição maior (300x300) deve expandir o buffer
    renderer._scratch_buffer.configure((300, 300), ImageFormat.RGBA)
    _ = renderer._scratch_buffer[Region.from_size(300, 300)]
    buf3 = renderer._scratch_buffer._image
    assert buf3 is not buf1


def test_render_single_edit_full_frame_returns_direct_image():
    """Valida se _render_single_edit retorna a imagem diretamente quando cobre 100% da área do frame."""
    layer = make_layer(w=100, h=80, color=(255, 0, 0, 255))
    renderer = CanvasRender()
    frame = CanvasFrame(layer, Canvas.from_size(100, 80))

    result = renderer._render_single_edit(
        layer.edits[0], layer.format, frame, InterpMode.LANCZOS
    )

    assert result is not None
    assert result.size == (100, 80)
    assert np.all(result[...][:, :, 0] == 255)


def test_render_single_edit_partial_patch_blends_into_layer_image():
    """Valida se _render_single_edit compõe no layer_image quando o patch do edit é menor que a camada."""
    layer = make_layer(w=100, h=100, color=(0, 0, 0, 0))
    patch_img = Image(
        np.full((40, 40, 4), (0, 255, 0, 255), dtype=np.uint8), ImageFormat.RGBA
    )
    patch_edit = EditLayer(
        patch_img, Region.from_rect(30, 30, 40, 40), np.identity(3, dtype=np.float32)
    )

    renderer = CanvasRender()
    frame = CanvasFrame(layer, Canvas.from_size(100, 100))

    result = renderer._render_single_edit(
        patch_edit, layer.format, frame, InterpMode.LANCZOS
    )

    assert result is not None
    assert result.size == (100, 100)
    assert np.all(result[30:70, 30:70, 1] == 255)
    assert np.all(result[0:30, 0:30, 3] == 0)


def test_render_area_out_of_bounds_returns_none():
    """Valida se render_area retorna None quando o layer está fora da área visível do frame."""
    layer = make_layer(w=100, h=100, color=(255, 0, 0, 255))
    layer.transform.translate(500, 500)

    renderer = CanvasRender()
    canvas = Canvas.from_size(100, 100)
    frame = CanvasFrame(layer, canvas)

    result = renderer.render_area(layer, frame, InterpMode.LANCZOS)

    assert result is None


@pytest.mark.parametrize(
    "fmt, color",
    [
        pytest.param(ImageFormat.RGB, (200, 100, 50), id="format_rgb"),
        pytest.param(ImageFormat.RGBA, (10, 20, 30, 255), id="format_rgba"),
        pytest.param(ImageFormat.GRAY, (128,), id="format_gray"),
    ],
)
def test_render_single_edit_preserves_image_format(fmt, color):
    """Valida se _render_single_edit preserva o formato de cor correto da camada."""
    data = np.zeros((40, 60, len(color)), dtype=np.uint8)
    data[:] = color
    layer = Layer(Image(data, fmt))
    renderer = CanvasRender()
    frame = CanvasFrame(layer, Canvas.from_size(60, 40))

    result = renderer._render_single_edit(
        layer.edits[0], layer.format, frame, InterpMode.LANCZOS
    )

    assert result is not None
    assert result.format == fmt
    assert result.size == (60, 40)


def test_render_container_combina_camadas_lado_a_lado():
    """Valida se render_container calcula o Canvas automaticamente pela uniao das global_regions."""
    l1 = make_layer(w=50, h=50, color=(255, 0, 0, 255))
    l2 = make_layer(w=50, h=50, color=(0, 0, 255, 255))
    l2.transform.translate(50, 0)

    renderer = CanvasRender()
    result = renderer.render_container([l1, l2], format=ImageFormat.RGBA)

    assert result is not None
    assert result.size == (100, 50)
    assert np.array_equal(result[0, 0], [255, 0, 0, 255])
    assert np.array_equal(result[0, 50], [0, 0, 255, 255])


def test_render_container_respeita_ordem_de_sobreposicao():
    """Valida se a ordem da lista determina qual camada fica no topo na composicao."""
    l1 = make_layer(w=50, h=50, color=(255, 0, 0, 255))
    l2 = make_layer(w=50, h=50, color=(0, 255, 0, 255))

    renderer = CanvasRender()
    result_over = renderer.render_container([l1, l2], format=ImageFormat.RGBA)
    result_under = renderer.render_container([l2, l1], format=ImageFormat.RGBA)

    assert result_over is not None and result_under is not None
    assert np.array_equal(result_over[0, 0], [0, 255, 0, 255])
    assert np.array_equal(result_under[0, 0], [255, 0, 0, 255])


def test_render_container_com_sequencia_vazia_retorna_none():
    """Valida se render_container retorna None quando recebe uma sequencia sem camadas."""
    renderer = CanvasRender()
    result = renderer.render_container([])

    assert result is None


def test_render_container_com_camada_invisivel_retorna_none():
    """Valida se render_container retorna None quando todas as camadas sao invisiveis."""
    l1 = make_layer(w=50, h=50)
    l1.visible = False

    renderer = CanvasRender()
    result = renderer.render_container([l1])

    assert result is None


def test_render_container_com_bg_color():
    """Valida se render_container aplica o bg_color no canvas gerado automaticamente."""
    l1 = make_layer(w=50, h=50, color=(255, 0, 0, 255))
    l1.transform.translate(50, 50)

    renderer = CanvasRender()
    result = renderer.render_container(
        [l1], format=ImageFormat.RGBA, bg_color=(0, 255, 0, 255)
    )

    assert result is not None
    assert result.size == (50, 50)
    assert np.array_equal(result[0, 0], [255, 0, 0, 255])


def test_render_single_edit_com_distorcao_nao_retorna_referencia_ao_scratch_buffer():
    """Valida se _render_single_edit com distorcao retorna uma imagem isolada e nao o scratch buffer compartilhado."""
    layer = make_layer(w=60, h=60, color=(255, 0, 0, 255))
    layer.transform.rotate(45)

    renderer = CanvasRender()
    frame = CanvasFrame(layer, Canvas(layer.global_region))

    rendered = renderer.render_area(layer, frame)

    assert rendered is not None
    scratch_allocated = renderer._scratch_buffer._ensure_allocated()
    assert not np.shares_memory(rendered[...], scratch_allocated[...])


def test_warp_patch_com_padding_nas_quatro_bordas_preenche_transparencia():
    """Valida se warp_patch com Lanczos sob rotacao preenche bordas externas com transparencia via ScratchBuffer."""
    img = make_img(w=40, h=40, color=(255, 0, 0, 255))
    layer = Layer(img)
    layer.transform.rotate(45)
    dst_region = layer.global_region

    result = warp_patch(
        img,
        layer.transform.matrix,
        dst_region,
        interp=InterpMode.LANCZOS,
    )

    assert result is not None
    assert result.shape[0] == int(round(dst_region.height))
    assert result.shape[1] == int(round(dst_region.width))
    assert result[0, 0, 3] == 0


def test_warp_patch_recorte_parcial_posiciona_pixels_corretamente_no_scratch():
    """Valida se warp_patch posiciona pixels reais no offset correto quando a regiao alvo extrapola a imagem."""
    img = make_img(w=50, h=50, color=(0, 255, 0, 255))
    m_global = mat_translation(10, 10)
    dst_region = Region.from_size(20, 20)

    result = warp_patch(
        img,
        m_global,
        dst_region,
        interp=InterpMode.LANCZOS,
    )

    assert result is not None
    assert result.shape == (20, 20, 4)
    np.testing.assert_array_equal(result[10, 10], [0, 255, 0, 255])
