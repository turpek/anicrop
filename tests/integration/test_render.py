from __future__ import annotations

import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.container import GroupLayer
from anicrop.document import Document
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.render import BaseRenderer, CanvasRender, ViewportRender
from anicrop.spatial import Region
from anicrop.transform import mat_final
from anicrop.viewport import Viewport


@pytest.fixture
def canvas_render():
    """Retorna uma instância de CanvasRender para os testes de integração."""
    return CanvasRender()


@pytest.fixture
def viewport_render():
    """Retorna uma instância de ViewportRender para os testes de integração."""
    return ViewportRender()


def make_img(
    w: int = 100,
    h: int = 100,
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
    form: ImageFormat = ImageFormat.RGBA,
) -> Image:
    img_data = np.zeros((h, w, 4), dtype=np.uint8)
    img_data[:] = color
    return Image(img_data, form)


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


def test_canvas_render_identidade_sem_transformacao(canvas_render):
    """Testa se um Layer sem transformações é renderizado exatamente igual à imagem original."""
    original_layer = make_layer(w=50, h=50, color=(100, 150, 200, 255))

    rendered_image = canvas_render.render_layer(original_layer)

    assert rendered_image.size == (50, 50)
    np.testing.assert_array_equal(
        rendered_image[...], original_layer.edits[0].image[...]
    )


def test_canvas_render_rotacao_expansao_segura(canvas_render):
    """Testa se o CanvasRender expande a imagem corretamente ao rotacionar em 45 graus."""
    cor_original = (255, 0, 0, 255)
    layer = make_layer(w=100, h=100, color=cor_original)
    layer.transform.rotate(45)

    rendered_image = canvas_render.render_layer(layer)
    rect = layer.global_region

    assert rendered_image.size == rect.size.to_int()

    img_array = rendered_image[...]
    centro_x, centro_y = int(rect.width // 2), int(rect.height // 2)

    assert img_array[centro_y, centro_x, 3] == 255
    np.testing.assert_array_equal(img_array[centro_y, centro_x], cor_original)
    assert img_array[0, 0, 3] == 0


def test_canvas_render_achatar_edicoes_e_transformar(canvas_render):
    """Testa o achatamento de edições sobre a imagem base antes e após rotação do Layer."""
    cor_azul = (0, 0, 255, 255)
    cor_vermelha = (255, 0, 0, 255)
    layer = make_layer(w=100, h=100, color=cor_azul)
    img_edicao = make_img(w=20, h=20, color=cor_vermelha)
    layer.add_edit(img_edicao, Region.from_rect(10, 10, 20, 20))

    img_renderizada = canvas_render.render_layer(layer)
    array_final = img_renderizada[...]

    np.testing.assert_array_equal(array_final[0, 0], cor_azul)
    np.testing.assert_array_equal(array_final[20, 20], cor_vermelha)

    layer.transform.rotate(90)
    img_rotacionada = canvas_render.render_layer(layer)
    array_rotacionado = img_rotacionada[...]

    assert not np.array_equal(array_rotacionado[20, 20], cor_vermelha)
    np.testing.assert_array_equal(array_rotacionado[50, 50], cor_azul)


def test_render_fluxo_real_com_quina(canvas_render):
    """Testa fluxo encadeado de rotação inicial, adição de edição na quina, rotação inversa e escala."""
    bg_image = make_img(w=100, h=100, color=(0, 0, 255, 255))
    layer = Layer(bg_image)
    layer.transform.rotate(90)

    edit_data = np.zeros((20, 20, 4), dtype=np.uint8)
    edit_data[0:10, 0:10] = (255, 0, 0, 255)
    edit_img = Image(edit_data, ImageFormat.RGBA)
    layer.add_edit(edit_img, Region.from_rect(80, 0, 20, 20))

    layer.transform.rotate(-45)
    layer.transform.scale(2.0, 2.0)

    result_image = canvas_render.render_layer(layer)
    data = result_image[...]

    m_final = mat_final(layer, *layer.global_region.top_left)
    edit_obj = layer._edits[1]
    full_edit_matrix = m_final @ edit_obj.local_matrix

    p = full_edit_matrix @ np.array([5, 5, 1.0])
    cx, cy = int(round(p[0])), int(round(p[1]))
    pixel = data[cy, cx]

    assert pixel[0] > 240
    assert pixel[2] < 10


@pytest.mark.parametrize(
    "edit_rect, edit_color, sample_coord, expected_color",
    [
        pytest.param(
            (50, 50, 10, 10),
            (255, 0, 0, 255),
            (55, 55),
            (255, 0, 0, 255),
            id="edit_centro_transladado",
        ),
        pytest.param(
            (-10, 10, 20, 20),
            (255, 0, 0, 255),
            (15, 0),
            (255, 0, 0, 255),
            id="edit_vazando_esquerda",
        ),
        pytest.param(
            (90, 10, 20, 20),
            (0, 0, 255, 255),
            (15, 95),
            (0, 0, 255, 255),
            id="edit_vazando_direita",
        ),
    ],
)
def test_canvas_render_layer_edits_positioning_and_clipping(
    canvas_render, edit_rect, edit_color, sample_coord, expected_color
):
    """Valida o posicionamento e o recorte de EditLayers sobrepostos à camada base."""
    layer = make_layer(w=100, h=100, color=(0, 0, 0, 0))
    edit_img = make_img(w=edit_rect[2], h=edit_rect[3], color=edit_color)
    layer.add_edit(edit_img, Region.from_rect(*edit_rect))

    result = canvas_render.render_layer(layer)
    y, x = sample_coord

    np.testing.assert_array_equal(result[y, x], expected_color)


def test_viewport_render_scene_posicionamento_camadas(viewport_render):
    """Valida se o ViewportRender posiciona e compõe corretamente camadas transladadas."""
    viewport = Viewport((800, 600), 1.0)

    fundo = make_layer(w=1080, h=719, color=(0, 0, 255, 255))
    logo = make_layer(w=200, h=200, x=150, y=100, color=(255, 0, 0, 255))

    comp = viewport_render.render_scene([fundo, logo], viewport)

    assert comp.size == (800, 600)
    np.testing.assert_array_equal(comp[0, 0], [0, 0, 255, 255])
    np.testing.assert_array_equal(comp[350, 450], [255, 0, 0, 255])


# ==============================================================================
# Composição de Cenas com Grupos (GroupLayer no Canvas)
# ==============================================================================


def test_canvas_render_patch_with_restricting_view_region():
    """Valida se o render_patch restringe e retorna a imagem com o tamanho exato da view_region para uma camada simples."""
    layer = make_layer(w=100, h=100, x=50, y=50, color=(255, 0, 0, 255))
    canvas = Canvas.from_size(200, 200)
    view_region = Region.from_rect(70, 70, 50, 50)

    renderer = CanvasRender()
    result_image = renderer.render_patch([layer], canvas, view_region=view_region)

    assert result_image.size == (50, 50)
    np.testing.assert_array_equal(result_image[0, 0], [255, 0, 0, 255])


def test_canvas_render_patch_com_group_layer_e_view_region():
    """Valida se o render_patch restringe e retorna a imagem com o tamanho exato da view_region para um GroupLayer."""
    group = GroupLayer()
    group.transform.translate(50, 50)
    child = make_layer(w=100, h=100, x=20, y=20, color=(0, 255, 0, 255))
    group.append(child)

    canvas = Canvas.from_size(300, 300)
    view_region = Region.from_rect(70, 70, 50, 50)

    renderer = CanvasRender()
    result_image = renderer.render_patch([group], canvas, view_region=view_region)

    assert result_image.size == (50, 50)
    np.testing.assert_array_equal(result_image[0, 0], [0, 255, 0, 255])


def test_canvas_render_patch_outside_surface_returns_none():
    """Valida se o render_patch retorna None quando a view_region solicitada nao intercepta o surface."""
    layer = make_layer(w=100, h=100, x=50, y=50)
    canvas = Canvas.from_size(200, 200)
    view_region = Region.from_rect(500, 500, 50, 50)

    renderer = CanvasRender()
    result_image = renderer.render_patch([layer], canvas, view_region=view_region)

    assert result_image is None


def test_canvas_render_patch_partial_overlap_returns_effective_size():
    """Valida se o render_patch recorta e retorna o tamanho da intersecção quando a view_region ultrapassa as bordas do surface."""
    layer = make_layer(w=100, h=100, x=0, y=0)
    canvas = Canvas.from_size(200, 200)
    view_region = Region.from_rect(-50, -50, 100, 100)

    renderer = CanvasRender()
    result_image = renderer.render_patch([layer], canvas, view_region=view_region)

    assert result_image is not None
    assert result_image.size == (50, 50)


@pytest.mark.parametrize(
    "group_trans, group_frame_rect, child_rect, expect_child_slice, expect_group_slice",
    [
        pytest.param(
            (50, 50),
            (40, 40, 120, 120),
            (20, 20, 50, 50),
            (30, 30, 50, 50),
            (40, 40, 120, 120),
            id="expanded_border",
        ),
        pytest.param(
            (50, 50),
            (50, 50, 100, 100),
            (50, 50, 100, 100),
            (50, 50, 50, 50),
            (50, 50, 100, 100),
            id="child_clipping",
        ),
        pytest.param(
            (200, 200),
            (200, 200, 150, 150),
            (20, 20, 50, 50),
            (20, 20, 50, 50),
            (200, 200, 100, 100),
            id="group_clipping",
        ),
    ],
)
def test_canvas_render_scene_group_layer_scenarios(
    mocker,
    group_trans,
    group_frame_rect,
    child_rect,
    expect_child_slice,
    expect_group_slice,
):
    """Testa cenários espaciais de GroupLayer: enquadramento com borda, clipping interno e recorte na borda do Canvas."""

    def fake_render_area(layer, frame, interp=None):
        if frame.dst_region is not None:
            return Image.new(frame.dst_region.size, ImageFormat.RGBA)
        return None

    mocker.patch.object(BaseRenderer, "render_area", side_effect=fake_render_area)

    group = GroupLayer()
    gx, gy = group_trans
    group.transform.translate(gx, gy)

    cx, cy, cw, ch = child_rect
    child = make_layer(w=cw, h=ch, x=cx, y=cy)
    group.append(child)

    mocker.patch.object(
        type(group.control.frame),
        "global_region",
        new_callable=mocker.PropertyMock,
        return_value=Region.from_rect(*group_frame_rect),
    )

    canvas = Canvas.from_size(300, 300)
    spy_view = mocker.spy(Image, "view")
    renderer = CanvasRender()
    result_image = renderer.render_scene([group], canvas)

    assert result_image.size == (300, 300)
    assert spy_view.call_count == 2
    assert spy_view.call_args_list[0][0][1] == Region.from_rect(*expect_child_slice)
    assert spy_view.call_args_list[1][0][1] == Region.from_rect(*expect_group_slice)


def test_canvas_render_scene_nested_group_layers(mocker, monkeypatch):
    """Testa a composição em cascata de múltiplos níveis de GroupLayers aninhados."""

    def fake_render_area(layer, frame, interp=None):
        if frame.dst_region is not None:
            return Image.new(frame.dst_region.size, ImageFormat.RGBA)
        return None

    mocker.patch.object(BaseRenderer, "render_area", side_effect=fake_render_area)

    root_group = GroupLayer()
    sub_group = GroupLayer()
    root_group.append(sub_group)

    child = make_layer(w=40, h=40, x=20, y=20)
    sub_group.append(child)

    root_group.transform.translate(50, 50)
    sub_group.transform.translate(30, 30)

    def mock_global_region(self):
        if self is sub_group.control.frame:
            return Region.from_rect(80, 80, 80, 80)
        elif self is root_group.control.frame:
            return Region.from_rect(50, 50, 150, 150)
        return self._calculate_region("global_region")

    monkeypatch.setattr(
        type(root_group.control.frame),
        "global_region",
        property(mock_global_region),
    )

    canvas = Canvas.from_size(300, 300)
    spy_view = mocker.spy(Image, "view")
    renderer = CanvasRender()
    _ = renderer.render_scene([root_group], canvas)

    assert spy_view.call_count == 3
    assert spy_view.call_args_list[0][0][1] == Region.from_rect(20, 20, 40, 40)
    assert spy_view.call_args_list[1][0][1] == Region.from_rect(30, 30, 80, 80)
    assert spy_view.call_args_list[2][0][1] == Region.from_rect(50, 50, 150, 150)


def test_canvas_render_scene_com_grupo_vazio_retorna_imagem_vazia():
    """Valida se CanvasRender.render_scene executa com sucesso retornando canvas limpo quando o grupo é vazio."""
    group = GroupLayer()
    canvas = Canvas.from_size(200, 200)
    renderer = CanvasRender()

    result = renderer.render_scene([group], canvas)

    assert result.size == (200, 200)
    assert np.all(result[...] == 0)


def test_canvas_render_scene_grupos_aninhados_profundos_3_niveis():
    """Valida a renderização correta de uma hierarquia de 3 níveis de GroupLayers aninhados."""
    root_group = GroupLayer()
    mid_group = GroupLayer()
    leaf_group = GroupLayer()

    child = make_layer(w=40, h=40, color=(255, 0, 0, 255))
    leaf_group.append(child)
    mid_group.append(leaf_group)
    root_group.append(mid_group)

    root_group.transform.translate(50, 50)
    mid_group.transform.translate(30, 20)
    leaf_group.transform.translate(10, 10)

    canvas = Canvas.from_size(300, 300)
    renderer = CanvasRender()

    result = renderer.render_scene([root_group], canvas)

    assert result.size == (300, 300)
    np.testing.assert_array_equal(result[80, 90], [255, 0, 0, 255])
    assert result[0, 0, 3] == 0


def test_canvas_render_scene_early_exit_por_oclusao_total(mocker):
    """Valida se CanvasRender.render_scene interrompe a travessia sem renderizar camadas sob oclusão total."""
    canvas = Canvas.from_size(200, 200)
    top_solid_layer = make_layer(w=200, h=200, color=(255, 0, 0, 255))
    bottom_layer = make_layer(w=200, h=200, color=(0, 255, 0, 255))

    renderer = CanvasRender()
    spy_area = mocker.spy(renderer, "render_area")

    result = renderer.render_scene([bottom_layer, top_solid_layer], canvas)

    assert result.size == (200, 200)
    assert spy_area.call_count == 1


def test_canvas_render_scene_ignora_camada_com_visible_false(mocker):
    """Valida se CanvasRender.render_scene ignora camadas com visible=False sem chamar render_area."""
    canvas = Canvas.from_size(200, 200)
    invisible_layer = make_layer(w=200, h=200, color=(255, 0, 0, 255))
    invisible_layer.visible = False
    visible_layer = make_layer(w=200, h=200, color=(0, 0, 255, 255))

    renderer = CanvasRender()
    spy_area = mocker.spy(renderer, "render_area")

    result = renderer.render_scene([invisible_layer, visible_layer], canvas)

    assert result.size == (200, 200)
    assert spy_area.call_count == 1
    np.testing.assert_array_equal(result[0, 0], [0, 0, 255, 255])


# ==============================================================================
# Casos de Borda: Transformações Geométricas Extremas (Rotações e Escalas)
# ==============================================================================


@pytest.mark.parametrize(
    "angle, expected_size_approx",
    [
        pytest.param(45, (141, 141), id="rotacao_45_graus"),
        pytest.param(135, (141, 141), id="rotacao_135_graus"),
        pytest.param(30, (136, 136), id="rotacao_30_graus"),
    ],
)
def test_canvas_render_rotacoes_nao_ortogonais(
    canvas_render, angle, expected_size_approx
):
    """Valida se o render_layer gera a imagem com a bounding box expandida e pixel central preservado."""
    layer = make_layer(w=100, h=100, color=(255, 0, 0, 255))
    layer.transform.rotate(angle)

    result = canvas_render.render_layer(layer)

    assert abs(result.width - expected_size_approx[0]) <= 2
    assert abs(result.height - expected_size_approx[1]) <= 2
    cx, cy = result.width // 2, result.height // 2
    np.testing.assert_array_equal(result[cy, cx], [255, 0, 0, 255])


def test_canvas_render_camada_dimensao_minima_1x1(canvas_render):
    """Valida se uma camada de 1x1px é renderizada sem erros aritméticos de divisão por zero."""
    layer = make_layer(w=1, h=1, color=(255, 128, 0, 255))
    layer.transform.translate(10, 10).rotate(45)

    result = canvas_render.render_layer(layer)

    assert result.width >= 1
    assert result.height >= 1


@pytest.mark.parametrize(
    "scale_x, scale_y, expected_size",
    [
        pytest.param(0.1, 0.1, (10, 10), id="downscaling_extremo_10_porcento"),
        pytest.param(5.0, 5.0, (500, 500), id="upscaling_extremo_500_porcento"),
    ],
)
def test_canvas_render_escalas_extremas(canvas_render, scale_x, scale_y, expected_size):
    """Valida a renderização com fatores de escala extremos de redução e ampliação."""
    layer = make_layer(w=100, h=100, color=(0, 255, 0, 255))
    layer.transform.scale(scale_x, scale_y)

    result = canvas_render.render_layer(layer)

    assert result.size == expected_size


def test_render_layer_multiplos_edits_formatos_mistos(canvas_render):
    """Valida render_layer para camada contendo múltiplos EditLayers com formatos mistos (RGB, RGBA, GRAY)."""
    base_img = Image(
        np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8), ImageFormat.RGB
    )
    layer = Layer(base_img)

    sticker_data = np.full((30, 30, 4), [255, 0, 0, 255], dtype=np.uint8)
    sticker_img = Image(sticker_data, ImageFormat.RGBA)
    layer.add_edit(sticker_img, Region.from_rect(10, 10, 30, 30))

    stamp_data = np.full((20, 20, 1), 128, dtype=np.uint8)
    stamp_img = Image(stamp_data, ImageFormat.GRAY)
    layer.add_edit(stamp_img, Region.from_rect(50, 50, 20, 20))

    result = canvas_render.render_layer(layer)

    assert result.format == ImageFormat.RGB
    np.testing.assert_array_equal(result[0, 0], [0, 0, 255])
    np.testing.assert_array_equal(result[15, 15], [255, 0, 0])
    np.testing.assert_array_equal(result[55, 55], [128, 128, 128])


def test_render_crop_and_rotate_mariachi_scenario(canvas_render):
    """Valida o cenário mariachi com crop de janela 400x400 e rotação de 45° sem corte em 90° e sem efeito pêndulo."""
    doc = Document("Mariachi Test", 736, 1104, history=False)
    cor_fundo = (10, 50, 200, 255)
    img_data = np.full((1000, 1000, 4), cor_fundo, dtype=np.uint8)
    img = Image(img_data, ImageFormat.RGBA)
    layer = Layer(img, name="fundo")
    doc.add(layer)

    doc.content.crop(layer, (100, 50, 400, 400))
    layer.transform.rotate(45)

    result = doc.render()

    np.testing.assert_array_equal(result[250, 300], cor_fundo)
    assert result[50, 100, 3] == 0
    assert result[5, 300, 3] == 255


def test_render_crop_rotate_align_and_fit_content_mariachi_scenario():
    """Valida o ciclo completo mariachi de crop, rotacao 45, align ao canvas e fit_content sem corte do chapeu."""
    doc = Document("Mariachi Full Test", 736, 1104, history=False)

    y_coords, x_coords = np.indices((1104, 736))
    img_data = np.zeros((1104, 736, 4), dtype=np.uint8)
    img_data[..., 0] = (x_coords % 256).astype(np.uint8)
    img_data[..., 1] = (y_coords % 256).astype(np.uint8)
    img_data[..., 2] = ((x_coords + y_coords) % 256).astype(np.uint8)
    img_data[..., 3] = 255

    img = Image(img_data, ImageFormat.RGBA)
    layer = Layer(img, name="fundo")
    doc.add(layer)

    doc.content.crop(layer, (100, 50, 400, 400))
    layer.transform.rotate(45)
    doc.layout.align(layer, doc.canvas)

    assert layer.global_region.top_left.to_int() == (85, 269)
    assert layer.global_region.size.to_int() == (566, 566)
    assert layer.control._offset.top_left == (-100, -50)

    img_cropped = doc.render()

    doc.layout.fit_content(layer)

    assert layer.control._offset.top_left.to_int() == (0, 0)
    assert layer.global_region.size.to_int() == (1301, 1301)

    img_uncropped = doc.render()
    assert img_uncropped.size == (736, 1104)

    mask_cropped = img_cropped[...][..., 3] == 255
    diff = np.abs(
        img_cropped[...][mask_cropped].astype(int)
        - img_uncropped[...][mask_cropped].astype(int)
    )
    assert np.mean(diff) < 3.0


def test_render_crop_rotate_align_and_fit_content_mariachi_scenario_com_borda_transparente_1px():
    """Valida o ciclo completo mariachi de crop, rotacao 45, align e fit_content em imagem com borda transparente de 1px."""
    doc = Document("Mariachi 1px Border Test", 736, 1104, history=False)

    y_coords, x_coords = np.indices((1104, 736))
    img_data = np.zeros((1104, 736, 4), dtype=np.uint8)
    img_data[..., 0] = (x_coords % 256).astype(np.uint8)
    img_data[..., 1] = (y_coords % 256).astype(np.uint8)
    img_data[..., 2] = ((x_coords + y_coords) % 256).astype(np.uint8)
    img_data[..., 3] = 255

    # Aplica borda transparente de 1px nas 4 extremidades
    img_data[0, :, 3] = 0
    img_data[-1, :, 3] = 0
    img_data[:, 0, 3] = 0
    img_data[:, -1, 3] = 0

    img = Image(img_data, ImageFormat.RGBA)
    layer = Layer(img, name="fundo")
    doc.add(layer)

    doc.content.crop(layer, (100, 50, 400, 400))
    layer.transform.rotate(45)
    doc.layout.align(layer, doc.canvas)

    assert layer.global_region.top_left.to_int() == (85, 269)
    assert layer.global_region.size.to_int() == (566, 566)
    assert layer.control._offset.top_left == (-100, -50)

    img_cropped = doc.render()

    doc.layout.fit_content(layer)

    assert layer.control._offset.top_left.to_int() == (0, -1)
    assert layer.global_region.size.to_int() == (1298, 1298)

    img_uncropped = doc.render()
    assert img_uncropped.size == (736, 1104)

    mask_cropped = img_cropped[...][..., 3] == 255
    diff = np.abs(
        img_cropped[...][mask_cropped].astype(int)
        - img_uncropped[...][mask_cropped].astype(int)
    )
    assert np.mean(diff) < 3.0


def test_render_scene_with_single_edit_layers_and_opacity(canvas_render):
    """Valida se CanvasRender compõe corretamente múltiplas camadas simples com opacidade no Canvas."""
    bg_layer = make_layer(w=100, h=100, color=(255, 0, 0, 255))
    fg_layer = make_layer(w=100, h=100, color=(0, 0, 255, 255))
    fg_layer.opacity = 0.5

    canvas = Canvas.from_size(100, 100)
    result = canvas_render.render_scene([bg_layer, fg_layer], canvas)

    assert result.size == (100, 100)
    pixel = result[50, 50]
    assert pixel[0] == 127
    assert pixel[1] == 0
    assert pixel[2] == 128
    assert pixel[3] == 255


def test_render_scene_mixed_single_and_multi_edit_layers(canvas_render):
    """Valida se CanvasRender compõe cena com camadas de edit único (Fast-Path) e multi-edits."""
    base_layer = make_layer(w=100, h=100, color=(255, 0, 0, 255))

    multi_layer = make_layer(w=100, h=100, color=(0, 0, 255, 255))
    multi_layer.add_edit(
        Image(np.full((50, 50, 4), (0, 255, 0, 255), dtype=np.uint8), ImageFormat.RGBA),
        Region.from_rect(0, 0, 50, 50),
    )

    canvas = Canvas.from_size(100, 100)
    result = canvas_render.render_scene([base_layer, multi_layer], canvas)

    assert result.size == (100, 100)
    np.testing.assert_array_equal(result[25, 25], [0, 255, 0, 255])
    np.testing.assert_array_equal(result[75, 75], [0, 0, 255, 255])


def test_render_group_layer_com_multiplos_filhos_rotacionados_nao_contamina_buffers(
    canvas_render,
):
    """Valida se renderizar GroupLayer rotacionado com multiplos filhos nao contamina buffers entre camadas."""
    group = GroupLayer(name="Group")

    child_small = make_layer(w=40, h=40, color=(0, 255, 0, 255))
    child_small.transform.translate(140, 140)

    child_large = make_layer(w=200, h=200, color=(255, 0, 0, 255))

    group.append(child_small)
    group.append(child_large)
    group.transform.rotate(45)

    result = canvas_render.render_container(group)

    assert result is not None
    # Verifica que a regiao superior esquerda de child_large nao contem pixels verdes vazados de child_small
    assert not np.any(
        (result[0:60, 0:60, 0] == 0)
        & (result[0:60, 0:60, 1] == 255)
        & (result[0:60, 0:60, 2] == 0)
    )
