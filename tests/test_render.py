from anicrop.enums import RenderFlags, WarpMode, BlendMode, InterpolationOption
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.render import (
    LayerRender,
    render_patch,
    generate_opacity_mask,
    ViewportRender,
)
from anicrop.viewport import Viewport
from anicrop.spatial import Region, Span
from anicrop.transform import mat_final
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
import gc
import weakref


# Fixture da classe que vamos testar
@pytest.fixture
def lr():
    """Retorna uma instância de LayerRender."""
    return LayerRender()


# Funções auxiliares para gerar Layers e Edits
def make_img(w=100, h=100, color=(255, 0, 0, 255), form=ImageFormat.RGBA):
    # Gera uma imagem com uma cor sólida
    img_data = np.zeros((h, w, 4), dtype=np.uint8)
    img_data[:] = color
    return Image(img_data, form)


def make_layer(w=100, h=100, x=0, y=0, color=(255, 0, 0, 255)):
    img = make_img(w, h, color)
    layer = Layer(img)
    if x != 0:
        layer.x = x
    if y != 0:
        layer.y = y
    return layer


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_LayerRender_identidade_sem_transformacao(lr, method):
    """
    Testa se um Layer sem transformações (Escala=1, Rotação=0, Pos=0,0)
    é renderizado exatamente igual à imagem original, pixel por pixel.
    """
    width, height = 50, 50
    original_layer = make_layer(w=width, h=height, color=(100, 150, 200, 255))

    # Renderiza o layer usando o método escolhido
    render_fn = getattr(lr, method)
    rendered_image = render_fn(original_layer)

    assert rendered_image.width == width
    assert rendered_image.height == height
    np.testing.assert_array_equal(rendered_image[...], original_layer.image[...])


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_LayerRender_rotacao_expansao_segura(lr, method):
    """
    Testa se o LayerRender expande a imagem corretamente ao rotacionar
    em 45 graus.
    """
    width, height = 100, 100
    cor_original = (255, 0, 0, 255)
    layer = make_layer(w=width, h=height, color=cor_original)

    layer.rotation.angle = 45

    render_fn = getattr(lr, method)
    rendered_image = render_fn(layer)
    bbox = layer.global_region

    assert rendered_image.width == bbox.width
    assert rendered_image.height == bbox.height

    img_array = rendered_image[...]
    centro_x, centro_y = bbox.width // 2, bbox.height // 2
    pixel_central = img_array[centro_y, centro_x]

    assert pixel_central[3] == 255
    np.testing.assert_array_equal(pixel_central, cor_original)

    pixel_canto = img_array[0, 0]
    assert pixel_canto[3] == 0


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_LayerRender_achatar_edicoes_e_transformar(lr, method):
    """
    Testa se o renderizador consegue fazer o merge das edições (EditLayer)
    sobre a imagem base.
    """
    cor_azul = (0, 0, 255, 255)
    layer = make_layer(w=100, h=100, color=cor_azul)

    cor_vermelha = (255, 0, 0, 255)
    img_edicao = make_img(w=20, h=20, color=cor_vermelha)

    regiao_edicao = Region(Span(10, 20), Span(10, 20))
    layer.add_edit(img_edicao, regiao_edicao)

    render_fn = getattr(lr, method)
    img_renderizada = render_fn(layer)
    array_final = img_renderizada[...]

    np.testing.assert_array_equal(array_final[0, 0], cor_azul)
    np.testing.assert_array_equal(array_final[20, 20], cor_vermelha)

    layer.rotation.angle = 90
    img_rotacionada = render_fn(layer)
    array_rotacionado = img_rotacionada[...]

    assert not np.array_equal(array_rotacionado[20, 20], cor_vermelha)
    np.testing.assert_array_equal(array_rotacionado[50, 50], cor_azul)


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_render_fluxo_real_com_quina(lr, method):
    bg_data = np.zeros((100, 100, 4), dtype=np.uint8)
    bg_data[:] = [0, 0, 255, 255]
    bg_image = Image(bg_data, ImageFormat.RGBA)
    layer = Layer(bg_image, Region(Span(0, 100), Span(0, 100)))

    layer.rotation.angle = 90

    edit_data = np.zeros((20, 20, 4), dtype=np.uint8)
    C_TL = (255, 0, 0, 255)
    edit_data[0:10, 0:10] = C_TL
    edit_img = Image(edit_data, ImageFormat.RGBA)

    clique_region = Region(Span(80, 20), Span(0, 20))
    layer.add_edit(edit_img, clique_region)

    layer.rotation.angle += -45
    layer.scale = (2.0, 2.0)

    render_fn = getattr(lr, method)
    result_image = render_fn(layer)
    data = result_image[...]

    m_final = mat_final(layer, *layer.global_region.top_left)
    edit_obj = layer._edits[1]
    full_edit_matrix = m_final @ edit_obj.local_matrix

    p = full_edit_matrix @ np.array([5, 5, 1.0])
    cx, cy = int(round(p[0])), int(round(p[1]))

    pixel = data[cy, cx]
    assert pixel[0] > 240
    assert pixel[2] < 10


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_render_bug_offset_translation(lr, method):
    base_img = Image.new((100, 100), ImageFormat.RGBA, color=0)
    layer = Layer(base_img)

    edit_data = np.zeros((10, 10, 4), dtype=np.uint8)
    edit_data[:] = [255, 0, 0, 255]
    edit_img = Image(edit_data, ImageFormat.RGBA)

    offset_region = Region(Span(50, 10), Span(50, 10))
    layer.add_edit(edit_img, offset_region)

    render_fn = getattr(lr, method)
    result = render_fn(layer)
    data = result[...]

    pixel_center = data[55, 55]
    np.testing.assert_array_equal(pixel_center, [255, 0, 0, 255])


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_render_edit_parcialmente_fora(lr, method):
    base_img = Image.new((100, 100), ImageFormat.RGBA, color=0)
    layer = Layer(base_img)

    edit_data = np.zeros((20, 20, 4), dtype=np.uint8)
    edit_data[:] = [255, 0, 0, 255]
    edit_img = Image(edit_data, ImageFormat.RGBA)

    offset_region = Region(Span(-10, 20), Span(10, 20))
    layer.add_edit(edit_img, offset_region)

    render_fn = getattr(lr, method)
    result = render_fn(layer)
    data = result[...]

    assert np.array_equal(data[15, 0], [255, 0, 0, 255])
    assert np.array_equal(data[15, 11], [0, 0, 0, 0])


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_render_edit_parcialmente_fora_bicolor(lr, method):
    base_img = Image.new((100, 100), ImageFormat.RGBA, color=0)
    layer = Layer(base_img)

    edit_data = np.zeros((20, 20, 4), dtype=np.uint8)
    edit_data[:, :10] = [0, 255, 0, 255]
    edit_data[:, 10:] = [255, 0, 0, 255]
    edit_img = Image(edit_data, ImageFormat.RGBA)

    offset_region = Region(Span(-10, 20), Span(10, 20))
    layer.add_edit(edit_img, offset_region)

    render_fn = getattr(lr, method)
    result = render_fn(layer)
    data = result[...]

    assert np.array_equal(data[15, 0], [255, 0, 0, 255])


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_render_edit_borda_direita(lr, method):
    base_img = Image.new((100, 100), ImageFormat.RGBA, color=0)
    layer = Layer(base_img)

    edit_data = np.zeros((20, 20, 4), dtype=np.uint8)
    edit_data[:] = [0, 0, 255, 255]
    edit_img = Image(edit_data, ImageFormat.RGBA)

    offset_region = Region(Span(90, 20), Span(10, 20))
    layer.add_edit(edit_img, offset_region)

    render_fn = getattr(lr, method)
    result = render_fn(layer)

    assert np.array_equal(result[15, 95], [0, 0, 255, 255])


def test_render_cache_limpeza_automatica(lr):
    """
    Valida se o cache no LayerRender é limpo automaticamente quando o Layer morre.
    """
    img = make_img(10, 10)
    layer = Layer(img)
    dummy_image = Image.new(layer.region.size, layer.format)

    lr._cache[layer._id] = dummy_image
    id_ref = weakref.ref(layer._id)

    assert layer._id in lr._cache

    del layer
    gc.collect()

    assert id_ref() is None, "O Id do layer não foi coletado!"
    assert len(lr._cache) == 0, "O cache não foi limpo!"


# ############################# Testes de Lógica de Cache (Stress/Integration) #####################################


def test_render_cache_falha_inicial(mocker, lr):
    """Garante que o __flatten_edits é chamado no primeiro render COMPLETO."""
    layer = make_layer(color=(255, 0, 0, 255))
    mock_img = Image.new(layer.region.size, layer.format, color=(255, 0, 0, 255))

    mock_flatten = mocker.patch.object(
        LayerRender, "_LayerRender__flatten_edits", return_value=mock_img
    )

    lr.render(layer)

    assert mock_flatten.call_count == 1
    assert layer._id in lr._cache


def test_render_cache_reaproveita_na_translacao(mocker, lr):
    """Garante que translação pura NÃO dispara um novo __flatten_edits se houver cache completo."""
    layer = make_layer()
    mock_img = Image.new(layer.region.size, layer.format)
    mock_flatten = mocker.patch.object(
        LayerRender, "_LayerRender__flatten_edits", return_value=mock_img
    )

    # 1. Renderização Completa para popular o cache mestre
    lr.render(layer)
    assert mock_flatten.call_count == 1

    # 2. Translação (Deve dar cache hit nos pixels)
    layer.x += 50
    layer.y += 50

    result = lr.render(layer)

    assert mock_flatten.call_count == 1, (
        "ERRO: __flatten_edits foi chamado desnecessariamente!"
    )
    assert result is not None


def test_render_cache_reprocessa_na_distorcao(mocker, lr):
    """Garante que rotação/escala DISPARA um novo __flatten_edits."""
    layer = make_layer()
    mock_img = Image.new(layer.region.size, layer.format)
    mock_flatten = mocker.patch.object(
        LayerRender, "_LayerRender__flatten_edits", return_value=mock_img
    )

    # 1. Primeiro render completo
    lr.render(layer)

    # 2. Rotação (Invalida pixels)
    layer.rotation = 45
    lr.render(layer)

    assert mock_flatten.call_count == 2, (
        "ERRO: __flatten_edits deveria ter sido chamado!"
    )


def test_render_cache_recorte_de_area(mocker, lr):
    """Valida se o render_area usa o cache mestre para recortes."""
    layer = make_layer(w=100, h=100)
    full_img = Image.new((100, 100), layer.format, color=(255, 0, 0, 255))

    # Simula cache populado via render
    lr._cache[layer._id] = full_img
    layer._commit_render_state()

    view_region = Region(Span(10, 20), Span(10, 20))
    mocker.patch.object(
        LayerRender, "_LayerRender__render_region", return_value=view_region
    )
    mock_flatten = mocker.patch.object(LayerRender, "_LayerRender__flatten_edits")

    # Ação
    result = lr.render_area(layer, view_region=view_region)

    assert mock_flatten.call_count == 0
    assert result.width == 20
    assert result.height == 20
    np.testing.assert_array_equal(result[0, 0], [255, 0, 0, 255])


def test_render_area_sem_sujar_cache_global(mocker, lr):
    """Garante que renderizações PARCIAIS não povoam o cache global."""
    layer = make_layer(w=100, h=100)
    layer._commit_render_state()

    view_region = Region(Span(0, 10), Span(0, 10))
    mocker.patch.object(
        LayerRender, "_LayerRender__render_region", return_value=view_region
    )

    lr.render_area(layer, view_region=view_region)

    assert layer._id not in lr._cache


def test_render_area_coordenada_local_do_cache(mocker, lr):
    """Verifica se o render_area aplica o crop correto no cache."""
    layer = make_layer(w=100, h=100, x=50, y=50)

    cache_img_data = np.zeros((100, 100, 4), dtype=np.uint8)
    cache_img_data[:] = [255, 0, 0, 255]
    cache_img_data[10, 10] = [0, 255, 0, 255]
    cache_img = Image(cache_img_data, layer.format)

    lr._cache[layer._id] = cache_img
    layer._commit_render_state()

    view_region = Region(Span(60, 1), Span(60, 1))
    mocker.patch.object(
        LayerRender, "_LayerRender__render_region", return_value=view_region
    )
    mocker.patch.object(LayerRender, "_LayerRender__flatten_edits")

    result = lr.render_area(layer, view_region=view_region)

    np.testing.assert_array_equal(result[0, 0], [0, 255, 0, 255])


def test_render_area_sem_cache_e_flags_limpas_reprocessa_pixels(mocker, lr):
    """
    Simula o cenário: render_area é chamado (não popula cache mestre),
    o layer é commitado, e render_area é chamado novamente.
    Espera-se que __flatten_edits seja chamado duas vezes.
    """
    layer = make_layer(w=100, h=100)
    mock_img = Image.new(layer.region.size, layer.format)
    mock_flatten = mocker.patch.object(
        LayerRender, "_LayerRender__flatten_edits", return_value=mock_img
    )

    # 1. Primeira chamada ao render_area (ele renderiza, mas NÃO popula o self._cache)
    result_first_call = lr.render_area(layer)
    assert result_first_call is not None  # Esperamos uma imagem aqui
    assert mock_flatten.call_count == 1
    assert layer._id not in lr._cache  # O cache não deve ter sido populado

    # 2. O layer é 'commitado'. As flags ficam NONE.
    layer._commit_render_state()
    assert layer._resolve_render() == RenderFlags.NONE

    # 3. Segunda chamada ao render_area.
    # flags & RenderFlags.PIXELS é FALSO.
    # layer._id not in lr._cache é VERDADEIRO (ainda).
    # Então __flatten_edits será chamado NOVAMENTE.
    result_second_call = lr.render_area(layer)

    assert result_second_call is not None
    assert mock_flatten.call_count == 2, (
        "BUG: __flatten_edits foi chamado novamente mesmo com flags limpas e sem alteração!"
    )


# ############################# Testes de Motor de Projeção (Warp Dispatch) #####################################


def test_render_patch_delegates_to_warp_affine_by_default(mocker):
    """Garante que render_patch chama warp_affine quando solicitado."""
    # Setup mínimo para render_patch
    img = make_img(10, 10)
    layer = Layer(img)
    edit = layer._edits[0]

    # Mocks das funções internas
    mock_affine = mocker.patch("anicrop.render.warp_affine")
    mock_perspective = mocker.patch("anicrop.render.warp_perspective")

    # Injetamos os mocks no dicionário de dispatch do módulo
    mocker.patch.dict(
        "anicrop.render.WARP_MODE",
        {WarpMode.AFFINE: mock_affine, WarpMode.PERSPECTIVE: mock_perspective},
    )

    # Ação: Renderiza o patch passando o modo explicitamente
    render_patch(edit.image, np.eye(3), edit.region, warp_mode=WarpMode.AFFINE)

    assert mock_affine.called
    assert not mock_perspective.called


def test_render_patch_delegates_to_warp_perspective(mocker):
    """Garante que render_patch chama warp_perspective quando solicitado."""
    img = make_img(10, 10)
    layer = Layer(img)
    edit = layer._edits[0]

    mock_affine = mocker.patch("anicrop.render.warp_affine")
    mock_perspective = mocker.patch("anicrop.render.warp_perspective")

    # Injetamos os mocks no dicionário de dispatch do módulo
    mocker.patch.dict(
        "anicrop.render.WARP_MODE",
        {WarpMode.AFFINE: mock_affine, WarpMode.PERSPECTIVE: mock_perspective},
    )

    # Ação: Renderiza o patch forçando PERSPECTIVE via argumento
    render_patch(edit.image, np.eye(3), edit.region, warp_mode=WarpMode.PERSPECTIVE)

    assert mock_perspective.called
    assert not mock_affine.called


def test_render_patch_fallback_to_warp_affine(mocker):
    """Garante que render_patch usa warp_affine como fallback se o modo não existir no dict."""
    img = make_img(10, 10)
    layer = Layer(img)
    edit = layer._edits[0]

    mock_affine = mocker.patch("anicrop.render.warp_affine")

    # Simulamos um dicionário VAZIO para forçar o fallback
    mocker.patch.dict("anicrop.render.WARP_MODE", {}, clear=True)

    # Chamamos com um modo que "não existe" no dicionário limpo
    render_patch(edit.image, np.eye(3), edit.region, warp_mode=WarpMode.PERSPECTIVE)

    assert mock_affine.called, "Deveria ter caído no fallback do warp_affine!"


@pytest.mark.parametrize(
    "img_format, test_scenario, is_expected_opaque",
    [
        (ImageFormat.RGBA, "Opaca", True),
        (ImageFormat.RGBA, "Transparencia total", False),
        (ImageFormat.RGBA, "1 pixel no minimo possível para ser transparente", False),
        (ImageFormat.RGB, "Sem canal alpha", True),
    ],
)
def test_generate_opacity_mask(img_format, test_scenario, is_expected_opaque):
    width, height = 100, 100
    channels = img_format.channels
    data = np.zeros((height, width, channels), dtype=np.uint8)

    if test_scenario == "Opaca":
        data[:] = 255
    elif test_scenario == "Transparencia total":
        data[:] = 0
    elif test_scenario == "1 pixel no minimo possível para ser transparente":
        data[:] = 255
        data[50, 50, 3] = 254  # Canal Alpha = 254 em 1 pixel
    elif test_scenario == "Sem canal alpha":
        data[:] = 255

    img = Image(data, img_format)
    # Passamos a região de 100x100 e a viewport de 100x100 para simular "tela cheia"
    mask = generate_opacity_mask(
        img, Region(Span(100), Span(100)), (100, 100), target_size=(32, 32)
    )

    assert mask.shape == (32, 32), "A máscara deve ter o tamanho target_size"

    is_opaque = bool(np.all(mask == 255))
    assert is_opaque == is_expected_opaque


def test_generate_opacity_mask_spatial_mapping():
    """Valida se a miniatura é posicionada e dimensionada corretamente na matriz 32x32."""
    width, height = 200, 200
    data = np.full((height, width, 4), 255, dtype=np.uint8)
    img = Image(data, ImageFormat.RGBA)

    # Layer está na coordenada X=200, Y=400 na tela
    # A tela tem tamanho 800x800
    region = Region(Span(200, 200), Span(400, 200))
    viewport_size = (800, 800)

    # Escala para 32x32: 32 / 800 = 0.04
    # X esperado: start = 200 * 0.04 = 8, end = 400 * 0.04 = 16
    # Y esperado: start = 400 * 0.04 = 16, end = 600 * 0.04 = 24

    mask = generate_opacity_mask(
        img, render_region=region, viewport_size=viewport_size, target_size=(32, 32)
    )

    assert mask.shape == (32, 32)

    # Valida a área interna (deve ser 255)
    inner_area = mask[16:24, 8:16]
    assert np.all(inner_area == 255), "A região mapeada deveria estar opaca (255)!"

    # Valida a área externa (deve ser 0)
    expected_mask = np.zeros((32, 32), dtype=np.uint8)
    expected_mask[16:24, 8:16] = 255

    assert np.array_equal(mask, expected_mask), (
        "A máscara vazou opacidade ou calculou as coordenadas erradas!"
    )


def test_render_scene_culling_no_occlusion():
    """Caso 1: Sem oclusão (todos os layers são renderizados)"""
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()

    layers = [make_layer(w=800, h=600), make_layer(w=800, h=600)]
    layers[0].opacity = 1.0
    layers[0].blend_mode = BlendMode.NORMAL
    layers[1].opacity = 1.0
    layers[1].blend_mode = BlendMode.NORMAL

    rendered = []

    def mock_render(layer, vp, interp=InterpolationOption.LANCZOS):
        rendered.append(layer)
        layer._opacity_mask = np.zeros((32, 32), dtype=np.uint8)
        return make_img(w=800, h=600)

    with patch.object(vr, "render_area", side_effect=mock_render):
        vr.render_scene(layers, viewport)

    assert rendered == [layers[0], layers[1]]


def test_render_scene_culling_total_occlusion_top_layer():
    """Caso 2: Oclusão total pelo layer do topo (índice 0) -> interrompe antes de renderizar o índice 1"""
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()

    layers = [make_layer(w=800, h=600), make_layer(w=800, h=600)]
    layers[0].opacity = 1.0
    layers[0].blend_mode = BlendMode.NORMAL
    layers[1].opacity = 1.0
    layers[1].blend_mode = BlendMode.NORMAL

    rendered = []

    def mock_render(layer, vp, interp=InterpolationOption.LANCZOS):
        rendered.append(layer)
        if layer == layers[0]:
            mask_val = int(255 * layer.opacity)
            layer._opacity_mask = np.full((32, 32), mask_val, dtype=np.uint8)
        else:
            layer._opacity_mask = np.zeros((32, 32), dtype=np.uint8)
        return make_img(w=800, h=600)

    with patch.object(vr, "render_area", side_effect=mock_render):
        vr.render_scene(layers, viewport)

    assert rendered == [layers[0]]


def test_render_scene_culling_occlusion_middle_layer():
    """Caso 3: Oclusão pelo layer do meio (índice 1) em pilha de 3 layers (0=Topo, 1=Meio, 2=Fundo)"""
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()

    layers = [
        make_layer(w=800, h=600),
        make_layer(w=800, h=600),
        make_layer(w=800, h=600),
    ]
    for lyr in layers:
        lyr.opacity = 1.0
        lyr.blend_mode = BlendMode.NORMAL

    rendered = []

    def mock_render(layer, vp, interp=InterpolationOption.LANCZOS):
        rendered.append(layer)
        if layer == layers[1]:  # O meio é opaco
            mask_val = int(255 * layer.opacity)
            layer._opacity_mask = np.full((32, 32), mask_val, dtype=np.uint8)
        else:
            layer._opacity_mask = np.zeros((32, 32), dtype=np.uint8)
        return make_img(w=800, h=600)

    with patch.object(vr, "render_area", side_effect=mock_render):
        vr.render_scene(layers, viewport)

    assert rendered == [layers[0], layers[1]]


def test_render_scene_culling_top_layer_opacity_lt_1():
    """Caso 4: Layer do topo cobre tudo mas tem opacidade < 1.0 -> sem oclusão"""
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()

    layers = [make_layer(w=800, h=600), make_layer(w=800, h=600)]
    layers[0].opacity = 0.9  # Topo não tem 1.0 de opacidade
    layers[0].blend_mode = BlendMode.NORMAL
    layers[1].opacity = 1.0
    layers[1].blend_mode = BlendMode.NORMAL

    rendered = []

    def mock_render(layer, vp, interp=InterpolationOption.LANCZOS):
        rendered.append(layer)
        mask_val = int(255 * layer.opacity)
        layer._opacity_mask = np.full((32, 32), mask_val, dtype=np.uint8)
        return make_img(w=800, h=600)

    with patch.object(vr, "render_area", side_effect=mock_render):
        vr.render_scene(layers, viewport)

    assert rendered == [layers[0], layers[1]]


def test_render_scene_integration_positioning():
    """Valida se o render_scene respeita a posição global (translação) das camadas na composição final."""
    from anicrop.viewport import Viewport

    vr = ViewportRender()
    viewport = Viewport((800, 600), 1.0)

    # Fundo 1080x719 Azul (nasce na origem 0,0)
    fundo = make_layer(w=1080, h=719, color=(0, 0, 255, 255))
    fundo.opacity = 1.0
    fundo.blend_mode = BlendMode.NORMAL

    # Logo 200x200 Vermelha
    logo = make_layer(w=200, h=200, color=(255, 0, 0, 255))

    # MUDANÇA CRUCIAL: Movemos a logo para (150, 100)!
    logo.x = 150
    logo.y = 100

    logo.opacity = 1.0
    logo.blend_mode = BlendMode.NORMAL

    layers = [logo, fundo]

    # Ação
    comp = vr.render_scene(layers, viewport)
    data = comp[...]

    # Verificações
    assert comp.width == 800
    assert comp.height == 600

    final_region = vr._final_region(logo, viewport)
    logo_tela_x, logo_tela_y = final_region.top_left

    # Validação 1: O canto superior esquerdo (0,0) da tela DEVE ser Azul!
    assert np.array_equal(data[0, 0], [0, 0, 255, 255]), (
        "Bug: A logo ignorou as transformações e grudou no (0,0)!"
    )

    # Validação 2: A cor Vermelha deve estar exatamente na coordenada transladada
    assert np.array_equal(data[logo_tela_y, logo_tela_x], [255, 0, 0, 255]), (
        "A logo não apareceu na posição correta da tela!"
    )
