from anicrop.enums import RenderFlags, WarpMode, BlendMode, InterpolationOption
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.render import (
    render_patch,
    generate_opacity_mask,
    CanvasRender,
    ViewportRender,
    ViewportPlan,
    CanvasPlan,
    render_edit,
    render_image,
)
from anicrop.viewport import Viewport
from anicrop.spatial import Region, Span
from anicrop.transform import mat_final, TransformRel
from unittest.mock import patch
import numpy as np
import pytest
import gc
import weakref


# Fixture da classe que vamos testar
@pytest.fixture
def cr():
    """Retorna uma instância de CanvasRender."""
    return CanvasRender()


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
def test_CanvasRender_identidade_sem_transformacao(cr, method):
    """
    Testa se um Layer sem transformações (Escala=1, Rotação=0, Pos=0,0)
    é renderizado exatamente igual à imagem original, pixel por pixel.
    """
    width, height = 50, 50
    original_layer = make_layer(w=width, h=height, color=(100, 150, 200, 255))

    # Renderiza o layer usando o método escolhido
    render_fn = getattr(cr, method)
    rendered_image = render_fn(original_layer)

    assert rendered_image.width == width
    assert rendered_image.height == height
    np.testing.assert_array_equal(
        rendered_image[...], original_layer.image[...])


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_CanvasRender_rotacao_expansao_segura(cr, method):
    """
    Testa se o CanvasRender expande a imagem corretamente ao rotacionar
    em 45 graus.
    """
    width, height = 100, 100
    cor_original = (255, 0, 0, 255)
    layer = make_layer(w=width, h=height, color=cor_original)

    layer.transform.rotate(45)

    render_fn = getattr(cr, method)
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
def test_CanvasRender_achatar_edicoes_e_transformar(cr, method):
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

    render_fn = getattr(cr, method)
    img_renderizada = render_fn(layer)
    array_final = img_renderizada[...]

    np.testing.assert_array_equal(array_final[0, 0], cor_azul)
    np.testing.assert_array_equal(array_final[20, 20], cor_vermelha)

    layer.transform.rotate(90)
    img_rotacionada = render_fn(layer)
    array_rotacionado = img_rotacionada[...]

    assert not np.array_equal(array_rotacionado[20, 20], cor_vermelha)
    np.testing.assert_array_equal(array_rotacionado[50, 50], cor_azul)


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_render_fluxo_real_com_quina(cr, method):
    bg_data = np.zeros((100, 100, 4), dtype=np.uint8)
    bg_data[:] = [0, 0, 255, 255]
    bg_image = Image(bg_data, ImageFormat.RGBA)
    layer = Layer(bg_image, Region(Span(0, 100), Span(0, 100)))

    layer.transform.rotate(90)

    edit_data = np.zeros((20, 20, 4), dtype=np.uint8)
    C_TL = (255, 0, 0, 255)
    edit_data[0:10, 0:10] = C_TL
    edit_img = Image(edit_data, ImageFormat.RGBA)

    clique_region = Region(Span(80, 20), Span(0, 20))
    layer.add_edit(edit_img, clique_region)

    layer.transform.rotate(-45)
    layer.transform.scale(2.0, 2.0)

    render_fn = getattr(cr, method)
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
def test_render_bug_offset_translation(cr, method):
    base_img = Image.new((100, 100), ImageFormat.RGBA, color=0)
    layer = Layer(base_img)

    edit_data = np.zeros((10, 10, 4), dtype=np.uint8)
    edit_data[:] = [255, 0, 0, 255]
    edit_img = Image(edit_data, ImageFormat.RGBA)

    offset_region = Region(Span(50, 10), Span(50, 10))
    layer.add_edit(edit_img, offset_region)

    render_fn = getattr(cr, method)
    result = render_fn(layer)
    data = result[...]

    pixel_center = data[55, 55]
    np.testing.assert_array_equal(pixel_center, [255, 0, 0, 255])


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_render_edit_parcialmente_fora(cr, method):
    base_img = Image.new((100, 100), ImageFormat.RGBA, color=0)
    layer = Layer(base_img)

    edit_data = np.zeros((20, 20, 4), dtype=np.uint8)
    edit_data[:] = [255, 0, 0, 255]
    edit_img = Image(edit_data, ImageFormat.RGBA)

    offset_region = Region(Span(-10, 20), Span(10, 20))
    layer.add_edit(edit_img, offset_region)

    render_fn = getattr(cr, method)
    result = render_fn(layer)
    data = result[...]

    assert np.array_equal(data[15, 0], [255, 0, 0, 255])
    assert np.array_equal(data[15, 11], [0, 0, 0, 0])


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_render_edit_parcialmente_fora_bicolor(cr, method):
    base_img = Image.new((100, 100), ImageFormat.RGBA, color=0)
    layer = Layer(base_img)

    edit_data = np.zeros((20, 20, 4), dtype=np.uint8)
    edit_data[:, :10] = [0, 255, 0, 255]
    edit_data[:, 10:] = [255, 0, 0, 255]
    edit_img = Image(edit_data, ImageFormat.RGBA)

    offset_region = Region(Span(-10, 20), Span(10, 20))
    layer.add_edit(edit_img, offset_region)

    render_fn = getattr(cr, method)
    result = render_fn(layer)
    data = result[...]

    assert np.array_equal(data[15, 0], [255, 0, 0, 255])


@pytest.mark.parametrize("method", ["render", "render_area"])
def test_render_edit_borda_direita(cr, method):
    base_img = Image.new((100, 100), ImageFormat.RGBA, color=0)
    layer = Layer(base_img)

    edit_data = np.zeros((20, 20, 4), dtype=np.uint8)
    edit_data[:] = [0, 0, 255, 255]
    edit_img = Image(edit_data, ImageFormat.RGBA)

    offset_region = Region(Span(90, 20), Span(10, 20))
    layer.add_edit(edit_img, offset_region)

    render_fn = getattr(cr, method)
    result = render_fn(layer)

    assert np.array_equal(result[15, 95], [0, 0, 255, 255])


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
    render_patch(edit.image, np.eye(3), edit.region,
                 warp_mode=WarpMode.PERSPECTIVE)

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
    render_patch(edit.image, np.eye(3), edit.region,
                 warp_mode=WarpMode.PERSPECTIVE)

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
        img, render_region=region, viewport_size=viewport_size, target_size=(
            32, 32)
    )

    assert mask.shape == (32, 32)

    # Valida a área interna (deve ser 255)
    inner_area = mask[16:24, 8:16]
    assert np.all(
        inner_area == 255), "A região mapeada deveria estar opaca (255)!"

    # Valida a área externa (deve ser 0)
    expected_mask = np.zeros((32, 32), dtype=np.uint8)
    expected_mask[16:24, 8:16] = 255

    assert np.array_equal(mask, expected_mask), (
        "A máscara vazou opacidade ou calculou as coordenadas erradas!"
    )


def test_render_scene_culling_no_occlusion(mocker):
    """Caso 1: Sem oclusão (todos os layers são renderizados)"""
    mocker.patch("anicrop.render.BLEND_MODE")
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()

    layers = [make_layer(w=800, h=600), make_layer(w=800, h=600)]
    layers[0].opacity = 1.0
    layers[0].blend_mode = BlendMode.NORMAL
    layers[1].opacity = 1.0
    layers[1].blend_mode = BlendMode.NORMAL

    rendered = []

    def mock_render(layer, plan, interp=InterpolationOption.LANCZOS):
        rendered.append(layer)
        layer._opacity_mask = np.zeros((32, 32), dtype=np.uint8)
        return make_img(w=800, h=600)

    with patch.object(vr, "render_area", side_effect=mock_render):
        vr.render_scene(layers, viewport)

    assert rendered == [layers[0], layers[1]]


def test_render_scene_culling_total_occlusion_top_layer(mocker):
    """Caso 2: Oclusão total pelo layer do topo (índice 0) -> interrompe antes de renderizar o índice 1"""
    mocker.patch("anicrop.render.BLEND_MODE")
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()

    layers = [make_layer(w=800, h=600), make_layer(w=800, h=600)]
    layers[0].opacity = 1.0
    layers[0].blend_mode = BlendMode.NORMAL
    layers[1].opacity = 1.0
    layers[1].blend_mode = BlendMode.NORMAL

    rendered = []

    def mock_render(layer, plan, interp=InterpolationOption.LANCZOS):
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


def test_render_scene_culling_occlusion_middle_layer(mocker):
    """Caso 3: Oclusão pelo layer do meio (índice 1) em pilha de 3 layers (0=Topo, 1=Meio, 2=Fundo)"""
    mocker.patch("anicrop.render.BLEND_MODE")
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

    def mock_render(layer, plan, interp=InterpolationOption.LANCZOS):
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


def test_render_scene_culling_top_layer_opacity_lt_1(mocker):
    """Caso 4: Layer do topo cobre tudo mas tem opacidade < 1.0 -> sem oclusão"""
    mocker.patch("anicrop.render.BLEND_MODE")
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()

    layers = [make_layer(w=800, h=600), make_layer(w=800, h=600)]
    layers[0].opacity = 0.9  # Topo não tem 1.0 de opacidade
    layers[0].blend_mode = BlendMode.NORMAL
    layers[1].opacity = 1.0
    layers[1].blend_mode = BlendMode.NORMAL

    rendered = []

    def mock_render(layer, plan, interp=InterpolationOption.LANCZOS):
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

    final_region = ViewportPlan(logo, viewport).dst_region
    logo_tela_x, logo_tela_y = final_region.top_left

    # Validação 1: O canto superior esquerdo (0,0) da tela DEVE ser Azul!
    assert np.array_equal(data[0, 0], [0, 0, 255, 255]), (
        "Bug: A logo ignorou as transformações e grudou no (0,0)!"
    )

    # Validação 2: A cor Vermelha deve estar exatamente na coordenada transladada
    assert np.array_equal(data[logo_tela_y, logo_tela_x], [255, 0, 0, 255]), (
        "A logo não apareceu na posição correta da tela!"
    )


def test_render_plan_viewport_full_overlap():
    # Viewport de 800x600 sem escala
    viewport = Viewport((800, 600), 1.0)
    # Layer de 200x200 posicionado em (100, 100)
    layer = make_layer(w=200, h=200, x=100, y=100)

    plan = ViewportPlan(layer, viewport)

    # bounds deve ser os limites projetados da camada (400, 300, 200, 200)
    assert plan.bounds == Region(Span(400, 200), Span(300, 200))
    # src_region é a própria imagem por inteiro
    assert plan.src_region == Region(Span(0, 200), Span(0, 200))


def test_edit_renderer_mexican_hat():
    """O Mexicano e seu Chapéu: Testa se um edit adicionado no espaço global após a rotação do Layer cai na posição local correta."""
    from anicrop.layer import Layer
    from anicrop.spatial import Region, Span
    from anicrop.image import Image
    from anicrop.render import render_edit, CanvasRender
    from anicrop.transform import TransformRel
    import numpy as np
    import cv2
    from anicrop.enums import InterpolationOption, ImageFormat

    # 1. Mexicano deitado (Barra horizontal 60x20, centralizada em um Canvas 100x100)
    img_mexican_data = np.zeros((100, 100, 4), dtype=np.uint8)
    img_mexican_data[40:60, 20:80] = [255, 0, 0, 255]  # Pinta o mexicano
    img_mexican = Image(img_mexican_data, ImageFormat.RGBA)

    # Passa a imagem do mexicano direto para ser o Edit 0 (com matriz Identidade original)
    layer = Layer(img_mexican)

    # 2. Gira o layer em -90 graus (como você usou)
    layer.set_transform(TransformRel().rotate(-90))

    # 3. Cria o Chapéu (quadrado xadrez 20x20, 4 cores)
    red = [255, 0, 0, 255]
    green = [0, 255, 0, 255]
    blue = [0, 0, 255, 255]
    yellow = [255, 255, 0, 255]

    img_hat_data = np.zeros((20, 20, 4), dtype=np.uint8)
    img_hat_data[0:10, 0:10] = red      # Top-Left
    img_hat_data[0:10, 10:20] = blue    # Top-Right
    img_hat_data[10:20, 0:10] = yellow  # Bottom-Left
    img_hat_data[10:20, 10:20] = green  # Bottom-Right
    img_hat = Image(img_hat_data, ImageFormat.RGBA)

    # Posiciona no topo central do global (y = 0 a 20, x = 40 a 60)
    layer.add_edit(img_hat, Region(Span(40, 20), Span(0, 20)))

    # 4. Renderiza o Chapéu (Edit 1) individualmente para ver os números
    plan = CanvasPlan(layer, local=True)
    warped_image, dest_region = render_edit(
        layer._edits[1], plan, interp=InterpolationOption.NEAREST)

    # 5. Validações (render_edit retorna a região relativa ao plano)
    assert dest_region.x.start == 80
    assert dest_region.x.length == 20
    assert dest_region.y.start == 40
    assert dest_region.y.length == 20

    # 6. Validação da Rotação Interna (Cores)
    arr = warped_image[...]
    assert np.array_equal(arr[5, 5], yellow)     # TL
    assert np.array_equal(arr[5, 15], red)       # TR
    assert np.array_equal(arr[15, 5], green)     # BL
    assert np.array_equal(arr[15, 15], blue)     # BR


def test_edit_renderer_with_global_render_region_clipping():
    """Valida se o EditRenderer recebe uma render_region global, aplica a matriz inversa do Layer e recorta o edit cirurgicamente."""
    from anicrop.layer import Layer
    from anicrop.spatial import Region, Span
    from anicrop.image import Image
    from anicrop.render import render_edit
    from anicrop.transform import TransformRel
    import numpy as np
    from anicrop.enums import InterpolationOption, ImageFormat

    # 1. Base Layer 100x100
    img_mexican = Image(
        np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = Layer(img_mexican)
    layer.set_transform(TransformRel().rotate(-90))

    # 2. Chapéu 20x20 com 4 cores no Global Y=0..20, X=40..60
    red = [255, 0, 0, 255]
    green = [0, 255, 0, 255]
    blue = [0, 0, 255, 255]
    yellow = [255, 255, 0, 255]

    img_hat_data = np.zeros((20, 20, 4), dtype=np.uint8)
    img_hat_data[0:10, 0:10] = red
    img_hat_data[0:10, 10:20] = blue
    img_hat_data[10:20, 0:10] = yellow
    img_hat_data[10:20, 10:20] = green
    img_hat = Image(img_hat_data, ImageFormat.RGBA)

    layer.add_edit(img_hat, Region(Span(40, 20), Span(0, 20)))

    # 3. Solicitamos a renderização passando o CanvasPlan(local=True) com a região global (Y=0..10)
    global_render_region = Region(Span(40, 20), Span(0, 10))
    plan = CanvasPlan(layer, view_region=global_render_region, local=True)

    result = render_edit(
        layer._edits[1],
        plan=plan,
        interp=InterpolationOption.NEAREST
    )

    assert result is not None
    warped_image, dest_region = result

    # A rotação de -90° converte Global Y=0..10 para Local X=90..100!
    # A dest_region relativa ao plan.dst_region (que começa em X=90, Y=40) tem inicio em X=0, Y=0!
    assert dest_region.x.start == 0
    assert dest_region.x.length == 10
    assert dest_region.y.start == 0
    assert dest_region.y.length == 20

    # A imagem recortada resultante deve ter tamanho 20x10 (altura 20, largura 10)
    assert warped_image.width == 10
    assert warped_image.height == 20


def test_render_plan_viewport_partial_overlap():
    # Viewport de 800x600 sem escala
    viewport = Viewport((800, 600), 1.0)
    # Layer de 200x200 posicionado em (-400, -300) para projetar na borda da tela (-100, -100)
    layer = make_layer(w=200, h=200, x=-400, y=-300)

    plan = ViewportPlan(layer, viewport)

    # bounds projetado
    assert plan.bounds == Region(Span(-100, 200), Span(-100, 200))
    # dst_region visível na tela (0, 0, 100, 100)
    assert plan.dst_region == Region(Span(0, 100), Span(0, 100))
    # src_region de crop deve capturar o pedaço inferior direito da camada (100, 100, 100, 100)
    assert plan.src_region == Region(Span(100, 100), Span(100, 100))


def test_render_plan_canvas_space_partial_overlap():
    # Layer de 200x200 posicionado no canvas em (50, 50)
    layer = make_layer(w=200, h=200, x=50, y=50)
    # Região de visualização arbitrária (0, 0, 100, 100)
    view_region = Region(Span(0, 100), Span(0, 100))

    # Sem viewport, calculando no espaço do canvas
    plan = CanvasPlan(layer, view_region=view_region)

    # bounds deve ser a região global da camada no canvas (50, 50, 200, 200)
    assert plan.bounds == Region(Span(50, 200), Span(50, 200))
    # dst_region visível na interseção (50, 50, 50, 50)
    assert plan.dst_region == Region(Span(50, 50), Span(50, 50))
    # src_region do crop na coordenada local da camada (0, 0, 50, 50)
    assert plan.src_region == Region(Span(0, 50), Span(0, 50))


def test_render_plan_full_render_no_clipping():
    # Layer de 200x200 posicionado no canvas em (150, 250)
    layer = make_layer(w=200, h=200, x=150, y=250)

    # Sem viewport e sem view_region (renderização completa da camada)
    plan = CanvasPlan(layer)

    # bounds deve ser a região global no canvas (150, 250, 200, 200)
    assert plan.bounds == Region(Span(150, 200), Span(250, 200))
    # Sem cortes, dst_region é exatamente a região total (bounds)
    assert plan.dst_region == plan.bounds
    # O crop src deve cobrir todo o buffer da camada (0, 0, 200, 200)
    assert plan.src_region == Region(Span(0, 200), Span(0, 200))


def test_canvas_plan_local_state():
    """Valida se CanvasPlan com local=True calcula os bounds no espaço local (0,0,W,H) e converte a view_region global via matriz inversa."""
    img_mexican = Image(
        np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = Layer(img_mexican)
    layer.set_transform(TransformRel().rotate(-90))

    # Região Global Y=0..10 no topo da tela
    global_view_region = Region(Span(40, 20), Span(0, 10))

    plan = CanvasPlan(layer, view_region=global_view_region, local=True)

    # 1. Bounds no espaço local deve ser o tamanho original unrotated (0, 0, 100, 100)
    assert plan.bounds == Region(Span(0, 100), Span(0, 100))

    # 2. A rotação -90° converte Global Y=0..10 para Local X=90..100
    assert plan.dst_region == Region(Span(90, 10), Span(40, 20))
    assert plan.src_region == Region(Span(90, 10), Span(40, 20))


def test_edit_renderer_render_final_mexican_hat_full():
    """Valida se render_final projeta o EditLayer diretamente no espaço global/tela na posição e orientação finais esperadas."""
    # 1. Layer base 100x100 rotacionado em -90°
    img_mexican = Image(
        np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = Layer(img_mexican)
    layer.set_transform(TransformRel().rotate(-90))

    # 2. Chapéu 20x20 xadrez com 4 cores no Global Y=0..20, X=40..60
    red = [255, 0, 0, 255]
    green = [0, 255, 0, 255]
    blue = [0, 0, 255, 255]
    yellow = [255, 255, 0, 255]

    img_hat_data = np.zeros((20, 20, 4), dtype=np.uint8)
    img_hat_data[0:10, 0:10] = red
    img_hat_data[0:10, 10:20] = blue
    img_hat_data[10:20, 0:10] = yellow
    img_hat_data[10:20, 10:20] = green
    img_hat = Image(img_hat_data, ImageFormat.RGBA)

    layer.add_edit(img_hat, Region(Span(40, 20), Span(0, 20)))

    # 3. Executa o render com CanvasPlan configurado para global (local=False)
    plan = CanvasPlan(layer, local=False)
    result = render_edit(
        layer._edits[1],
        plan=plan,
        interp=InterpolationOption.NEAREST
    )

    assert result is not None
    warped_image, dest_region = result

    # No espaço Global final da tela, a dest_region relativa ao plan.dst_region (top_left=0,0) é (40, 0, 20, 20)
    assert dest_region.x.start == 40
    assert dest_region.x.length == 20
    assert dest_region.y.start == 0
    assert dest_region.y.length == 20

    # As rotações do layer (-90°) e da matriz local (+90°) se anulam na tela global.
    arr = warped_image[...]
    assert np.array_equal(arr[5, 5], red)        # Top-Left
    assert np.array_equal(arr[5, 15], blue)      # Top-Right
    assert np.array_equal(arr[15, 5], yellow)    # Bottom-Left
    assert np.array_equal(arr[15, 15], green)    # Bottom-Right


def test_edit_renderer_render_final_mexican_hat_with_clipping():
    """Valida se render com CanvasPlan(local=False) e render_region global recorta o EditLayer diretamente no espaço global da tela."""
    # 1. Layer base 100x100 rotacionado em -90°
    img_mexican = Image(
        np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = Layer(img_mexican)
    layer.set_transform(TransformRel().rotate(-90))

    # 2. Chapéu 20x20 xadrez com 4 cores no Global Y=0..20, X=40..60
    red = [255, 0, 0, 255]
    green = [0, 255, 0, 255]
    blue = [0, 0, 255, 255]
    yellow = [255, 255, 0, 255]

    img_hat_data = np.zeros((20, 20, 4), dtype=np.uint8)
    img_hat_data[0:10, 0:10] = red
    img_hat_data[0:10, 10:20] = blue
    img_hat_data[10:20, 0:10] = yellow
    img_hat_data[10:20, 10:20] = green
    img_hat = Image(img_hat_data, ImageFormat.RGBA)

    layer.add_edit(img_hat, Region(Span(40, 20), Span(0, 20)))

    # 3. Solicita render com CanvasPlan(local=False) para a metade superior da tela (Global Y=0..10, X=40..60)
    global_render_region = Region(Span(40, 20), Span(0, 10))
    plan = CanvasPlan(layer, view_region=global_render_region, local=False)

    result = render_edit(
        layer._edits[1],
        plan=plan,
        interp=InterpolationOption.NEAREST
    )

    assert result is not None
    warped_image, dest_region = result

    # dest_region relativa ao plan.dst_region (que começa em 40, 0) é (0, 0, 20, 10)
    assert dest_region.x.start == 0
    assert dest_region.x.length == 20
    assert dest_region.y.start == 0
    assert dest_region.y.length == 10

    # Imagem recortada deve ter largura 20 e altura 10
    assert warped_image.width == 20
    assert warped_image.height == 10

    # Cores da metade de cima da tela: Red no Top-Left, Blue no Top-Right
    arr = warped_image[...]
    assert np.array_equal(arr[5, 5], red)
    assert np.array_equal(arr[5, 15], blue)


def test_edit_renderer_with_viewport_plan():
    """Valida se o EditRenderer.render funciona polimorficamente com ViewportPlan (projeção direta na tela da Viewport)."""
    img_mexican = Image(
        np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = Layer(img_mexican)
    layer.set_transform(TransformRel().rotate(-90))

    img_hat = Image(np.zeros((20, 20, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer.add_edit(img_hat, Region(Span(40, 20), Span(0, 20)))

    viewport = Viewport((800, 600), 1.0)
    plan = ViewportPlan(layer, viewport)

    result = render_edit(
        layer._edits[1], plan=plan, interp=InterpolationOption.NEAREST)

    assert result is not None
    warped_image, dest_region = result
    # render_edit retorna a dest_region relativa ao plan.dst_region da camada na tela: (390-350, 250-250) = (40, 0, 20, 20)
    assert dest_region == Region(Span(40, 20), Span(0, 20))


def test_viewport_plan_local_state():
    """Valida se ViewportPlan com local=True projeta o Layer unrotated (Mexicano Deitado) na tela da Viewport com a Câmera."""
    img_mexican = Image(
        np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = Layer(img_mexican)
    layer.set_transform(TransformRel().rotate(-90))

    img_hat = Image(np.zeros((20, 20, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer.add_edit(img_hat, Region(Span(40, 20), Span(0, 20)))

    viewport = Viewport((800, 600), 1.0)
    plan = ViewportPlan(layer, viewport, local=True)

    result = render_edit(
        layer._edits[1], plan=plan, interp=InterpolationOption.NEAREST)

    assert result is not None
    warped_image, dest_region = result

    # render_edit retorna a dest_region relativa ao plan.dst_region local na tela: (430-350, 290-250) = (80, 40, 20, 20)
    assert dest_region == Region(Span(80, 20), Span(40, 20))


def test_render_image_direto():
    """Testa a função atômica render_image diretamente com uma Image e um CanvasPlan."""
    img = make_img(w=50, h=50, color=(0, 255, 0, 255))
    layer = make_layer(w=50, h=50, color=(0, 255, 0, 255))
    plan = CanvasPlan(layer)
    m_local = np.identity(3, dtype=np.float32)

    result = render_image(img, plan, m_local, interp=InterpolationOption.NEAREST)
    assert result is not None
    warped_image, dest_region = result
    assert warped_image.width == 50
    assert warped_image.height == 50
    assert dest_region == Region(Span(0, 50), Span(0, 50))
