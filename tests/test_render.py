from anicrop.enums import InterpolationOption, RenderDirty
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.render import LayerRender
from anicrop.spatial import Region, Span
from anicrop.transform import mat_final
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
def make_img(w=100, h=100, color=(255, 0, 0, 255)):
    # Gera uma imagem com uma cor sólida
    img_data = np.zeros((h, w, 4), dtype=np.uint8)
    img_data[:] = color
    return Image(img_data, ImageFormat.RGBA)


def make_layer(w=100, h=100, x=0, y=0, color=(255, 0, 0, 255)):
    img = make_img(w, h, color)
    layer = Layer(img)
    if x != 0:
        layer.x = x
    if y != 0:
        layer.y = y
    return layer


@pytest.mark.parametrize(
    'method',
    ['render', 'render_area']
)
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
    np.testing.assert_array_equal(
        rendered_image[...], original_layer.image[...])


@pytest.mark.parametrize(
    'method',
    ['render', 'render_area']
)
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
    bbox = layer.canvas_region

    assert rendered_image.width == bbox.width
    assert rendered_image.height == bbox.height

    img_array = rendered_image[...]
    centro_x, centro_y = bbox.width // 2, bbox.height // 2
    pixel_central = img_array[centro_y, centro_x]

    assert pixel_central[3] == 255
    np.testing.assert_array_equal(pixel_central, cor_original)

    pixel_canto = img_array[0, 0]
    assert pixel_canto[3] == 0


@pytest.mark.parametrize(
    'method',
    ['render', 'render_area']
)
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


@pytest.mark.parametrize(
    'method',
    ['render', 'render_area']
)
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

    m_final = mat_final(layer, *layer.canvas_region.top_left)
    edit_obj = layer._edits[1]
    full_edit_matrix = m_final @ edit_obj.local_matrix

    p = full_edit_matrix @ np.array([5, 5, 1.0])
    cx, cy = int(round(p[0])), int(round(p[1]))

    pixel = data[cy, cx]
    assert pixel[0] > 240
    assert pixel[2] < 10


@pytest.mark.parametrize(
    'method',
    ['render', 'render_area']
)
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


@pytest.mark.parametrize(
    'method',
    ['render', 'render_area']
)
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


@pytest.mark.parametrize(
    'method',
    ['render', 'render_area']
)
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


@pytest.mark.parametrize(
    'method',
    ['render', 'render_area']
)
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
        LayerRender, '_LayerRender__flatten_edits', return_value=mock_img)

    lr.render(layer)

    assert mock_flatten.call_count == 1
    assert layer._id in lr._cache


def test_render_cache_reaproveita_na_translacao(mocker, lr):
    """Garante que translação pura NÃO dispara um novo __flatten_edits se houver cache completo."""
    layer = make_layer()
    mock_img = Image.new(layer.region.size, layer.format)
    mock_flatten = mocker.patch.object(
        LayerRender, '_LayerRender__flatten_edits', return_value=mock_img)

    # 1. Renderização Completa para popular o cache mestre
    lr.render(layer)
    assert mock_flatten.call_count == 1

    # 2. Translação (Deve dar cache hit nos pixels)
    layer.x += 50
    layer.y += 50

    result = lr.render(layer)

    assert mock_flatten.call_count == 1, "ERRO: __flatten_edits foi chamado desnecessariamente!"
    assert result is not None


def test_render_cache_reprocessa_na_distorcao(mocker, lr):
    """Garante que rotação/escala DISPARA um novo __flatten_edits."""
    layer = make_layer()
    mock_img = Image.new(layer.region.size, layer.format)
    mock_flatten = mocker.patch.object(
        LayerRender, '_LayerRender__flatten_edits', return_value=mock_img)

    # 1. Primeiro render completo
    lr.render(layer)

    # 2. Rotação (Invalida pixels)
    layer.rotation = 45
    lr.render(layer)

    assert mock_flatten.call_count == 2, "ERRO: __flatten_edits deveria ter sido chamado!"


def test_render_cache_recorte_de_area(mocker, lr):
    """Valida se o render_area usa o cache mestre para recortes."""
    layer = make_layer(w=100, h=100)
    full_img = Image.new((100, 100), layer.format, color=(255, 0, 0, 255))
    
    # Simula cache populado via render
    lr._cache[layer._id] = full_img
    layer._commit_render_state()

    view_region = Region(Span(10, 20), Span(10, 20))
    mocker.patch.object(LayerRender, '_LayerRender__render_region',
                        return_value=view_region)
    mock_flatten = mocker.patch.object(LayerRender, '_LayerRender__flatten_edits')

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
    mocker.patch.object(LayerRender, '_LayerRender__render_region', return_value=view_region)
    
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
    mocker.patch.object(LayerRender, '_LayerRender__render_region',
                        return_value=view_region)
    mocker.patch.object(LayerRender, '_LayerRender__flatten_edits')

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
        LayerRender, '_LayerRender__flatten_edits', return_value=mock_img)
    
    # 1. Primeira chamada ao render_area (ele renderiza, mas NÃO popula o self._cache)
    result_first_call = lr.render_area(layer)
    assert result_first_call is not None # Esperamos uma imagem aqui
    assert mock_flatten.call_count == 1
    assert layer._id not in lr._cache    # O cache não deve ter sido populado
    
    # 2. O layer é 'commitado'. As flags ficam NONE.
    layer._commit_render_state()
    assert layer._resolve_dirty() == RenderDirty.NONE
    
    # 3. Segunda chamada ao render_area.
    # flags & RenderDirty.PIXELS é FALSO.
    # layer._id not in lr._cache é VERDADEIRO (ainda).
    # Então __flatten_edits será chamado NOVAMENTE.
    result_second_call = lr.render_area(layer)
    
    assert result_second_call is not None
    assert mock_flatten.call_count == 2, "BUG: __flatten_edits foi chamado novamente mesmo com flags limpas e sem alteração!"
