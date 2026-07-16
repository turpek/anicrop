from anicrop.image import Image, ImageFormat
from anicrop.layer import BlendMode, EditLayer, Layer
from anicrop.layer import Rotation, Scale
from anicrop.spatial import Region
from anicrop.enums import RenderFlags, WarpMode
from anicrop.transform import mat_global, Transform
from pytest import raises
import numpy as np
import pytest

W = H = 10  # Tamanhos padrão do image


def make_region(w=3, h=3, offset=0):
    return Region.from_size(w, h) + offset


def make_image(w=W, h=H, channel=4, color=None):
    channels = {
        1: ImageFormat.GRAY, 2: ImageFormat.GRAY_ALPHA, 3: ImageFormat.RGB,
        4: ImageFormat.RGBA, -4: ImageFormat.CMYK, 5: ImageFormat.CMYK_ALPHA
    }
    if color:
        return Image.new((h, w), channels.get(channel), color=color)
    return Image.new((h, w), channels.get(channel))


@pytest.fixture
def image():
    return make_image()


@pytest.fixture
def identity_matrix():
    return np.eye(3, dtype=np.float32)


# Novos testes para EditLayer (API atualizada)
def test_EditLayer_inicializacao(image, identity_matrix):
    region = make_region()
    edit = EditLayer(image, region, identity_matrix, name="Custom")

    assert edit.name == "Custom"
    assert edit.region == region
    assert np.array_equal(edit.matrix, identity_matrix)
    assert edit.blend_mode == BlendMode.NORMAL
    assert edit.image is image

# ############################# Teste da classe Layer (Originais) #############################################


def test_Layer_inicializando_com_valores_padroes(image):
    layer = Layer(image)
    assert layer.name == 'Layer'
    assert layer.opacity == 1.0
    assert layer.rotation == Rotation()
    assert layer.scale == Scale(1.0, 1.0)
    assert layer.blend_mode == BlendMode.NORMAL
    assert layer.region == Region.from_size(10, 10)
    assert np.array_equal(layer.image[...], image[...])


def test_Layer_inicializando_com_parametros(image):
    layer = Layer(image, opacity=0.8, rotation=45, scale=0.5,
                  blend_mode=BlendMode.MULTIPLY, name='Picture')
    assert layer.name == 'Picture'
    assert layer.opacity == 0.8
    assert layer.rotation == Rotation(45)
    assert layer.scale == Scale(0.5, 0.5)
    assert layer.blend_mode == BlendMode.MULTIPLY
    assert layer.region == Region.from_size(10, 10)
    assert np.array_equal(layer.image[...], image[...])


def test_Layer_rotate_mudanca_valor(image):
    layer = Layer(image)
    layer.rotation = 45
    assert layer.rotation == Rotation(45)

    layer.rotation -= 20
    assert layer.rotation == Rotation(25)


def test_Layer_scale_mudanca_valor(image):
    layer = Layer(image)
    layer.scale = 0.5
    assert layer.scale == Scale(0.5, 0.5)

    layer.scale += 0.3
    assert layer.scale == Scale(0.8, 0.8)


def test_Layer_opacity_mudanca_valor(image):
    layer = Layer(image)
    layer.opacity = 0.1
    assert layer.opacity == 0.1

    layer.opacity += 0.8
    assert layer.opacity == 0.9


def test_Layer_blend_mode_mudanca_valor(image):
    layer = Layer(image)
    layer.blend_mode = BlendMode.MULTIPLY
    assert layer.blend_mode == BlendMode.MULTIPLY


def test_Layer_region_mudanca_valor(image):
    layer = Layer(image)
    layer.region += (4, 0)
    assert layer.region == Region.from_size(H, W) + (4, 0)


def test_Layer_atribuindo_valor_qualquer_ao_region(image):
    with raises(TypeError, match="Expected Region, got int"):
        layer = Layer(image)
        layer.region = 4


def test_Layer_name_mudanca_valor(image):
    layer = Layer(image)
    layer.name = 'Picture'
    assert layer.name == 'Picture'


# ############################# Testes para add_edit (Novos) #####################################

def test_Layer_add_edit_cria_e_adiciona_edit_layer(image):
    layer = Layer(image)
    edit_image = make_image(w=5, h=5)
    region = make_region(w=5, h=5)

    layer.add_edit(edit_image, region)

    assert len(layer._edits) == 2
    edit = layer._edits[1]

    assert isinstance(edit, EditLayer)
    assert edit.image is edit_image
    assert edit.region == region
    assert edit.name == "Edit-2"
    # Verifica se a matriz foi calculada e armazenada
    assert isinstance(edit.matrix, np.ndarray)


def test_Layer_add_edit_calcula_inversa_corretamente(image):
    layer = Layer(image)
    # Acessando propriedades do Layer para transformar
    # Assumindo que x/y usam setters que atualizam a região internamente
    layer.x = 10
    layer.y = 20

    region = make_region()
    layer.add_edit(image, region)

    edit = layer._edits[1]

    # Matriz global esperada: Translação(10, 20)
    # Inversa esperada: Translação(-10, -20)
    expected_inv = np.array([
        [1., 0., -10.],
        [0., 1., -20.],
        [0., 0., 1.]
    ], dtype=np.float32)

    np.testing.assert_array_almost_equal(edit.matrix, expected_inv)


def test_Layer_add_edit_usa_blend_mode_passado(image):
    layer = Layer(image)
    layer.add_edit(image, make_region(), blend_mode=BlendMode.MULTIPLY)
    assert layer._edits[1].blend_mode == BlendMode.MULTIPLY


def test_Layer_add_edit_incrementa_nomes(image):
    layer = Layer(image)
    layer.add_edit(image, make_region())
    layer.add_edit(image, make_region())

    assert layer._edits[0].name == "Edit-1"
    assert layer._edits[1].name == "Edit-2"


# ############################# Testes de Invalidação de Cache (TDD) #####################################

def test_layer_cache_initial_state(image):
    """Cenário 1: Layer recém-instanciado deve retornar ALL."""
    layer = Layer(image)
    assert layer._resolve_render() & RenderFlags.ALL_DIRTY


@pytest.mark.parametrize("update_fn", [
    lambda layer: setattr(layer, 'x', layer.x.start + 10),
    lambda layer: setattr(layer, 'y', layer.y.start + 10),
    lambda layer: setattr(layer, 'region', layer.region + (5, 5)),
    lambda layer: layer.transform.translate(10, 10),
    lambda layer: layer.set_transform(Transform().translate(10, 10))
], ids=["set_x", "set_y", "set_reg", "tr_trans", "st_trans"])
def test_layer_cache_translation(image, update_fn):
    """Cenário 2: Translação pura deve retornar POSITION."""
    layer = Layer(image)
    layer._commit_render_state()

    update_fn(layer)
    assert layer._resolve_render() & RenderFlags.POSITION


@pytest.mark.parametrize("update_fn", [
    lambda layer: setattr(layer, 'rotation', 45),
    lambda layer: layer.transform.rotate(45),
    lambda layer: layer.set_transform(Transform().rotate(45))
], ids=["set_rot", "tr_rot", "st_rot"])
def test_layer_cache_rotation(image, update_fn):
    """Cenário 3: Rotação deve retornar PIXELS."""
    layer = Layer(image)
    layer._commit_render_state()

    update_fn(layer)
    assert layer._resolve_render() & RenderFlags.PIXELS


@pytest.mark.parametrize("update_fn", [
    lambda layer: setattr(layer, 'scale', 2.0),
    lambda layer: layer.transform.scale(2.0, 2.0),
    lambda layer: layer.set_transform(Transform().scale(2.0, 2.0))
], ids=["set_scal", "tr_scal", "st_scal"])
def test_layer_cache_scale(image, update_fn):
    """Cenário 4: Escala deve retornar PIXELS."""
    layer = Layer(image)
    layer._commit_render_state()

    update_fn(layer)
    assert layer._resolve_render() & RenderFlags.PIXELS


def test_layer_commit_render_state(image):
    """Cenário 5: Commit deve limpar o estado dirty e salvar a matriz."""
    layer = Layer(image)
    assert layer._resolve_render() & RenderFlags.ALL_DIRTY

    matrix = mat_global(layer)
    layer._commit_render_state()

    assert layer._resolve_render() == RenderFlags.NONE
    assert np.array_equal(layer._old_matrix, matrix)

    # Gera uma nova deformação (rotação)
    layer.rotation = 90
    assert layer._resolve_render() & RenderFlags.PIXELS

    new_matrix = mat_global(layer)
    layer._commit_render_state()

    assert layer._resolve_render() == RenderFlags.NONE
    assert np.array_equal(layer._old_matrix, new_matrix)


def test_layer_resolve_dirty_nao_deve_ser_acumulativo(image):
    """
    Verifica se o _resolve_dirty reflete apenas as mudanças desde o último commit,
    ou se ele 'esquece' de limpar flags antigas se chamado múltiplas vezes.
    """
    layer = Layer(image)
    layer._commit_render_state()

    # 1. Suja apenas a posição
    layer.x += 10
    assert layer._resolve_render() == RenderFlags.POSITION

    # 2. Se chamarmos de novo SEM commit, ele ainda deve ser POSITION
    assert layer._resolve_render() == RenderFlags.POSITION

    # 3. Faz o commit. Agora deve ser NONE.
    layer._commit_render_state()
    assert layer._resolve_render() == RenderFlags.NONE


# ############################# Testes de Modo de Projeção (WarpMode) #####################################

def test_layer_warp_mode_is_affine_even_with_translation(mocker, image):
    """Garante que translação pura ainda utiliza o motor AFFINE (Afim)."""
    # Matriz com Translação forte, mas última linha intacta [0, 0, 1]
    matrix = np.eye(3)
    matrix[0, 2] = 500.0
    matrix[1, 2] = 300.0
    mocker.patch("anicrop.layer.mat_global", return_value=matrix)

    layer = Layer(image)
    layer._resolve_render()

    # Deve ser AFFINE pois a última linha é [0, 0, 1]
    assert layer._warp_mode == WarpMode.AFFINE


def test_layer_warp_mode_perspective_triggered_by_z_line(mocker, image):
    """Garante que PERSPECTIVE é disparado apenas pela deformação da última linha (Z)."""
    # Matriz sem translação, mas com deformação de perspectiva na última linha
    matrix = np.eye(3)
    matrix[2, 0] = 0.0005  # Componente de perspectiva
    mocker.patch("anicrop.layer.mat_global", return_value=matrix)

    layer = Layer(image)
    layer._resolve_render()

    # Agora sim o esperado é PERSPECTIVE
    assert layer._warp_mode == WarpMode.PERSPECTIVE


def test_layer_canvas_size_without_canvas(image):
    """Cenário 1: canvas_size sem passar image deve ser o tamanho da imagem."""
    layer = Layer(image)  # image fixture size is (10, 10)
    assert layer.canvas_size == (10, 10)


def test_layer_canvas_size_with_canvas(mocker, image):
    """Cenário 2: canvas_size passando o image deve ser o tamanho do image."""
    mock_canvas = mocker.MagicMock()
    mock_canvas.size = (1920, 1080)

    layer = Layer(image, canvas=mock_canvas)
    assert layer.canvas_size == (1920, 1080)
