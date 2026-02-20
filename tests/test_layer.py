from anicrop.image import Image, ImageFormat
from anicrop.layer import BlendMode, EditLayer, Layer
from anicrop.layer import Rotation, Scale
from anicrop.spatial import Region
from pytest import raises
import numpy as np
import pytest

W = H = 10  # Tamanhos padrão do canvas


def make_region(w=3, h=3, offset=0):
    return Region.from_size(w, h) + offset


def make_canvas(w=W, h=H, channel=4):
    channels = {
        1: ImageFormat.GRAY, 2: ImageFormat.GRAY_ALPHA, 3: ImageFormat.RGB,
        4: ImageFormat.RGBA, -4: ImageFormat.CMYK, 5: ImageFormat.CMYK_ALPHA
    }
    return Image(np.zeros((h, w, abs(channel)), dtype=np.uint8), channels.get(channel))


@pytest.fixture
def canvas():
    return make_canvas()


@pytest.fixture
def identity_matrix():
    return np.eye(3, dtype=np.float32)


# Novos testes para EditLayer (API atualizada)
def test_EditLayer_inicializacao(canvas, identity_matrix):
    region = make_region()
    edit = EditLayer(canvas, region, identity_matrix, name="Custom")

    assert edit.name == "Custom"
    assert edit.region == region
    assert np.array_equal(edit.matrix, identity_matrix)
    assert edit.blend_mode == BlendMode.NORMAL
    assert edit.image is canvas

# ############################# Teste da classe Layer (Originais) #############################################


def test_Layer_inicializando_com_valores_padroes(canvas):
    layer = Layer(canvas)
    assert layer.name == 'Layer'
    assert layer.opacity == 1.0
    assert layer.rotation == Rotation()
    assert layer.scale == Scale(1.0, 1.0)
    assert layer.blend_mode == BlendMode.NORMAL
    assert layer.region == Region.from_size(10, 10)
    assert np.array_equal(layer.image[...], canvas[...])


def test_Layer_inicializando_com_parametros(canvas):
    layer = Layer(canvas, opacity=0.8, rotation=45, scale=0.5, blend_mode=BlendMode.MULTIPLY, name='Picture')
    assert layer.name == 'Picture'
    assert layer.opacity == 0.8
    assert layer.rotation == Rotation(45)
    assert layer.scale == Scale(0.5, 0.5)
    assert layer.blend_mode == BlendMode.MULTIPLY
    assert layer.region == Region.from_size(10, 10)
    assert np.array_equal(layer.image[...], canvas[...])


def test_Layer_rotate_mudanca_valor(canvas):
    layer = Layer(canvas)
    layer.rotation = 45
    assert layer.rotation == Rotation(45)

    layer.rotation -= 20
    assert layer.rotation == Rotation(25)


def test_Layer_scale_mudanca_valor(canvas):
    layer = Layer(canvas)
    layer.scale = 0.5
    assert layer.scale == Scale(0.5, 0.5)

    layer.scale += 0.3
    assert layer.scale == Scale(0.8, 0.8)


def test_Layer_opacity_mudanca_valor(canvas):
    layer = Layer(canvas)
    layer.opacity = 0.1
    assert layer.opacity == 0.1

    layer.opacity += 0.8
    assert layer.opacity == 0.9


def test_Layer_blend_mode_mudanca_valor(canvas):
    layer = Layer(canvas)
    layer.blend_mode = BlendMode.MULTIPLY
    assert layer.blend_mode == BlendMode.MULTIPLY


def test_Layer_region_mudanca_valor(canvas):
    layer = Layer(canvas)
    layer.region += (4, 0)
    assert layer.region == Region.from_size(H, W) + (4, 0)


def test_Layer_atribuindo_valor_qualquer_ao_region(canvas):
    with raises(TypeError, match="Expected Region, got int"):
        layer = Layer(canvas)
        layer.region = 4


def test_Layer_name_mudanca_valor(canvas):
    layer = Layer(canvas)
    layer.name = 'Picture'
    assert layer.name == 'Picture'


# ############################# Testes para add_edit (Novos) #####################################

def test_Layer_add_edit_cria_e_adiciona_edit_layer(canvas):
    layer = Layer(canvas)
    edit_image = make_canvas(w=5, h=5)
    region = make_region(w=5, h=5)

    layer.add_edit(edit_image, region)

    assert len(layer._edits) == 1
    edit = layer._edits[0]

    assert isinstance(edit, EditLayer)
    assert edit.image is edit_image
    assert edit.region == region
    assert edit.name == "Edit-1"
    # Verifica se a matriz foi calculada e armazenada
    assert isinstance(edit.matrix, np.ndarray)


def test_Layer_add_edit_calcula_inversa_corretamente(canvas):
    layer = Layer(canvas)
    # Acessando propriedades do Layer para transformar
    # Assumindo que x/y usam setters que atualizam a região internamente
    layer.x = 10
    layer.y = 20

    region = make_region()
    layer.add_edit(canvas, region)

    edit = layer._edits[0]

    # Matriz global esperada: Translação(10, 20)
    # Inversa esperada: Translação(-10, -20)
    expected_inv = np.array([
        [1., 0., -10.],
        [0., 1., -20.],
        [0., 0., 1.]
    ], dtype=np.float32)

    np.testing.assert_array_almost_equal(edit.matrix, expected_inv)


def test_Layer_add_edit_usa_blend_mode_passado(canvas):
    layer = Layer(canvas)
    layer.add_edit(canvas, make_region(), blend_mode=BlendMode.MULTIPLY)
    assert layer._edits[0].blend_mode == BlendMode.MULTIPLY


def test_Layer_add_edit_incrementa_nomes(canvas):
    layer = Layer(canvas)
    layer.add_edit(canvas, make_region())
    layer.add_edit(canvas, make_region())

    assert layer._edits[0].name == "Edit-1"
    assert layer._edits[1].name == "Edit-2"
