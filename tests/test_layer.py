from anicrop.image import Image, ImageFormat
from anicrop.layer import BlendMode, EditLayer, Layer
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


def test_EditLayer_inicializando_com_valores_padroes():
    img = Image(np.zeros((10, 10, 3), dtype=np.uint8), ImageFormat.RGB)
    edit = EditLayer(img)
    assert edit.name == 'Edit'
    assert edit.opacity == 1.0
    assert edit.rotate == 0.0
    assert edit.scale == 1.0
    assert edit.blend_mode == BlendMode.NORMAL
    assert edit.region == Region.from_size(10, 10)
    assert np.array_equal(edit.image[...], img[...])


def test_EditLayer_inicializando_com_parametros(canvas):
    region = Region.from_size(5, 3) + (2, 5)
    canvas[region] = 1
    edit = EditLayer(canvas, opacity=0.8, blend_mode=BlendMode.MULTIPLY, name='Paint')
    assert edit.name == 'Paint'
    assert edit.opacity == 0.8
    assert edit.blend_mode == BlendMode.MULTIPLY
    assert edit.region == region
    assert np.array_equal(edit.image[...], canvas[region])


@pytest.mark.parametrize(
    "canvas",
    [make_canvas(channel=2), make_canvas(channel=4), make_canvas(channel=5)],
    ids=["Gray+Alpha", "RGBA", "CMYK+Alpha"]
)
def test_EditLayer_com_alpha_e_sem_conteudo(canvas):
    with raises(ValueError, match="EditLayer cannot be created from a fully transparent image."):
        EditLayer(canvas)


@pytest.mark.parametrize(
    "canvas, region",
    [
        (make_canvas(channel=2), make_region(w=4, offset=2)),
        (make_canvas(channel=4), make_region(h=4, offset=2)),
        (make_canvas(channel=5), make_region(offset=2))
    ],
    ids=["Gray+Alpha", "RGBA", "CMYK+Alpha"]
)
def test_EditLayer_com_alpha_e_com_conteudo(canvas, region):
    canvas[region] = 1
    edit = EditLayer(canvas)
    assert edit.region == region
    assert np.array_equal(edit.image[...], canvas[region])


@pytest.mark.parametrize(
    "canvas, region",
    [
        (make_canvas(channel=1), make_region(w=4, offset=2)),
        (make_canvas(channel=3), make_region(h=4, offset=2)),
        (make_canvas(channel=-4), make_region(offset=2))  # canal -4 -> CMYK
    ],
    ids=["Gray", "RGB", "CMYK"]
)
def test_EditLayer_sem_canal_alpha(canvas, region):
    canvas[region] = 1
    bbox = Region.from_size(W, H)
    edit = EditLayer(canvas)
    assert edit.region == bbox
    assert np.array_equal(edit.image[...], canvas[bbox])


def test_EditLayer_image_interna_eh_copia():
    canvas = make_canvas()
    region = make_region(offset=2)
    canvas[region] = 255

    edit = EditLayer(canvas)
    # modifica o interno
    edit.image[..., 0] = 1

    # a região original no canvas não deve ser afetada
    assert not np.array_equal(canvas[edit.region], edit.image[...])


def test_EditLayer_bbox_eh_um_unico_pixel():
    canvas = make_canvas(channel=4)
    canvas[3, 2] = [255, 0, 0, 255]  # pixel (x=2, y=3)
    edit = EditLayer(canvas)
    assert edit.region == Region.from_size(1, 1) + (2, 3)
    np.testing.assert_array_equal(edit.image[...], canvas[edit.region])


def test_EditLayer_bbox_toca_as_bordas():
    canvas = make_canvas(channel=4)
    canvas[0, 0] = [1, 1, 1, 255]
    canvas[H - 1, W - 1] = [1, 1, 1, 255]
    edit = EditLayer(canvas)
    assert edit.region == Region.from_size(W, H)


def test_EditLayer_rotate_mudanca_valor(canvas):
    canvas[...] = 1
    edit = EditLayer(canvas)
    edit.rotate = 45
    assert edit.rotate == 45

    edit.rotate -= 20
    assert edit.rotate == 25


def test_EditLayer_scale_mudanca_valor(canvas):
    canvas[...] = 1
    edit = EditLayer(canvas)
    edit.scale = 0.5
    assert edit.scale == 0.5

    edit.scale += 0.3
    assert edit.scale == 0.8


def test_EditLayer_region_mudanca_valor(canvas):
    canvas[...] = 1
    edit = EditLayer(canvas)
    edit.region += (4, 0)
    assert edit.region == Region.from_size(H, W) + (4, 0)


def test_EditLayer_atribuindo_valor_qualquer_ao_region(canvas):
    canvas[...] = 1
    with raises(TypeError, match="Expected Region, got int"):
        edit = EditLayer(canvas)
        edit.region = 4

# ############################# Teste da classe Layer #############################################


def test_Layer_inicializando_com_valores_padroes(canvas):
    layer = Layer(canvas)
    assert layer.name == 'Layer'
    assert layer.opacity == 1.0
    assert layer.rotate == 0.0
    assert layer.scale == 1.0
    assert layer.blend_mode == BlendMode.NORMAL
    assert layer.region == Region.from_size(10, 10)
    assert np.array_equal(layer.image[...], canvas[...])


def test_Layer_inicializando_com_parametros(canvas):
    layer = Layer(canvas, opacity=0.8, rotate=45, scale=0.5, blend_mode=BlendMode.MULTIPLY, name='Picture')
    assert layer.name == 'Picture'
    assert layer.opacity == 0.8
    assert layer.rotate == 45
    assert layer.scale == 0.5
    assert layer.blend_mode == BlendMode.MULTIPLY
    assert layer.region == Region.from_size(10, 10)
    assert np.array_equal(layer.image[...], canvas[...])


def test_Layer_rotate_mudanca_valor(canvas):
    layer = Layer(canvas)
    layer.rotate = 45
    assert layer.rotate == 45

    layer.rotate -= 20
    assert layer.rotate == 25


def test_Layer_scale_mudanca_valor(canvas):
    layer = Layer(canvas)
    layer.scale = 0.5
    assert layer.scale == 0.5

    layer.scale += 0.3
    assert layer.scale == 0.8


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
