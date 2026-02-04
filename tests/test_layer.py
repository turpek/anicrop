from anicrop.image import Image, ImageFormat
from anicrop.layer import EditLayer, BlendMode
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
