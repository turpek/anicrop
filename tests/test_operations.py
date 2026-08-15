from anicrop.blend import BlendMode
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.operations import merge_down
from anicrop.spatial import Region
import numpy as np
import pytest

W = H = 10  # Tamanhos padrão do canvas


def make_region(w=3, h=3, offset=0):
    return Region.from_size(w, h) + offset


def make_canvas(w=W, h=H, channel=4, color=None):
    channels = {
        1: ImageFormat.GRAY, 2: ImageFormat.GRAY_ALPHA, 3: ImageFormat.RGB,
        4: ImageFormat.RGBA, -4: ImageFormat.CMYK, 5: ImageFormat.CMYK_ALPHA
    }
    if color:
        return Image.new((h, w), channels.get(channel), color=color)
    return Image.new((h, w), channels.get(channel))


@pytest.fixture
def canvas():
    return make_canvas()


@pytest.fixture
def identity_matrix():
    return np.eye(3, dtype=np.float32)


def make_layer(w=10, h=10, color=(0, 0, 0), top_left=(0, 0)):
    layer = Layer(make_canvas(w, h, color=color))
    layer.region += top_left
    return layer


@pytest.fixture
def template_layer():
    return make_layer


@pytest.mark.parametrize(
    'size, top_left, tl_up, tl_down, top_left_up, top_left_down',
    [
        ((10, 10), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)),
        ((15, 16), (-5, -6), (-5, 0), (0, -6), (0, 6), (5, 0)),
        ((20, 10), (-5, 0), (-5, 0), (5, 0), (0, 0), (10, 0)),
    ],
    ids=['overlapping_completely', 'overlapping_in_parts', 'without_overlapping']
)
def test_merge_down_em_varias_posicoes(
    mocker,
    template_layer,
    size,
    top_left,
    tl_up,
    tl_down,
    top_left_up,
    top_left_down,
):
    layer_up = template_layer(top_left=tl_up, color=(0, 0, 0, 0))
    layer_down = template_layer(top_left=tl_down, color=(0, 0, 0, 0))
    layer_down.blend_mode = BlendMode.MULTIPLY

    mock_image_down = mocker.MagicMock(spec=Image)
    mock_image_up = mocker.MagicMock(spec=Image)

    mock_blend_dict = mocker.patch('anicrop.operations.BLEND_MODE')
    # O mock_funcao_blend é a função falsa que o dicionário retorna quando chamamos .get()
    mock_blend = mock_blend_dict.get.return_value

    # Interceptamos o método render_layer() do CanvasRender.
    # O side_effect retorna o down na primeira vez que for chamado, e o up na segunda!
    mocker.patch(
        'anicrop.render.CanvasRender.render_layer',
        side_effect=[mock_image_down, mock_image_up]
    )

    # 3. SPY NA VIEW: Para pegarmos as Regions calculadas relativas ao canvas temporário
    spy_view = mocker.spy(Image, 'view')

    layer_mesclado = merge_down(layer_up, layer_down)

    assert mock_blend.call_count == 2

    # breakpoint()
    assert layer_mesclado.region.size == size
    assert layer_mesclado.region.top_left == top_left
    assert layer_mesclado.blend_mode == BlendMode.MULTIPLY

    args_view_down, _ = spy_view.call_args_list[0]
    region_down = args_view_down[1]

    args_view_up, _ = spy_view.call_args_list[1]
    region_up = args_view_up[1]

    assert region_down.top_left == top_left_down
    assert region_up.top_left == top_left_up


@pytest.mark.parametrize(
    'size, top_left, tl_up, tl_down, top_left_up, top_left_down',
    [
        # overlapping_completely: Ambas em (0,0)
        # Up (60°): tl=(-2,-2), size=14x14
        # Down (45°): tl=(-3,-3), size=16x16
        # Canvas: tl=(-3,-3), size=16x16
        ((16, 16), (-3, -3), (0, 0), (0, 0), (1, 1), (0, 0)),

        # overlapping_in_parts: Up em (-5, 0), Down em (0, -6)
        # Up (60°): tl=(-7,-2), size=14x14
        # Down (45°): tl=(-3,-9), size=16x16
        # Canvas: tl=(-7,-9), size=20x21
        ((20, 21), (-7, -9), (-5, 0), (0, -6), (0, 7), (4, 0)),

        # without_overlapping: Up em (-5, 0), Down em (5, 0)
        # Up (60°): tl=(-7,-2), size=14x14
        # Down (45°): tl=(2,-3), size=16x16
        # Canvas: tl=(-7,-3), size=25x16
        ((25, 16), (-7, -3), (-5, 0), (5, 0), (0, 1), (9, 0)),
    ],
    ids=['overlapping_completely', 'overlapping_in_parts', 'without_overlapping']
)
def test_merge_down_em_varias_posicoes_com_layer_up_60_graus_e_layer_down_45_graus(
    mocker,
    template_layer,
    size,
    top_left,
    tl_up,
    tl_down,
    top_left_up,
    top_left_down,
):
    layer_up = template_layer(top_left=tl_up, color=(0, 0, 0, 0))
    layer_up.transform.rotate(60)
    layer_down = template_layer(top_left=tl_down, color=(0, 0, 0, 0))
    layer_down.transform.rotate(45)
    layer_down.blend_mode = BlendMode.MULTIPLY

    mock_image_down = mocker.MagicMock(spec=Image)
    mock_image_up = mocker.MagicMock(spec=Image)

    mock_blend_dict = mocker.patch('anicrop.operations.BLEND_MODE')
    # O mock_funcao_blend é a função falsa que o dicionário retorna quando chamamos .get()
    mock_blend = mock_blend_dict.get.return_value

    # Interceptamos o método render_layer() do CanvasRender.
    # O side_effect retorna o down na primeira vez que for chamado, e o up na segunda!
    mocker.patch(
        'anicrop.render.CanvasRender.render_layer',
        side_effect=[mock_image_down, mock_image_up]
    )

    # 3. SPY NA VIEW: Para pegarmos as Regions calculadas relativas ao canvas temporário
    spy_view = mocker.spy(Image, 'view')

    # breakpoint()
    layer_mesclado = merge_down(layer_up, layer_down)

    assert mock_blend.call_count == 2

    assert layer_mesclado.region.size == size
    assert layer_mesclado.region.top_left == top_left
    assert layer_mesclado.blend_mode == BlendMode.MULTIPLY

    args_view_down, _ = spy_view.call_args_list[0]
    region_down = args_view_down[1]

    args_view_up, _ = spy_view.call_args_list[1]
    region_up = args_view_up[1]

    assert region_down.top_left == top_left_down
    assert region_up.top_left == top_left_up
