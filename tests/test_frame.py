import pytest
import numpy as np
from anicrop.canvas import Canvas

from anicrop.frame import CanvasFrame, ViewportFrame
from anicrop.layer import Layer
from anicrop.spatial import Region, Span
from anicrop.image import Image, ImageFormat
from anicrop.viewport import Viewport
from anicrop.transform import TransformRel


def make_layer(w=100, h=100, x=0, y=0, color=(255, 0, 0, 255)):
    img_data = np.zeros((h, w, 4), dtype=np.uint8)
    img_data[:] = color
    img = Image(img_data, ImageFormat.RGBA)
    layer = Layer(img)
    if x != 0 or y != 0:
        layer.transform.translate(x, y)
    return layer


def test_viewport_frame_full_overlap():
    viewport = Viewport((800, 600), 1.0)
    layer = make_layer(w=200, h=200, x=100, y=100)

    frame = ViewportFrame(layer, viewport)

    assert frame.bounds == Region(Span(400, 200), Span(300, 200))
    assert frame.src_region == Region(Span(0, 200), Span(0, 200))


def test_viewport_frame_partial_overlap():
    viewport = Viewport((800, 600), 1.0)
    layer = make_layer(w=200, h=200, x=-400, y=-300)

    frame = ViewportFrame(layer, viewport)

    assert frame.bounds == Region(Span(-100, 200), Span(-100, 200))
    assert frame.dst_region == Region(Span(0, 100), Span(0, 100))
    assert frame.src_region == Region(Span(100, 100), Span(100, 100))


def test_viewport_frame_local_state():
    """Valida se ViewportFrame com local=True projeta o Layer unrotated (Mexicano Deitado) na tela da Viewport com a Câmera."""
    img_mexican = Image(
        np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = Layer(img_mexican)
    layer.set_transform(TransformRel().rotate(-90))

    img_hat = Image(np.zeros((20, 20, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer.add_edit(img_hat, Region(Span(40, 20), Span(0, 20)))

    viewport = Viewport((800, 600), 1.0)
    frame = ViewportFrame(layer, viewport, local=True)

    assert frame.bounds == Region(Span(350, 100), Span(250, 100))
    assert frame.dst_region == Region(Span(350, 100), Span(250, 100))


def test_viewport_frame_with_explicit_view_region_global():
    """Valida se view_region restringe o dst_region na tela da Viewport no modo global."""
    viewport = Viewport((800, 600), 1.0)
    layer = make_layer(w=200, h=200, x=100, y=100)

    # bounds da camada na viewport: (400, 300, 200, 200)
    view_region = Region(Span(420, 50), Span(330, 40))
    frame = ViewportFrame(layer, viewport, view_region=view_region)

    assert frame.bounds == Region(Span(400, 200), Span(300, 200))
    assert frame.dst_region == Region(Span(420, 50), Span(330, 40))
    assert frame.src_region == Region(Span(20, 50), Span(30, 40))


def test_viewport_frame_with_explicit_view_region_local():
    """Valida se view_region restringe o dst_region na tela da Viewport no modo local."""
    img_mexican = Image(
        np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = Layer(img_mexican)
    layer.set_transform(TransformRel().rotate(-90).translate(150, 250))

    viewport = Viewport((800, 600), 1.0)
    # bounds no modo local (100x100 centralizado): (350, 250, 100, 100)
    view_region = Region(Span(370, 30), Span(260, 20))
    frame = ViewportFrame(layer, viewport, view_region=view_region, local=True)

    assert frame.bounds == Region(Span(350, 100), Span(250, 100))
    assert frame.dst_region == Region(Span(370, 30), Span(260, 20))
    assert frame.src_region == Region(Span(20, 30), Span(10, 20))


def test_viewport_frame_culling_layer_outside_viewport():
    """Valida culling quando a camada projetada está totalmente fora da Viewport."""
    viewport = Viewport((800, 600), 1.0)
    layer = make_layer(w=200, h=200, x=1000, y=1000)

    frame = ViewportFrame(layer, viewport)

    assert frame.dst_region is None
    assert frame.src_region is None


def test_viewport_frame_culling_view_region_outside_layer():
    """Valida culling quando view_region está na Viewport mas fora da camada."""
    viewport = Viewport((800, 600), 1.0)
    layer = make_layer(w=200, h=200, x=0, y=0)  # bounds na viewport: (300, 200, 200, 200)
    view_region = Region(Span(100, 50), Span(100, 50))

    frame = ViewportFrame(layer, viewport, view_region=view_region)

    assert frame.dst_region is None
    assert frame.src_region is None


def test_viewport_frame_culling_view_region_outside_viewport():
    """Valida culling quando view_region está totalmente fora da Viewport."""
    viewport = Viewport((800, 600), 1.0)
    layer = make_layer(w=200, h=200, x=0, y=0)
    view_region = Region(Span(1000, 50), Span(1000, 50))

    frame = ViewportFrame(layer, viewport, view_region=view_region)

    assert frame.dst_region is None
    assert frame.src_region is None


def test_viewport_frame_global_matrix_and_bounds():
    """Valida projeção analítica de pontos de controle pela câmera da Viewport no modo global."""
    layer = make_layer(w=100, h=100, x=10, y=20)
    layer.transform.rotate(90)
    viewport = Viewport((800, 600), 1.0)

    frame = ViewportFrame(layer, viewport, local=False)

    # 1. Bounds na tela da Viewport: (350+10, 250+20, 100, 100) = (360, 270, 100, 100)
    assert frame.bounds == Region(Span(360, 100), Span(270, 100))

    # 2. Ponto de Controle: Centro da camada (50, 50) vai para (410, 320)
    centro_projetado = frame.matrix @ np.array([50.0, 50.0, 1.0])
    np.testing.assert_allclose(centro_projetado[:2], [410.0, 320.0], atol=1e-5)

    # 3. Ponto de Controle: Top-Left (0, 0) vai para (460, 270)
    top_left_projetado = frame.matrix @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(top_left_projetado[:2], [460.0, 270.0], atol=1e-5)


def test_viewport_frame_local_matrix_and_bounds():
    """Valida se no modo local a Viewport centraliza a camada no estado unrotated."""
    layer = make_layer(w=100, h=100, x=10, y=20)
    layer.transform.rotate(90)
    viewport = Viewport((800, 600), 1.0)

    frame = ViewportFrame(layer, viewport, local=True)

    assert frame.bounds == Region(Span(350, 100), Span(250, 100))
    centro_projetado = frame.matrix @ np.array([50.0, 50.0, 1.0])
    np.testing.assert_allclose(centro_projetado[:2], [400.0, 300.0], atol=1e-5)


def test_canvas_frame_space_partial_overlap():
    layer = make_layer(w=200, h=200, x=50, y=50)
    # view_region = Region(Span(0, 100), Span(0, 100))
    canvas = Canvas.from_size(100, 100)

    frame = CanvasFrame(layer, canvas)

    assert frame.bounds == Region(Span(50, 200), Span(50, 200))
    assert frame.dst_region == Region(Span(50, 50), Span(50, 50))
    assert frame.src_region == Region(Span(0, 50), Span(0, 50))


def test_canvas_frame_full_render_no_clipping():
    layer = make_layer(w=200, h=200, x=150, y=250)

    canvas = Canvas.from_rect(150, 250, 200, 200)
    frame = CanvasFrame(layer, canvas)

    assert frame.bounds == Region(Span(150, 200), Span(250, 200))
    assert frame.dst_region == frame.bounds
    assert frame.src_region == Region(Span(0, 200), Span(0, 200))


def test_canvas_frame_local_state():
    """Valida se CanvasFrame com local=True calcula os bounds no espaço local (0,0,W,H) e converte a view_region global via matriz inversa."""
    img_mexican = Image(
        np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = Layer(img_mexican)
    layer.set_transform(TransformRel().rotate(-90).translate(150, 250))

    canvas = Canvas.from_rect(190, 250, 20, 10)
    frame = CanvasFrame(layer, canvas, local=True)

    assert frame.bounds == Region(Span(0, 100), Span(0, 100))
    assert frame.dst_region == Region(Span(90, 10), Span(40, 20))
    assert frame.src_region == Region(Span(90, 10), Span(40, 20))


def test_canvas_frame_with_explicit_view_region_global():
    """Valida se view_region restringe o dst_region no espaço global."""
    layer = make_layer(w=200, h=200, x=0, y=0)
    canvas = Canvas.from_size(300, 300)
    view_region = Region(Span(20, 50), Span(30, 40))

    frame = CanvasFrame(layer, canvas, view_region=view_region)

    assert frame.bounds == Region(Span(0, 200), Span(0, 200))
    assert frame.dst_region == Region(Span(20, 50), Span(30, 40))
    assert frame.src_region == Region(Span(20, 50), Span(30, 40))


def test_canvas_frame_with_explicit_view_region_local():
    """Valida se view_region global explícita é mapeada via matriz inversa no modo local."""
    img_mexican = Image(
        np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = Layer(img_mexican)
    layer.set_transform(TransformRel().rotate(-90).translate(150, 250))

    canvas = Canvas.from_size(500, 500)
    view_region = Region(Span(190, 20), Span(250, 10))

    frame = CanvasFrame(layer, canvas, view_region=view_region, local=True)

    assert frame.bounds == Region(Span(0, 100), Span(0, 100))
    assert frame.dst_region == Region(Span(90, 10), Span(40, 20))
    assert frame.src_region == Region(Span(90, 10), Span(40, 20))


def test_canvas_frame_culling_layer_outside_canvas():
    """Valida culling quando a camada está totalmente fora do Canvas."""
    layer = make_layer(w=100, h=100, x=500, y=500)
    canvas = Canvas.from_size(200, 200)

    frame = CanvasFrame(layer, canvas)

    assert frame.dst_region is None
    assert frame.src_region is None


def test_canvas_frame_culling_view_region_outside_layer():
    """Valida culling quando view_region não intersecta a camada."""
    layer = make_layer(w=100, h=100, x=0, y=0)
    canvas = Canvas.from_size(500, 500)
    view_region = Region(Span(300, 50), Span(300, 50))

    frame = CanvasFrame(layer, canvas, view_region=view_region)

    assert frame.dst_region is None
    assert frame.src_region is None


def test_canvas_frame_culling_view_region_outside_canvas():
    """Valida culling quando view_region está totalmente fora do Canvas."""
    layer = make_layer(w=100, h=100, x=0, y=0)
    canvas = Canvas.from_size(200, 200)
    view_region = Region(Span(400, 50), Span(400, 50))

    frame = CanvasFrame(layer, canvas, view_region=view_region)

    assert frame.dst_region is None
    assert frame.src_region is None


def test_canvas_frame_global_matrix_and_bounds():
    """Valida se a matriz do frame projeta coordenadas locais para os pontos analíticos reais no Canvas."""
    layer = make_layer(w=100, h=100, x=10, y=20)
    layer.transform.rotate(90)
    canvas = Canvas.from_size(300, 400)

    frame = CanvasFrame(layer, canvas, local=False)

    # 1. Bounds no Canvas
    assert frame.bounds == Region(Span(10, 100), Span(20, 100))

    # 2. Ponto de Controle: Centro da camada (50, 50) vai para o centro global (60, 70)
    centro_projetado = frame.matrix @ np.array([50.0, 50.0, 1.0])
    np.testing.assert_allclose(centro_projetado[:2], [60.0, 70.0], atol=1e-5)

    # 3. Ponto de Controle: Top-Left (0, 0) vai para (110, 20)
    top_left_projetado = frame.matrix @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(top_left_projetado[:2], [110.0, 20.0], atol=1e-5)


def test_canvas_frame_local_matrix_and_bounds():
    """Valida se no modo local a matriz é a identidade e os bounds são no espaço local."""
    layer = make_layer(w=100, h=100, x=10, y=20)
    layer.transform.rotate(90)
    canvas = Canvas.from_size(300, 400)

    frame = CanvasFrame(layer, canvas, local=True)

    assert frame.bounds == Region(Span(0, 100), Span(0, 100))
    np.testing.assert_allclose(frame.matrix, np.identity(3, dtype=np.float32))


def test_canvas_frame_surface_size():
    """Valida se surface_size reflete canvas.size."""
    layer = make_layer(w=100, h=100, x=10, y=20)
    canvas = Canvas.from_size(300, 400)

    frame = CanvasFrame(layer, canvas)
    assert frame.surface_size == (300, 400)


@pytest.mark.parametrize(
    "layer_rect, canvas_size, view_rect, expect_dst_rect, expect_src_rect",
    [
        pytest.param(
            (50, 50, 100, 100),
            (300, 300),
            (0, 0, 200, 200),
            (50, 50, 100, 100),
            (0, 0, 100, 100),
            id="view_region_larger_than_layer",
        ),
        pytest.param(
            (-50, -50, 200, 200),
            (300, 300),
            (50, 50, 200, 200),
            (50, 50, 100, 100),
            (100, 100, 100, 100),
            id="triple_partial_overlap",
        ),
        pytest.param(
            (0, 0, 100, 100),
            (200, 200),
            (60, 70, 100, 100),
            (60, 70, 40, 30),
            (60, 70, 40, 30),
            id="corner_clipping",
        ),
    ],
)
def test_canvas_frame_view_region_edge_cases(
    layer_rect,
    canvas_size,
    view_rect,
    expect_dst_rect,
    expect_src_rect,
):
    """Testa restrições e recortes de dst_region e src_region em CanvasFrame com view_region explícita."""
    x, y, w, h = layer_rect
    layer = make_layer(w=w, h=h, x=x, y=y)
    canvas = Canvas.from_size(*canvas_size)
    view_region = Region.from_rect(*view_rect)

    frame = CanvasFrame(layer, canvas, view_region=view_region)

    assert frame.dst_region == Region.from_rect(*expect_dst_rect)
    assert frame.src_region == Region.from_rect(*expect_src_rect)


@pytest.mark.parametrize(
    "layer_rect, view_rect, expect_dst_rect, expect_src_rect",
    [
        pytest.param(
            (0, 0, 100, 100),
            (200, 100, 400, 400),
            (350, 250, 100, 100),
            (0, 0, 100, 100),
            id="view_region_larger_than_projected_layer",
        ),
        pytest.param(
            (0, 0, 100, 100),
            (380, 270, 50, 50),
            (380, 270, 50, 50),
            (30, 20, 50, 50),
            id="partial_screen_crop",
        ),
        pytest.param(
            (-400, -300, 200, 200),
            (20, 20, 60, 60),
            (20, 20, 60, 60),
            (120, 120, 60, 60),
            id="triple_partial_screen_overlap",
        ),
    ],
)
def test_viewport_frame_view_region_edge_cases(
    layer_rect,
    view_rect,
    expect_dst_rect,
    expect_src_rect,
):
    """Testa projeção e recortes de dst_region e src_region na ViewportFrame com view_region explícita."""
    x, y, w, h = layer_rect
    viewport = Viewport((800, 600), 1.0)
    layer = make_layer(w=w, h=h, x=x, y=y)
    view_region = Region.from_rect(*view_rect)

    frame = ViewportFrame(layer, viewport, view_region=view_region)

    assert frame.dst_region == Region.from_rect(*expect_dst_rect)
    assert frame.src_region == Region.from_rect(*expect_src_rect)


@pytest.mark.parametrize(
    "layer_rect, canvas_size, expected_dst",
    [
        pytest.param((1000, 1000, 100, 100), (500, 500), None, id="culling_total_100_porcento_fora"),
        pytest.param((-40, -30, 100, 100), (500, 500), Region.from_rect(0, 0, 60, 70), id="recorte_topo_esquerdo_coords_negativas"),
        pytest.param((450, 470, 100, 100), (500, 500), Region.from_rect(450, 470, 50, 30), id="recorte_base_direita_limite_canvas"),
    ],
)
def test_canvas_frame_recorte_e_culling_nas_bordas(layer_rect, canvas_size, expected_dst):
    """Valida o cálculo exato de dst_region no CanvasFrame para culling total e recortes parciais de borda."""
    canvas = Canvas.from_size(*canvas_size)
    layer = make_layer(w=layer_rect[2], h=layer_rect[3], x=layer_rect[0], y=layer_rect[1])

    frame = CanvasFrame(layer, canvas)

    assert frame.dst_region == expected_dst


def test_frame_targ_region_com_view_region_calcula_offset_relativo():
    """Valida se targ_region desconta o view_region quando uma janela de visualização está ativa."""
    canvas = Canvas.from_size(1000, 1000)
    layer = make_layer(w=200, h=200, x=300, y=400)
    view_region = Region.from_rect(200, 200, 500, 500)

    frame = CanvasFrame(layer, canvas, view_region=view_region)

    assert frame.dst_region == Region.from_rect(300, 400, 200, 200)
    assert frame.targ_region == Region.from_rect(100, 200, 200, 200)


def test_frame_targ_region_sem_view_region_retorna_dst_region_integral():
    """Valida se targ_region retorna exatamente dst_region quando view_region é None."""
    canvas = Canvas.from_size(1000, 1000)
    layer = make_layer(w=200, h=200, x=300, y=400)

    frame = CanvasFrame(layer, canvas, view_region=None)

    assert frame.dst_region == Region.from_rect(300, 400, 200, 200)
    assert frame.targ_region == Region.from_rect(300, 400, 200, 200)


def test_calculate_mask_rect_combines_matrices():
    """Valida se calculate_mask_rect combina corretamente a matriz da máscara com a matriz espacial."""
    from anicrop.frame import calculate_mask_rect
    from anicrop.mask import Mask
    from anicrop.transform import mat_translation

    mask_img = Image(np.zeros((50, 60, 1), dtype=np.uint8), ImageFormat.GRAY)
    mask = Mask(mask_img, Region.from_size(50, 60), np.identity(3, dtype=np.float32))

    global_mat = mat_translation(100, 200)
    region = calculate_mask_rect(mask, global_mat)

    assert region == Region.from_rect(100, 200, 50, 60)


def test_canvas_frame_effective_view_with_mask():
    """Valida se CanvasFrame utiliza a região da máscara como effective_view quando view_region é None."""
    canvas = Canvas.from_size(1000, 1000)
    layer = make_layer(w=200, h=200, x=0, y=0)

    mask_img = Image(np.zeros((50, 50, 1), dtype=np.uint8), ImageFormat.GRAY)
    layer.set_mask(mask_img, Region.from_rect(20, 30, 50, 50))
    layer.transform.translate(100, 100)

    frame = CanvasFrame(layer, canvas)

    assert frame.dst_region == Region.from_rect(120, 130, 50, 50)


def test_canvas_frame_view_region_has_priority_over_mask():
    """Valida se view_region explícita possui prioridade sobre a máscara no CanvasFrame."""
    canvas = Canvas.from_size(1000, 1000)
    layer = make_layer(w=200, h=200, x=0, y=0)

    mask_img = Image(np.zeros((50, 50, 1), dtype=np.uint8), ImageFormat.GRAY)
    layer.set_mask(mask_img, Region.from_rect(20, 30, 50, 50))
    layer.transform.translate(100, 100)

    view_region = Region.from_rect(100, 100, 150, 150)
    frame = CanvasFrame(layer, canvas, view_region=view_region)

    assert frame.dst_region == Region.from_rect(100, 100, 150, 150)


def test_canvas_frame_expand_bounds_with_effects_padding():
    """Valida se CanvasFrame expande os bounds geométricos de acordo com o padding dos efeitos."""
    class DummyPaddingEffect:
        def __init__(self, visible: bool = True):
            self.visible = visible
            self.matrix = np.identity(3, dtype=np.float32)

        def prepare(self, frame):
            pass

        def get_padding(self):
            return (10, 15, 20, 25)

        def apply(self, image, matrix=None):
            return image

        def merge(self, other, matrix):
            return None

    canvas = Canvas.from_size(1000, 1000)
    layer = make_layer(w=100, h=100, x=200, y=200)
    layer.effects.append(DummyPaddingEffect())

    frame = CanvasFrame(layer, canvas)

    assert frame.bounds == Region.from_rect(175, 190, 140, 130)
    assert frame.dst_region == Region.from_rect(175, 190, 140, 130)
