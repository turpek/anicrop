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
    canvas = Canvas(100, 100)

    frame = CanvasFrame(layer, canvas)

    assert frame.bounds == Region(Span(50, 200), Span(50, 200))
    assert frame.dst_region == Region(Span(50, 50), Span(50, 50))
    assert frame.src_region == Region(Span(0, 50), Span(0, 50))


def test_canvas_frame_full_render_no_clipping():
    layer = make_layer(w=200, h=200, x=150, y=250)

    canvas = Canvas(200, 200)
    canvas._region += (150, 250)
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

    canvas = Canvas(20, 10)
    canvas._region += (190, 250)
    frame = CanvasFrame(layer, canvas, local=True)

    assert frame.bounds == Region(Span(0, 100), Span(0, 100))
    assert frame.dst_region == Region(Span(90, 10), Span(40, 20))
    assert frame.src_region == Region(Span(90, 10), Span(40, 20))


def test_canvas_frame_with_explicit_view_region_global():
    """Valida se view_region restringe o dst_region no espaço global."""
    layer = make_layer(w=200, h=200, x=0, y=0)
    canvas = Canvas(300, 300)
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

    canvas = Canvas(500, 500)
    view_region = Region(Span(190, 20), Span(250, 10))

    frame = CanvasFrame(layer, canvas, view_region=view_region, local=True)

    assert frame.bounds == Region(Span(0, 100), Span(0, 100))
    assert frame.dst_region == Region(Span(90, 10), Span(40, 20))
    assert frame.src_region == Region(Span(90, 10), Span(40, 20))


def test_canvas_frame_culling_layer_outside_canvas():
    """Valida culling quando a camada está totalmente fora do Canvas."""
    layer = make_layer(w=100, h=100, x=500, y=500)
    canvas = Canvas(200, 200)

    frame = CanvasFrame(layer, canvas)

    assert frame.dst_region is None
    assert frame.src_region is None


def test_canvas_frame_culling_view_region_outside_layer():
    """Valida culling quando view_region não intersecta a camada."""
    layer = make_layer(w=100, h=100, x=0, y=0)
    canvas = Canvas(500, 500)
    view_region = Region(Span(300, 50), Span(300, 50))

    frame = CanvasFrame(layer, canvas, view_region=view_region)

    assert frame.dst_region is None
    assert frame.src_region is None


def test_canvas_frame_culling_view_region_outside_canvas():
    """Valida culling quando view_region está totalmente fora do Canvas."""
    layer = make_layer(w=100, h=100, x=0, y=0)
    canvas = Canvas(200, 200)
    view_region = Region(Span(400, 50), Span(400, 50))

    frame = CanvasFrame(layer, canvas, view_region=view_region)

    assert frame.dst_region is None
    assert frame.src_region is None


def test_canvas_frame_global_matrix_and_bounds():
    """Valida se a matriz do frame projeta coordenadas locais para os pontos analíticos reais no Canvas."""
    layer = make_layer(w=100, h=100, x=10, y=20)
    layer.transform.rotate(90)
    canvas = Canvas(300, 400)

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
    canvas = Canvas(300, 400)

    frame = CanvasFrame(layer, canvas, local=True)

    assert frame.bounds == Region(Span(0, 100), Span(0, 100))
    np.testing.assert_allclose(frame.matrix, np.identity(3, dtype=np.float32))


@pytest.mark.skip(reason="surface_size será tratado na Tarefa 17")
def test_canvas_frame_surface_size():
    """Valida se surface_size reflete canvas.size."""
    layer = make_layer(w=100, h=100, x=10, y=20)
    canvas = Canvas(300, 400)

    frame = CanvasFrame(layer, canvas)
    assert frame.surface_size == (300, 400)
