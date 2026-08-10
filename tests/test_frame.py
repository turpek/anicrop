import pytest
import numpy as np
from anicrop.frame import BaseFrame, CanvasFrame, ViewportFrame
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


def test_canvas_frame_space_partial_overlap():
    layer = make_layer(w=200, h=200, x=50, y=50)
    view_region = Region(Span(0, 100), Span(0, 100))

    frame = CanvasFrame(layer, view_region=view_region)

    assert frame.bounds == Region(Span(50, 200), Span(50, 200))
    assert frame.dst_region == Region(Span(50, 50), Span(50, 50))
    assert frame.src_region == Region(Span(0, 50), Span(0, 50))


def test_canvas_frame_full_render_no_clipping():
    layer = make_layer(w=200, h=200, x=150, y=250)

    frame = CanvasFrame(layer)

    assert frame.bounds == Region(Span(150, 200), Span(250, 200))
    assert frame.dst_region == frame.bounds
    assert frame.src_region == Region(Span(0, 200), Span(0, 200))


def test_canvas_frame_local_state():
    """Valida se CanvasFrame com local=True calcula os bounds no espaço local (0,0,W,H) e converte a view_region global via matriz inversa."""
    img_mexican = Image(
        np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA)
    layer = Layer(img_mexican)
    layer.set_transform(TransformRel().rotate(-90))

    global_view_region = Region(Span(40, 20), Span(0, 10))

    frame = CanvasFrame(layer, view_region=global_view_region, local=True)

    assert frame.bounds == Region(Span(0, 100), Span(0, 100))
    assert frame.dst_region == Region(Span(90, 10), Span(40, 20))
    assert frame.src_region == Region(Span(90, 10), Span(40, 20))
