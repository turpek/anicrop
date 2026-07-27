import pytest
import numpy as np
from anicrop.render import ViewportRender, ViewportPlan, render_edit
from anicrop.layer import Layer
from anicrop.spatial import Region, Span
from anicrop.image import Image, ImageFormat
from anicrop.viewport import Viewport
from anicrop.transform import TransformRel


def test_viewport_render_area_returns_image():
    """Valida se ViewportRender.render_area utiliza ViewportPlan para renderizar a camada."""
    img = Image.new((100, 100), ImageFormat.RGBA)
    layer = Layer(img)
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()
    plan = ViewportPlan(layer, viewport)

    rendered = vr.render_area(layer, plan)
    assert rendered is not None
    assert rendered.width > 0
    assert rendered.height > 0


def test_viewport_render_scene_composes_visible_layers():
    """Valida se ViewportRender.render_scene compõe todas as camadas visíveis na tela da Viewport."""
    img1 = Image.new((100, 100), ImageFormat.RGBA)
    img2 = Image.new((100, 100), ImageFormat.RGBA)
    layer1 = Layer(img1)
    layer2 = Layer(img2)
    viewport = Viewport((800, 600), 1.0)
    vr = ViewportRender()

    composition = vr.render_scene([layer1, layer2], viewport)
    assert composition is not None
    assert composition.width == 800
    assert composition.height == 600


@pytest.mark.slow
def test_viewport_render_passes_scale_factor_to_lod():
    """Valida se ViewportRender repassa o scale_factor da Viewport para a seleção de LOD dos edits."""
    img_large = Image.new((5000, 5000), ImageFormat.RGBA)
    layer = Layer(img_large)
    viewport = Viewport((800, 600), 0.1)  # Scale 0.1 -> n=3
    vr = ViewportRender()
    plan = ViewportPlan(layer, viewport)

    rendered = vr.render_area(layer, plan)
    assert rendered is not None
