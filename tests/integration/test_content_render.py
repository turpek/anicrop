from __future__ import annotations
import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.content import Content
from anicrop.edit_layer import CropEditLayer
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.layout import LayerLayoutStrategy
from anicrop.render import CanvasRender


def test_content_crop_render_pipeline_rgba():
    """Valida a renderizacao completa de camada RGBA com corte de conteudo via CanvasRender."""
    canvas = Canvas.from_size(100, 100)
    data = np.zeros((100, 100, 4), dtype=np.uint8)
    data[:] = [255, 0, 0, 255]
    layer = Layer(Image(data, ImageFormat.RGBA))

    content = Content()
    content.crop(layer, (25, 25, 50, 50))

    renderer = CanvasRender()
    rendered = renderer.render_scene([layer], canvas)

    np.testing.assert_array_equal(rendered[50, 50], [255, 0, 0, 255])
    np.testing.assert_array_equal(rendered[10, 10], [0, 0, 0, 0])


def test_content_crop_render_pipeline_rgb():
    """Valida a renderizacao de camada RGB onde o corte enquadra o conteudo via CanvasRender."""
    canvas = Canvas.from_size(100, 100)
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    data[:] = [0, 0, 255]
    layer = Layer(Image(data, ImageFormat.RGB))

    content = Content()
    content.crop(layer, (25, 25, 50, 50))

    renderer = CanvasRender()
    rendered = renderer.render_scene([layer], canvas)

    np.testing.assert_array_equal(rendered[50, 50, :3], [0, 0, 255])
    np.testing.assert_array_equal(rendered[10, 10], [0, 0, 0, 0])


def test_crop_followed_by_rotation_clears_outer_pixels():
    """Valida se CropEditLayer apaga os pixels externos em cenarios com rotacao."""
    canvas = Canvas.from_size(200, 200)
    data = np.zeros((100, 100, 4), dtype=np.uint8)
    data[:] = [255, 0, 0, 255]
    layer = Layer(Image(data, ImageFormat.RGBA))

    content = Content()
    content.crop(layer, (30, 30, 40, 40))
    layer.transform.rotate(45)

    assert isinstance(layer._edits[-1], CropEditLayer)

    renderer = CanvasRender()
    rendered = renderer.render_scene([layer], canvas)

    center_x = (layer.global_region.x.start + layer.global_region.x.end) // 2
    center_y = (layer.global_region.y.start + layer.global_region.y.end) // 2
    np.testing.assert_array_equal(rendered[center_y, center_x], [255, 0, 0, 255])
    np.testing.assert_array_equal(rendered[150, 150], [0, 0, 0, 0])
    np.testing.assert_array_equal(rendered[5, 5], [0, 0, 0, 0])


def test_crop_visibility_toggle_restores_base_image():
    """Valida se desativar visible no CropEditLayer e expandir o layout revela a imagem base."""
    canvas = Canvas.from_size(100, 100)
    data = np.zeros((100, 100, 4), dtype=np.uint8)
    data[:] = [255, 0, 0, 255]
    layer = Layer(Image(data, ImageFormat.RGBA))

    content = Content()
    content.crop(layer, (30, 30, 40, 40))

    crop_edit = layer._edits[-1]
    crop_edit.visible = False
    LayerLayoutStrategy.fit(layer, layer.base.region)

    renderer = CanvasRender()
    rendered = renderer.render_scene([layer], canvas)

    np.testing.assert_array_equal(rendered[10, 10], [255, 0, 0, 255])
    np.testing.assert_array_equal(rendered[50, 50], [255, 0, 0, 255])
    np.testing.assert_array_equal(rendered[90, 90], [255, 0, 0, 255])


def test_crop_followed_by_fit_content_restores_base_image_automatically():
    """Valida se fit_content desativa CropEditLayer e restaura a moldura para a imagem base total."""
    canvas = Canvas.from_size(100, 100)
    data = np.zeros((100, 100, 4), dtype=np.uint8)
    data[:] = [255, 0, 0, 255]
    layer = Layer(Image(data, ImageFormat.RGBA))

    content = Content()
    content.crop(layer, (30, 30, 40, 40))
    assert layer.global_region.size == (40, 40)

    success = LayerLayoutStrategy.fit_content(layer)
    assert success is True
    assert layer._edits[-1].visible is False
    assert layer.global_region.size == (100, 100)

    renderer = CanvasRender()
    rendered = renderer.render_scene([layer], canvas)

    np.testing.assert_array_equal(rendered[10, 10], [255, 0, 0, 255])
    np.testing.assert_array_equal(rendered[50, 50], [255, 0, 0, 255])
    np.testing.assert_array_equal(rendered[90, 90], [255, 0, 0, 255])


def test_content_flip_x_render_pipeline():
    """Valida a renderização de camada espelhada horizontalmente via flip_x."""
    canvas = Canvas.from_size(100, 100)
    data = np.zeros((100, 100, 4), dtype=np.uint8)
    data[:, :50] = [255, 0, 0, 255]  # Esquerda: vermelho
    data[:, 50:] = [0, 0, 255, 255]  # Direita: azul
    layer = Layer(Image(data, ImageFormat.RGBA))

    content = Content()
    content.flip_x(layer)

    renderer = CanvasRender()
    rendered = renderer.render_scene([layer], canvas)

    # Apos flip_x, esquerda deve ser azul e direita vermelha
    np.testing.assert_array_equal(rendered[50, 25], [0, 0, 255, 255])
    np.testing.assert_array_equal(rendered[50, 75], [255, 0, 0, 255])


def test_content_flip_y_render_pipeline():
    """Valida a renderização de camada espelhada verticalmente via flip_y."""
    canvas = Canvas.from_size(100, 100)
    data = np.zeros((100, 100, 4), dtype=np.uint8)
    data[:50, :] = [0, 255, 0, 255]    # Topo: verde
    data[50:, :] = [255, 255, 0, 255]  # Base: amarelo
    layer = Layer(Image(data, ImageFormat.RGBA))

    content = Content()
    content.flip_y(layer)

    renderer = CanvasRender()
    rendered = renderer.render_scene([layer], canvas)

    # Apos flip_y, topo deve ser amarelo e base verde
    np.testing.assert_array_equal(rendered[25, 50], [255, 255, 0, 255])
    np.testing.assert_array_equal(rendered[75, 50], [0, 255, 0, 255])
