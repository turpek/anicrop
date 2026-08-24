import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.container import GroupLayer
from anicrop.content import Content, FitContext
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.render import CanvasRender
from anicrop.spatial import Region


def test_content_crop_fits_layer_and_adds_clip_edit():
    """Valida se Content.crop ajusta a geometria e anexa um EditLayer com BlendMode.CLIP."""
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    crop_box = Region.from_rect(20, 20, 40, 40)
    result = content.crop(layer, crop_box)

    assert result is True
    assert len(layer._edits) == 2
    assert layer.global_region == Region.from_rect(20, 20, 40, 40)


def test_content_crop_same_region_returns_false():
    """Valida se Content.crop retorna False caso a regiao de corte seja identica a atual."""
    data = np.full((50, 50, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    result = content.crop(layer, (0, 0, 50, 50))
    assert result is False


def test_content_crop_with_canvas_reference():
    """Valida Content.crop passando uma instancia de Canvas como referencia."""
    canvas = Canvas.from_size(80, 80)
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))

    content = Content()
    result = content.crop(layer, canvas)

    assert result is True
    assert layer.global_region == Region.from_size(80, 80)


def test_content_crop_with_layer_reference():
    """Valida Content.crop passando outra camada como referencia."""
    ref_layer = Layer(Image(np.zeros((60, 60, 4), dtype=np.uint8), ImageFormat.RGBA))
    ref_layer.transform.translate(10, 10)

    target_layer = Layer(Image(np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA))
    content = Content()
    result = content.crop(target_layer, ref_layer)

    assert result is True
    assert target_layer.global_region == ref_layer.global_region


def test_content_crop_render_pipeline_rgba():
    """Valida a renderizacao completa de camada RGBA com corte de conteudo via CanvasRender."""
    canvas = Canvas.from_size(100, 100)
    # Camada vermelha 100x100
    data = np.zeros((100, 100, 4), dtype=np.uint8)
    data[:] = [255, 0, 0, 255]
    layer = Layer(Image(data, ImageFormat.RGBA))

    content = Content()
    content.crop(layer, (25, 25, 50, 50))

    renderer = CanvasRender()
    rendered = renderer.render_scene([layer], canvas)

    # Dentro da area de corte (50, 50): vermelho
    np.testing.assert_array_equal(rendered[50, 50], [255, 0, 0, 255])
    # Fora da area de corte (10, 10): transparente (fundo preto do Canvas)
    np.testing.assert_array_equal(rendered[10, 10], [0, 0, 0, 0])


def test_content_crop_render_pipeline_rgb():
    """Valida a renderizacao de camada RGB onde o corte enquadra o conteudo via CanvasRender."""
    canvas = Canvas.from_size(100, 100)
    # Camada azul sólida 100x100 em RGB
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    data[:] = [0, 0, 255]
    layer = Layer(Image(data, ImageFormat.RGB))

    content = Content()
    content.crop(layer, (25, 25, 50, 50))

    renderer = CanvasRender()
    rendered = renderer.render_scene([layer], canvas)

    # Dentro da area de corte (50, 50): azul solido
    np.testing.assert_array_equal(rendered[50, 50, :3], [0, 0, 255])
    # Fora da area de corte no Canvas (10, 10): fundo padrao do Canvas (transparente/preto)
    np.testing.assert_array_equal(rendered[10, 10], [0, 0, 0, 0])


def test_content_resize_scales_transform():
    """Valida se Content.resize altera as dimensoes globais via transform."""
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    result = content.resize(layer, 200, 150)
    assert result is True
    assert layer.global_region.size == (200, 150)


def test_content_resize_same_size_returns_false():
    """Valida se Content.resize retorna False caso o tamanho ja seja o desejado."""
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    result = content.resize(layer, 100, 100)
    assert result is False


def test_content_resize_invalid_size_raises_value_error():
    """Valida se Content.resize lanca ValueError para dimensoes nao positivas."""
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    with pytest.raises(ValueError, match="Dimensões inválidas para resize"):
        content.resize(layer, -50, 100)


def test_content_fit_scales_and_aligns_to_exact_reference():
    """Valida se Content.fit escala e posiciona o conteudo exatamente sobre a regiao de referencia."""
    data = np.full((100, 200, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    ref_region = Region.from_rect(20, 30, 300, 150)
    result = content.fit(layer, ref_region)

    assert result is True
    assert layer.global_region == Region.from_rect(20, 30, 300, 150)


def test_content_fit_same_region_returns_false():
    """Valida se Content.fit retorna False caso a camada ja coincida exatamente com a referencia."""
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    result = content.fit(layer, Region.from_size(100, 100))
    assert result is False


def test_content_fit_with_canvas_reference():
    """Valida Content.fit ajustando o conteudo exatamente a uma instancia de Canvas como referencia."""
    canvas = Canvas.from_rect(10, 20, 400, 300)
    data = np.full((200, 100, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    result = content.fit(layer, canvas)
    assert result is True
    assert layer.global_region == Region.from_rect(10, 20, 400, 300)


def test_content_fit_inside_translated_group():
    """Valida se Content.fit projeta a escala e translação corretamente para camada dentro de GroupLayer."""
    group = GroupLayer()
    group.transform.translate(50, 50)

    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    group.append(layer)

    content = Content()
    ref_region = Region.from_rect(100, 100, 200, 200)
    result = content.fit(layer, ref_region)

    assert result is True
    assert layer.global_region == Region.from_rect(100, 100, 200, 200)


def test_content_fit_helper_contain_dispatch():
    """Valida o uso de FitContext.fit_contain com dispatch automatico via @ovld no Content.fit."""
    data = np.full((100, 200, 4), 255, dtype=np.uint8)  # 200x100 (2:1)
    layer = Layer(Image(data, ImageFormat.RGBA))
    canvas = Canvas.from_size(100, 100)
    content = Content()

    # Contain 200x100 em 100x100 -> 100x50 centralizado em (0, 25)
    cf = FitContext(layer, canvas)
    result = content.fit(cf.fit_contain)

    assert result is True
    assert layer.global_region == Region.from_rect(0, 25, 100, 50)


def test_content_fit_helper_cover_dispatch():
    """Valida o uso de FitContext.fit_cover com dispatch automatico via @ovld no Content.fit."""
    data = np.full((100, 200, 4), 255, dtype=np.uint8)  # 200x100 (2:1)
    layer = Layer(Image(data, ImageFormat.RGBA))
    canvas = Canvas.from_size(100, 100)
    content = Content()

    # Cover 200x100 em 100x100 -> 200x100 centralizado em (-50, 0)
    cf = FitContext(layer, canvas)
    result = content.fit(cf.fit_cover)

    assert result is True
    assert layer.global_region == Region.from_rect(-50, 0, 200, 100)


def test_content_fit_helper_scale_width_and_height():
    """Valida o uso de FitContext.scale_width e scale_height com Content.fit."""
    data = np.full((100, 200, 4), 255, dtype=np.uint8)  # 200x100 (2:1)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    cf_w = FitContext(layer, (0, 0, 400, 100))
    content.fit(cf_w.scale_width)
    assert layer.global_region == Region.from_rect(0, 0, 400, 200)

    cf_h = FitContext(layer, (0, 0, 100, 100))
    content.fit(cf_h.scale_height)
    assert layer.global_region == Region.from_rect(0, 0, 200, 100)


def test_content_fit_helper_with_custom_factors():
    """Valida FitContext com fatores de alinhamento customizados passados no construtor."""
    data = np.full((100, 200, 4), 255, dtype=np.uint8)  # 200x100 (2:1)
    layer = Layer(Image(data, ImageFormat.RGBA))
    canvas = Canvas.from_size(100, 100)
    content = Content()

    # Alinhamento no canto superior esquerdo (0.0, 0.0)
    cf = FitContext(layer, canvas, x_factor=0.0, y_factor=0.0)
    content.fit(cf.fit_contain)
    assert layer.global_region == Region.from_rect(0, 0, 100, 50)


def test_content_flip_x_applies_scale_matrix():
    """Valida se flip_x aplica a escala de espelhamento horizontal preservando a regiao global."""
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    result = content.flip_x(layer)

    assert result is True
    assert layer.matrix[0, 0] == -1.0
    assert layer.global_region == Region.from_size(100, 100)


def test_content_flip_y_applies_scale_matrix():
    """Valida se flip_y aplica a escala de espelhamento vertical preservando a regiao global."""
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    result = content.flip_y(layer)

    assert result is True
    assert layer.matrix[1, 1] == -1.0
    assert layer.global_region == Region.from_size(100, 100)
