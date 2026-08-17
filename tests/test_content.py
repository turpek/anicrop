import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.content import Content
from anicrop.document import Document
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


def test_content_crop_via_document_facade():
    """Valida o uso do crop diretamente atraves da propriedade doc.content."""
    doc = Document("TestCrop", 100, 100, history=False)
    data = np.full((100, 100, 4), [0, 255, 0, 255], dtype=np.uint8)
    layer = doc.add(Layer(Image(data, ImageFormat.RGBA), name="verde"))

    success = doc.content.crop(layer, (30, 30, 40, 40))
    assert success is True

    result = doc.render()
    np.testing.assert_array_equal(result[50, 50], [0, 255, 0, 255])
    np.testing.assert_array_equal(result[10, 10], [0, 0, 0, 0])


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


def test_content_fit_preserves_aspect_ratio_and_centers():
    """Valida se Content.fit escala proporcionalmente mantendo proporcao e centralizando."""
    # Camada 200x100 (aspect ratio 2:1)
    data = np.full((100, 200, 4), 255, dtype=np.uint8)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    # Referencia 100x100 -> escala deve ser 0.5 (nova largura 100, nova altura 50)
    ref_region = Region.from_rect(0, 0, 100, 100)
    result = content.fit(layer, ref_region)

    assert result is True
    assert layer.global_region.size == (100, 50)
    # Centralizado verticalmente em y=25
    assert layer.global_region.y.start == 25
    assert layer.global_region.x.start == 0


def test_content_fit_with_canvas_reference():
    """Valida Content.fit usando uma instancia de Canvas como referencia."""
    canvas = Canvas.from_size(400, 400)
    data = np.full((200, 100, 4), 255, dtype=np.uint8)  # 100x200 (1:2)
    layer = Layer(Image(data, ImageFormat.RGBA))
    content = Content()

    result = content.fit(layer, canvas)
    assert result is True
    assert layer.global_region.size == (200, 400)
    assert layer.global_region.x.start == 100
    assert layer.global_region.y.start == 0


def test_content_resize_and_fit_via_document_facade():
    """Valida Content.resize e Content.fit atraves da fachada doc.content."""
    doc = Document("DocContentTest", 200, 200, history=False)
    data = np.full((50, 50, 4), 255, dtype=np.uint8)
    layer = doc.add(Layer(Image(data, ImageFormat.RGBA), name="L1"))

    doc.content.resize(layer, 100, 100)
    assert layer.global_region.size == (100, 100)

    doc.content.fit(layer, doc.canvas)
    assert layer.global_region.size == (200, 200)
