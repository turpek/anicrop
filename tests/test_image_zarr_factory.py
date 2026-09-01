import numpy as np
import pytest

from anicrop.buffer import ArrayBuffer, ZarrBuffer
from anicrop.image import Image, ImageFormat
from anicrop.layer import EditLayer
from anicrop.spatial import Region, Span


def test_image_new_creates_zarr_when_exceeding_threshold():
    """Garante que Image.new crie um backend Zarr quando a dimensão exceder o threshold (8Kx8K = 64MP)."""
    # 9000x9000 = 81MP > 64MP (8192x8192)
    img = Image.new((9000, 9000), ImageFormat.RGBA)
    assert isinstance(img._data, ZarrBuffer)
    assert img.width == 9000
    assert img.height == 9000


def test_image_new_creates_ndarray_when_below_threshold():
    """Garante que Image.new crie um backend ndarray (RAM) quando a dimensão for <= 8Kx8K."""
    img = Image.new((5000, 5000), ImageFormat.RGBA)
    assert isinstance(img._data, ArrayBuffer)
    assert img.width == 5000
    assert img.height == 5000


@pytest.mark.slow
def test_image_resize_smart_factory():
    """Verifica se Image.resize usa a fábrica inteligente para decidir entre Zarr ou ndarray."""
    # Cria imagem grande em Zarr (> 64MP)
    img_zarr = Image.new((9000, 9000), ImageFormat.RGBA)
    assert isinstance(img_zarr._data, ZarrBuffer)

    # Resize para 8500x8500 -> Continua sendo Zarr por exceder 64MP
    img_down_zarr = img_zarr.resize((8500, 8500))
    assert isinstance(img_down_zarr._data, ZarrBuffer)
    assert img_down_zarr.width == 8500

    # Resize para 1000x1000 -> Passa a ser ndarray por ser <= 64MP
    img_down_ram = img_zarr.resize((1000, 1000))
    assert isinstance(img_down_ram._data, ArrayBuffer)
    assert img_down_ram.width == 1000


@pytest.mark.slow
def test_edit_layer_lazy_lod_cache_for_large_images():
    """Garante que o EditLayer não gera LODs no __init__ e calcula sob demanda quando solicitado."""
    img_zarr = Image.new((9000, 9000), ImageFormat.RGBA)
    region = Region(Span(0, 9000), Span(0, 9000))
    matrix = np.identity(3, dtype=np.float32)

    edit_layer = EditLayer(img_zarr, region, matrix)

    # O cache de LOD inicia vazio (Lazy)
    assert len(edit_layer._lod_cache) == 0

    # Nível 1: 4500x4500 calculado sob demanda
    lod1, m_local1 = edit_layer.get_lod(0.5)
    assert len(edit_layer._lod_cache) == 1
    assert lod1.width == 4500


def test_set_memory_threshold_disables_zarr():
    """Valida que set_memory_threshold(None) desativa a criação de Zarr em disco."""
    from anicrop.image import get_memory_threshold, set_memory_threshold

    original = get_memory_threshold()
    try:
        set_memory_threshold(None)
        img = Image.new((10000, 10000), ImageFormat.RGBA)
        assert isinstance(img._data, ArrayBuffer)
    finally:
        set_memory_threshold(original)


def test_set_memory_threshold_custom_value():
    """Valida a definição de um threshold customizado via set_memory_threshold."""
    from anicrop.image import get_memory_threshold, set_memory_threshold

    original = get_memory_threshold()
    try:
        set_memory_threshold(500 * 500)
        img = Image.new((600, 600), ImageFormat.RGBA)
        assert isinstance(img._data, ZarrBuffer)
    finally:
        set_memory_threshold(original)
