import numpy as np
import pytest

from anicrop.image import Image, ImageFormat
from anicrop.layer import EditLayer
from anicrop.spatial import Region, Span


def test_image_new_creates_zarr_when_exceeding_threshold():
    """Garante que Image.new crie um backend Zarr quando a dimensão exceder 4Kx4K."""
    # 5000x5000 = 25MP > 16.7MP (4096x4096)
    img = Image.new((5000, 5000), ImageFormat.RGBA)
    assert img.is_zarr is True
    assert img.width == 5000
    assert img.height == 5000


def test_image_new_creates_ndarray_when_below_threshold():
    """Garante que Image.new crie um backend ndarray (RAM) quando a dimensão for <= 4Kx4K."""
    img = Image.new((1000, 1000), ImageFormat.RGBA)
    assert img.is_zarr is False
    assert img.width == 1000
    assert img.height == 1000


@pytest.mark.slow
def test_image_resize_smart_factory():
    """Verifica se Image.resize usa a fábrica inteligente para decidir entre Zarr ou ndarray."""
    # Cria imagem grande em Zarr
    img_zarr = Image.new((6000, 6000), ImageFormat.RGBA)
    assert img_zarr.is_zarr is True

    # Resize para 5000x5000 -> Continua sendo Zarr por exceder 4Kx4K
    img_down_zarr = img_zarr.resize((5000, 5000))
    assert img_down_zarr.is_zarr is True
    assert img_down_zarr.width == 5000

    # Resize para 1000x1000 -> Passa a ser ndarray por ser <= 4Kx4K
    img_down_ram = img_zarr.resize((1000, 1000))
    assert img_down_ram.is_zarr is False
    assert img_down_ram.width == 1000


@pytest.mark.slow
def test_edit_layer_init_prebuilds_lod_cache_for_zarr_images():
    """Garante que o construtor do EditLayer pré-gera a pirâmide de LODs quando a imagem for Zarr."""
    img_zarr = Image.new((6000, 6000), ImageFormat.RGBA)
    region = Region(Span(0, 6000), Span(0, 6000))
    matrix = np.identity(3, dtype=np.float32)

    edit_layer = EditLayer(img_zarr, region, matrix)

    # O cache de LOD deve ter sido pré-construído no __init__
    assert len(edit_layer._lod_cache) > 0

    # Nível 1: 3000x3000 (Ainda > 4Kx4K? Não, 3000x3000=9MP <= 16MP -> ndarray)
    lod1, m_local1 = edit_layer.get_lod(0.5)
    assert lod1.width == 3000
