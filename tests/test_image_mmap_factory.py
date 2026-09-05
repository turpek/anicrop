from pathlib import Path

import numpy as np
import pytest

from anicrop.buffer import ArrayBuffer, MMapBuffer
from anicrop.image import (
    Image,
    ImageFormat,
    get_memory_threshold,
    set_memory_threshold,
)
from anicrop.layer import EditLayer
from anicrop.spatial import Region, Span


def test_image_new_creates_mmap_when_exceeding_threshold():
    """Garante que Image.new crie um MMapBuffer quando a dimensão exceder o threshold."""
    img = Image.new((9000, 9000), ImageFormat.RGBA)
    assert isinstance(img._data, MMapBuffer)
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
    """Verifica se Image.resize usa a fábrica inteligente para decidir entre MMap ou ndarray."""
    img_mmap = Image.new((9000, 9000), ImageFormat.RGBA)
    assert isinstance(img_mmap._data, MMapBuffer)

    img_down_mmap = img_mmap.resize((8500, 8500))
    assert isinstance(img_down_mmap._data, MMapBuffer)
    assert img_down_mmap.width == 8500

    img_down_ram = img_mmap.resize((1000, 1000))
    assert isinstance(img_down_ram._data, ArrayBuffer)
    assert img_down_ram.width == 1000


def test_image_init_ndarray_exceeding_threshold_offloads_to_mmap():
    """Garante que instanciar Image com ndarray acima do threshold migre para MMapBuffer."""
    original = get_memory_threshold()
    try:
        set_memory_threshold(500 * 500)
        arr = np.zeros((600, 600, 3), dtype=np.uint8)
        img = Image(arr, ImageFormat.RGB)
        assert isinstance(img._data, MMapBuffer)
        assert img.width == 600
        assert img.height == 600
    finally:
        set_memory_threshold(original)


def test_image_init_with_memmap_array_uses_mmap_buffer(tmp_path: Path):
    """Garante que instanciar Image com np.memmap use MMapBuffer diretamente sem cópia."""
    raw_path = tmp_path / "test_mmap.raw"
    mm = np.memmap(str(raw_path), dtype=np.uint8, mode="w+", shape=(100, 100, 3))
    img = Image(mm, ImageFormat.RGB)
    assert isinstance(img._data, MMapBuffer)
    assert img.width == 100
    assert img.height == 100


@pytest.mark.slow
def test_edit_layer_lazy_lod_cache_for_large_images():
    """Garante que o EditLayer não gera LODs no __init__ e calcula sob demanda quando solicitado."""
    img_mmap = Image.new((9000, 9000), ImageFormat.RGBA)
    region = Region(Span(0, 9000), Span(0, 9000))
    matrix = np.identity(3, dtype=np.float32)

    edit_layer = EditLayer(img_mmap, region, matrix)
    assert len(edit_layer._lod_cache) == 0

    lod1, m_local1 = edit_layer.get_lod(0.5)
    assert len(edit_layer._lod_cache) == 1
    assert lod1.width == 4500


def test_set_memory_threshold_disables_mmap():
    """Valida que set_memory_threshold(None) desativa a criação de MMapBuffer em disco."""
    original = get_memory_threshold()
    try:
        set_memory_threshold(None)
        img = Image.new((10000, 10000), ImageFormat.RGBA)
        assert isinstance(img._data, ArrayBuffer)
    finally:
        set_memory_threshold(original)


def test_set_memory_threshold_custom_value():
    """Valida a definição de um threshold customizado via set_memory_threshold."""
    original = get_memory_threshold()
    try:
        set_memory_threshold(500 * 500)
        img = Image.new((600, 600), ImageFormat.RGBA)
        assert isinstance(img._data, MMapBuffer)
    finally:
        set_memory_threshold(original)
