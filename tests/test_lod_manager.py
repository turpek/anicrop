import gc
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from anicrop.render import LODManager


@pytest.fixture
def lod_manager():
    return LODManager()


@pytest.fixture
def mock_edit_layer():
    layer = MagicMock()
    mock_image = MagicMock()
    # No anicrop, acessamos os dados via image[...]
    mock_image.__getitem__.return_value = np.zeros(
        (3000, 4000, 3), dtype=np.uint8)
    mock_image.size = (4000, 3000)
    layer.image = mock_image
    return layer


@pytest.fixture
def mock_viewport():
    viewport = MagicMock()
    viewport.size = (800, 600)
    viewport.scale_factor = 1.0
    return viewport

# --- Grupo 1: A Regra da Área ---


def test_get_source_fallback_small_layer(lod_manager, mock_viewport, mock_edit_layer):
    layer_size = (400, 300)
    mock_viewport.scale_factor = 0.1
    with patch("cv2.resize") as mock_resize:
        pixels, m_adjust = lod_manager.get_source(
            mock_viewport, mock_edit_layer, layer_size)
        # Verifica se retornou os dados obtidos via image[...]
        assert pixels is mock_edit_layer.image.__getitem__.return_value
        assert np.array_equal(m_adjust, np.eye(3))
        mock_resize.assert_not_called()


def test_get_source_fallback_equal_layer(lod_manager, mock_viewport, mock_edit_layer):
    layer_size = (800, 600)
    mock_viewport.scale_factor = 0.1
    pixels, m_adjust = lod_manager.get_source(
        mock_viewport, mock_edit_layer, layer_size)
    assert pixels is mock_edit_layer.image.__getitem__.return_value
    assert np.array_equal(m_adjust, np.eye(3))

# --- Grupo 2: Roteamento de Escala ---


@pytest.mark.parametrize("scale_val", [1.0, 0.51])
def test_get_source_fallback_high_zoom(lod_manager, mock_viewport, mock_edit_layer, scale_val):
    layer_size = (4000, 3000)
    mock_viewport.scale_factor = scale_val
    pixels, m_adjust = lod_manager.get_source(
        mock_viewport, mock_edit_layer, layer_size)
    assert pixels is mock_edit_layer.image.__getitem__.return_value
    assert np.array_equal(m_adjust, np.eye(3))


@pytest.mark.parametrize("scale_val", [0.5, 0.26])
def test_get_source_l1_cache_routing(lod_manager, mock_viewport, mock_edit_layer, scale_val):
    layer_size = (4000, 3000)
    mock_viewport.scale_factor = scale_val
    with patch("cv2.resize", return_value=np.zeros((600, 800, 3))) as mock_resize:
        pixels, m_adjust = lod_manager.get_source(
            mock_viewport, mock_edit_layer, layer_size)
        mock_resize.assert_called_once()
        assert m_adjust[0, 0] == 5.0


@pytest.mark.parametrize("scale_val", [0.25, 0.1])
def test_get_source_l2_cache_routing(lod_manager, mock_viewport, mock_edit_layer, scale_val):
    layer_size = (4000, 3000)
    mock_viewport.scale_factor = scale_val
    with patch("cv2.resize", return_value=np.zeros((300, 400, 3))) as mock_resize:
        pixels, m_adjust = lod_manager.get_source(
            mock_viewport, mock_edit_layer, layer_size)
        mock_resize.assert_called_once()
        assert m_adjust[0, 0] == 10.0

# --- Grupo 3: Comportamento do Cache ---


def test_get_source_cache_hit_reuse(lod_manager, mock_viewport, mock_edit_layer):
    layer_size = (4000, 3000)
    mock_viewport.scale_factor = 0.5
    with patch("cv2.resize", return_value=np.zeros((600, 800, 3))) as mock_resize:
        lod_manager.get_source(mock_viewport, mock_edit_layer, layer_size)
        lod_manager.get_source(mock_viewport, mock_edit_layer, layer_size)
        assert mock_resize.call_count == 1

# --- Grupo 4: Memória ---


def test_lod_manager_memory_leak_weakref(lod_manager, mock_viewport):
    layer_size = (4000, 3000)
    mock_viewport.scale_factor = 0.5

    class MockLayer:
        def __init__(self):
            self.image = MagicMock()
            self.image.__getitem__.return_value = np.zeros(
                (3000, 4000, 3), dtype=np.uint8)
            self.image.size = (4000, 3000)

    local_layer = MockLayer()
    with patch("cv2.resize", return_value=np.zeros((600, 800, 3))):
        lod_manager.get_source(mock_viewport, local_layer, layer_size)
    assert len(lod_manager._l1_cache) == 1
    del local_layer
    gc.collect()
    assert len(lod_manager._l1_cache) == 0
