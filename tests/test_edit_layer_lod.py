from unittest.mock import patch

import cv2
import numpy as np
import pytest

from anicrop.image import Image, ImageFormat
from anicrop.layer import EditLayer
from anicrop.spatial import Region, Span


def make_edit_layer(width: int = 100, height: int = 100) -> EditLayer:
    """Função utilitária para criar EditLayers para testes de LOD."""
    data = np.zeros((height, width, 4), dtype=np.uint8)
    img = Image(data, ImageFormat.RGBA)
    region = Region(Span(0, width), Span(0, height))
    matrix = np.identity(3, dtype=np.float32)
    return EditLayer(img, region, matrix)


# --- 1. Seleção Correta do Nível de LOD (n = floor(-log2(f))) ---


@pytest.mark.parametrize(
    "scale_factor, expected_level, expected_lod_factor",
    [
        (1.5, 0, 1.0),
        (1.0, 0, 1.0),
        (0.8, 0, 1.0),  # -log2(0.8) = 0.32 -> floor = 0
        (0.5, 1, 0.5),  # -log2(0.5) = 1.0  -> floor = 1
        (0.3, 1, 0.5),  # -log2(0.3) = 1.73 -> floor = 1
        (0.25, 2, 0.25),  # -log2(0.25) = 2.0 -> floor = 2
        (0.1, 3, 0.125),  # -log2(0.1) = 3.32 -> floor = 3
    ],
)
def test_edit_layer_lod_level_calculation(
    scale_factor, expected_level, expected_lod_factor
):
    """Verifica se o cálculo do nível de LOD e o fator discreto 2^-n correspondem à especificação."""
    edit_layer = make_edit_layer(1000, 1000)
    lod_image, m_local = edit_layer.get_lod(scale_factor)

    expected_w = int(1000 * expected_lod_factor)
    expected_h = int(1000 * expected_lod_factor)
    assert lod_image.width == expected_w
    assert lod_image.height == expected_h
    assert m_local is not None


# --- 2. Geração de LOD sob demanda com Cache Reutilizável ---


def test_edit_layer_lod_lazy_caching():
    """Garante que o cálculo de LOD ocorre sob demanda e é reutilizado do cache em chamadas subsequentes."""
    edit_layer = make_edit_layer(1000, 1000)

    with patch("cv2.resize", wraps=cv2.resize) as mock_resize:
        # Primeira chamada: calcula sob demanda
        lod_img1, m_local1 = edit_layer.get_lod(0.5)
        assert lod_img1.width == 500
        assert mock_resize.call_count == 1

        # Segunda chamada: deve vir do cache de LOD sem chamar resize novamente
        lod_img2, m_local2 = edit_layer.get_lod(0.5)
        assert lod_img2.width == 500
        assert lod_img1 is lod_img2
        assert mock_resize.call_count == 1

        # Verifica se usou a interpolação cv2.INTER_AREA
        _, kwargs = mock_resize.call_args_list[0]
        assert kwargs.get("interpolation") == cv2.INTER_AREA


# --- 3. Escala >= 1.0 Não Dispara Resize ---


@pytest.mark.parametrize("scale_factor", [1.0, 2.0, 5.0])
def test_edit_layer_lod_full_resolution_no_resize(scale_factor):
    """Garante que zoom normal ou zoom in (scale >= 1.0) retorna a imagem original sem resize."""
    edit_layer = make_edit_layer(1000, 1000)
    with patch("cv2.resize") as mock_resize:
        lod_image, m_local = edit_layer.get_lod(scale_factor)
        assert np.array_equal(m_local, edit_layer.local_matrix)
        assert lod_image is edit_layer.image
        mock_resize.assert_not_called()
