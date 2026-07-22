import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from anicrop.image import Image, ImageFormat
from anicrop.spatial import Region, Span
from anicrop.layer import EditLayer


def make_edit_layer(width: int = 100, height: int = 100, is_zarr: bool = False) -> EditLayer:
    """Função utilitária para criar EditLayers com suporte a Zarr ou ndarray."""
    if is_zarr:
        mock_zarr = MagicMock()
        mock_zarr.ndim = 3
        mock_zarr.shape = (height, width, 4)
        mock_zarr.dtype = np.uint8
        mock_zarr.__getitem__.return_value = np.zeros(
            (height, width, 4), dtype=np.uint8)
        img = Image(mock_zarr, ImageFormat.RGBA)
    else:
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
        (0.8, 0, 1.0),       # -log2(0.8) = 0.32 -> floor = 0
        (0.5, 1, 0.5),       # -log2(0.5) = 1.0  -> floor = 1
        (0.3, 1, 0.5),       # -log2(0.3) = 1.73 -> floor = 1
        (0.25, 2, 0.25),     # -log2(0.25) = 2.0 -> floor = 2
        (0.1, 3, 0.125),     # -log2(0.1) = 3.32 -> floor = 3
    ]
)
def test_edit_layer_lod_level_calculation(scale_factor, expected_level, expected_lod_factor):
    """Verifica se o cálculo do nível de LOD e o fator discreto 2^-n correspondem à especificação."""
    edit_layer = make_edit_layer(1000, 1000)
    lod_image, m_local = edit_layer.get_lod(scale_factor)

    expected_w = int(1000 * expected_lod_factor)
    expected_h = int(1000 * expected_lod_factor)
    assert lod_image.width == expected_w
    assert lod_image.height == expected_h
    assert m_local is not None


# --- 2. Geração de LOD sob demanda sem cache para Imagens Comuns (<= 4Kx4K) ---

def test_edit_layer_lod_on_demand_no_cache_for_normal_images():
    """Garante que para imagens comuns (<= 4Kx4K) o resize com INTER_AREA ocorra sob demanda e sem cache."""
    edit_layer = make_edit_layer(1000, 1000)  # Imagem comum (1MP <= 16MP)

    with patch("cv2.resize", wraps=cv2.resize) as mock_resize:
        # Primeira chamada
        lod_img1, m_local1 = edit_layer.get_lod(0.5)
        assert lod_img1.width == 500
        assert mock_resize.call_count == 1

        # Segunda chamada para a mesma escala em imagem comum -> DEVE chamar resize sob demanda novamente!
        lod_img2, m_local2 = edit_layer.get_lod(0.5)
        assert lod_img2.width == 500
        assert mock_resize.call_count == 2

        # Verifica se usou a interpolação cv2.INTER_AREA
        _, kwargs = mock_resize.call_args_list[0]
        assert kwargs.get("interpolation") == cv2.INTER_AREA


# --- 3. Geração de Cache para Imagens Grandes (> 4Kx4K) ---

def test_edit_layer_lod_caching_for_large_images():
    """Verifica se imagens grandes (> 4Kx4K, ex: 5000x5000) alocam Zarr via Image.new e reutilizam o cache do LOD."""
    # Imagem grande 5000x5000 criada via Image.new (automaticamente vira Zarr)
    img_large = Image.new((5000, 5000), ImageFormat.RGBA)
    region = Region(Span(0, 5000), Span(0, 5000))
    matrix = np.identity(3, dtype=np.float32)
    edit_layer = EditLayer(img_large, region, matrix)

    assert edit_layer.image.is_zarr is True

    # O cache é pré-gerado e reutilizado
    lod_img1, m_local1 = edit_layer.get_lod(0.5)
    lod_img2, m_local2 = edit_layer.get_lod(0.5)
    assert lod_img1.width == 2500
    assert lod_img1 is lod_img2


# --- 4. Geração de Cache para Imagens Zarr ---

def test_edit_layer_lod_caching_for_zarr_images():
    """Verifica se imagens Zarr usam cache independente do tamanho."""
    edit_layer = make_edit_layer(1000, 1000, is_zarr=True)

    lod_img1, m_local1 = edit_layer.get_lod(0.5)
    lod_img2, m_local2 = edit_layer.get_lod(0.5)

    # Para Zarr, o cache é ativado no __init__ e reutilizado
    assert lod_img1 is lod_img2
    assert lod_img1.width == 500


# --- 5. Escala >= 1.0 Não Dispara Resize ---

@pytest.mark.parametrize("scale_factor", [1.0, 2.0, 5.0])
def test_edit_layer_lod_full_resolution_no_resize(scale_factor):
    """Garante que zoom normal ou zoom in (scale >= 1.0) retorna a imagem original sem resize."""
    edit_layer = make_edit_layer(1000, 1000)
    with patch("cv2.resize") as mock_resize:
        lod_image, m_local = edit_layer.get_lod(scale_factor)
        assert np.array_equal(m_local, edit_layer.local_matrix)
        assert lod_image is edit_layer.image
        mock_resize.assert_not_called()
