import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from anicrop.render import ViewportRender, RenderFlags
from anicrop.spatial import Region
from anicrop.image import Image
from anicrop.type import Id, Scale
from anicrop.transform import mat_translation, mat_pivot, mat_position
from anicrop.viewport import Viewport


@pytest.fixture
def viewport_render():
    return ViewportRender()


@pytest.fixture
def mock_layer():
    layer = MagicMock()
    # Usamos a classe Id real para garantir compatibilidade com WeakKeyDictionary
    layer._id = Id()
    layer.format = MagicMock()
    layer._resolve_render.return_value = RenderFlags.NONE
    # Usamos uma Região real para que os cálculos de top_left e size funcionem no numpy
    layer.region = Region.from_size(800, 600)
    return layer


@pytest.fixture
def real_viewport():
    # Retorna uma Viewport real com tamanho 800x600 e fit_scale 1.0
    return Viewport((800, 600), 1.0)

# --- Grupo de Testes: ViewportRender.render_area ---


def test_render_area_none_region(viewport_render, mock_layer, real_viewport):
    """Cenário 1: Se __render_region retornar None, render_area retorna None."""
    with patch.object(ViewportRender, "_final_region"), \
            patch.object(ViewportRender, "_ViewportRender__render_region", return_value=None):
        result = viewport_render.render_area(mock_layer, real_viewport)
        assert result is None


def test_render_area_dirty_pixels_calls_flatten(viewport_render, mock_layer, real_viewport):
    """Cenário 2: Se flags & RenderFlags.PIXELS for verdadeiro, chama flatten_edits."""
    mock_region = MagicMock(spec=Region)
    mock_region.size = (800, 600)
    mock_layer._resolve_render.return_value = RenderFlags.PIXELS

    with patch.object(ViewportRender, "_final_region", return_value=mock_region), \
            patch.object(ViewportRender, "_ViewportRender__render_region", return_value=mock_region), \
            patch.object(ViewportRender, "_ViewportRender__flatten_edits") as mock_flatten, \
            patch("anicrop.render.Image.new") as mock_image_new:

        viewport_render.render_area(mock_layer, real_viewport)

        mock_flatten.assert_called_once()
        mock_image_new.assert_called_once_with(
            mock_region.size, mock_layer.format)


def test_render_area_cache_miss_calls_flatten(viewport_render, mock_layer, real_viewport):
    """Cenário 3: Se não estiver no cache, chama flatten_edits mesmo sem flag PIXELS."""
    mock_region = MagicMock(spec=Region)
    mock_region.size = (800, 600)
    mock_layer._resolve_render.return_value = RenderFlags.NONE

    # Garante que o cache está vazio para este ID
    if mock_layer._id in viewport_render._cache:
        del viewport_render._cache[mock_layer._id]

    with patch.object(ViewportRender, "_final_region", return_value=mock_region), \
            patch.object(ViewportRender, "_ViewportRender__render_region", return_value=mock_region), \
            patch.object(ViewportRender, "_ViewportRender__flatten_edits") as mock_flatten, \
            patch("anicrop.render.Image.new") as mock_image_new:

        viewport_render.render_area(mock_layer, real_viewport)

        mock_flatten.assert_called_once()
        mock_image_new.assert_called_once_with(
            mock_region.size, mock_layer.format)


def test_render_area_cache_hit_calls_crop(viewport_render, mock_layer, real_viewport):
    """Cenário 4: Se houver cache válido, usa overlap_with e crop."""
    mock_region = MagicMock(spec=Region)
    mock_layer._resolve_render.return_value = RenderFlags.NONE

    # Injeta mock no cache manualmente
    mock_cached_image = MagicMock(spec=Image)
    viewport_render._cache[mock_layer._id] = mock_cached_image

    final_region_mock = MagicMock(spec=Region)
    mock_view_coord = MagicMock()
    final_region_mock.overlap_with.return_value = mock_view_coord

    with patch.object(ViewportRender, "_final_region", return_value=final_region_mock), \
            patch.object(ViewportRender, "_ViewportRender__render_region", return_value=mock_region), \
            patch.object(ViewportRender, "_ViewportRender__flatten_edits") as mock_flatten:

        viewport_render.render_area(mock_layer, real_viewport)

        mock_flatten.assert_not_called()
        final_region_mock.overlap_with.assert_called_once_with(mock_region)
        mock_cached_image.crop.assert_called_once_with(mock_view_coord)


def test_render_area_populates_cache_at_scale_1(viewport_render, mock_layer, real_viewport):
    """Cenário 5: Ao renderizar com escala 1.0, o resultado deve ser salvo no cache."""
    mock_region = MagicMock(spec=Region)
    mock_region.size = (800, 600)
    mock_layer._resolve_render.return_value = RenderFlags.PIXELS

    # Define escala 1.0 na viewport (já é o padrão, mas reforçamos)
    real_viewport.scale = Scale(1.0, 1.0)

    mock_rendered_image = MagicMock(spec=Image)
    with patch.object(ViewportRender, "_final_region", return_value=mock_region), \
            patch.object(ViewportRender, "_ViewportRender__render_region", return_value=mock_region), \
            patch.object(ViewportRender, "_ViewportRender__flatten_edits", return_value=mock_rendered_image), \
            patch("anicrop.render.Image.new"):

        viewport_render.render_area(mock_layer, real_viewport)

        # Verifica se o resultado foi salvo no cache
        assert viewport_render._cache[mock_layer._id] is mock_rendered_image

# --- Grupo de Testes: ViewportRender._final_region ---


def test_final_region_1to1(viewport_render, mock_layer, real_viewport):
    """Cenário 6: Alinhamento 1:1. Layer e Viewport idênticos."""
    mock_layer.region = Region.from_size(800, 600)

    # Matriz global do layer é a identidade
    with patch("anicrop.render.mat_global", return_value=np.eye(3)):
        region = viewport_render._final_region(mock_layer, real_viewport)

        assert region.top_left == (0, 0)
        assert region.size == (800, 600)


def test_final_region_with_fit_scale(viewport_render, mock_layer):
    """Cenário 7: Layer gigante (4k) ajustado para Viewport pequena via fit_scale."""
    mock_layer.region = Region.from_size(4000, 3000)
    # Viewport 800x600 com fit_scale 0.2 (4000 * 0.2 = 800)
    viewport = Viewport((800, 600), 0.2)

    with patch("anicrop.render.mat_global", return_value=np.eye(3)):
        region = viewport_render._final_region(mock_layer, viewport)

        assert region.top_left == (0, 0)
        assert region.size == (800, 600)


def test_final_region_with_viewport_zoom(viewport_render, mock_layer, real_viewport):
    """Cenário 8: Viewport com Zoom de 2x (Centralizado)."""
    mock_layer.region = Region.from_size(800, 600)
    real_viewport.scale = Scale(2.0, 2.0)

    real_viewport.scale = Scale(2.0, 2.0)

    with patch("anicrop.render.mat_global", return_value=np.eye(3)):
        region = viewport_render._final_region(mock_layer, real_viewport)

        # No zoom de 2x centralizado, o ponto (0,0) vai para (-400, -300)
        # e o tamanho 800x600 vira 1600x1200 na tela
        assert region.top_left == (-400, -300)
        assert region.size == (1600, 1200)


def test_final_region_layer_offset_in_world(viewport_render, mock_layer, real_viewport):
    """Cenário 9: Layer deslocado via region.top_left, Viewport estática."""
    # Modifica a região real (Region é imutável no anicrop, então somamos o offset)
    mock_layer.region = Region.from_size(800, 600) + (100, 200)

    # mat_global(layer) agora retorna a matriz que posiciona o layer.
    m_layer_global = mat_position(mock_layer.region)

    with patch("anicrop.render.mat_global", return_value=m_layer_global):
        region = viewport_render._final_region(mock_layer, real_viewport)

        assert region.top_left == (100, 200)
        assert region.size == (800, 600)
