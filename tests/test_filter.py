import math

import numpy as np
import pytest

from anicrop.canvas import Canvas
from anicrop.effect import Effect
from anicrop.enums import BlurMode, ImageFormat
from anicrop.filter import BlurFilter
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.mask import Mask
from anicrop.render import CanvasRender
from anicrop.spatial import Region


def make_checkerboard_image(w: int = 40, h: int = 40) -> Image:
    """Cria imagem de teste com padrão xadrez de alto contraste para testar suavização."""
    data = np.zeros((h, w, 4), dtype=np.uint8)
    data[: h // 2, : w // 2] = [255, 255, 255, 255]
    data[h // 2 :, w // 2 :] = [255, 255, 255, 255]
    data[: h // 2, w // 2 :] = [0, 0, 0, 255]
    data[h // 2 :, : w // 2] = [0, 0, 0, 255]
    return Image(data, ImageFormat.RGBA)


def test_blur_filter_satisfies_effect_protocol():
    """Valida se BlurFilter atende formalmente ao protocolo Effect."""
    blur = BlurFilter(radius=5.0)
    assert isinstance(blur, Effect)


@pytest.mark.parametrize(
    "radius_input, expected_rx, expected_ry",
    [
        pytest.param(5.0, 5.0, 5.0, id="raio_escalar_isotropico"),
        pytest.param((10.0, 2.0), 10.0, 2.0, id="raio_tupla_anisotropico"),
        pytest.param(0, 0.0, 0.0, id="raio_zero"),
    ],
)
def test_blur_filter_radius_normalization(radius_input, expected_rx, expected_ry):
    """Valida normalização dos raios horizontal e vertical no construtor."""
    blur = BlurFilter(radius=radius_input)
    assert blur.radius_x == expected_rx
    assert blur.radius_y == expected_ry


@pytest.mark.parametrize(
    "radius, mode, affect_alpha, expected_padding",
    [
        pytest.param(
            3.0, BlurMode.GAUSSIAN, True, (9, 9, 9, 9), id="gaussiano_afeta_alfa_3sigma"
        ),
        pytest.param(
            (4.0, 2.0),
            BlurMode.GAUSSIAN,
            True,
            (6, 12, 6, 12),
            id="gaussiano_anisotropico_afeta_alfa",
        ),
        pytest.param(5.0, BlurMode.BOX, True, (5, 5, 5, 5), id="box_blur_afeta_alfa"),
        pytest.param(
            5.0,
            BlurMode.GAUSSIAN,
            False,
            (0, 0, 0, 0),
            id="alfa_nao_afetado_padding_zero",
        ),
    ],
)
def test_blur_filter_get_padding(radius, mode, affect_alpha, expected_padding):
    """Valida cálculo de padding de expansão espacial para diferentes modos e raios."""
    blur = BlurFilter(radius=radius, mode=mode, affect_alpha=affect_alpha)
    assert blur.get_padding() == expected_padding


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(BlurMode.GAUSSIAN, id="modo_gaussiano"),
        pytest.param(BlurMode.BOX, id="modo_box"),
    ],
)
def test_blur_filter_apply_smooths_checkerboard_contrast(mode):
    """Valida se desfoques gaussianos e de média reduzem o contraste local do padrão xadrez."""
    img = make_checkerboard_image(40, 40)
    blur = BlurFilter(radius=4.0, mode=mode)
    ident = np.identity(3, dtype=np.float32)
    result = blur.apply(img, ident)

    # Na junção central dos 4 quadrantes (y=20, x=20), a transição nítida 0/255 vira valor intermediário
    center_val = int(result[20, 20, 0])
    assert 50 < center_val < 200


def test_blur_filter_apply_median_removes_impulse_noise():
    """Valida se o filtro mediano elimina ruído impulsivo (pixels isolados) preservando a área uniforme."""
    data = np.zeros((30, 30, 4), dtype=np.uint8)
    data[..., -1] = 255
    data[15, 15, :3] = 255
    img = Image(data, ImageFormat.RGBA)

    blur = BlurFilter(radius=2.0, mode=BlurMode.MEDIAN)
    ident = np.identity(3, dtype=np.float32)
    result = blur.apply(img, ident)

    assert result[15, 15, 0] == 0


def test_blur_filter_affect_alpha_false_preserves_exact_alpha_channel():
    """Valida se affect_alpha=False mantém o canal alfa original sem modificação."""
    data = np.full((30, 30, 4), 255, dtype=np.uint8)
    data[:15, :, -1] = 100
    data[15:, :, -1] = 200
    img = Image(data, ImageFormat.RGBA)

    blur = BlurFilter(radius=5.0, affect_alpha=False)
    ident = np.identity(3, dtype=np.float32)
    result = blur.apply(img, ident)

    np.testing.assert_array_equal(result[..., -1], img[..., -1])


def test_blur_filter_strength_zero_returns_unmodified_image():
    """Valida se strength=0.0 retorna o buffer sem modificações."""
    img = make_checkerboard_image(20, 20)
    blur = BlurFilter(radius=5.0, strength=0.0)
    ident = np.identity(3, dtype=np.float32)
    result = blur.apply(img, ident)

    np.testing.assert_array_equal(result[...], img[...])


def test_blur_filter_merge_gaussian_radii():
    """Valida se merge de dois BlurFilters gaussianos combina os raios via soma pitagórica."""
    blur1 = BlurFilter(radius=(3.0, 4.0), mode=BlurMode.GAUSSIAN, affect_alpha=True)
    blur2 = BlurFilter(radius=(4.0, 3.0), mode=BlurMode.GAUSSIAN, affect_alpha=True)

    matrix = np.identity(3, dtype=np.float32)
    merged = blur1.merge(blur2, matrix)

    assert merged is not None
    assert isinstance(merged, BlurFilter)
    assert pytest.approx(merged.radius_x, 0.01) == 5.0
    assert pytest.approx(merged.radius_y, 0.01) == 5.0
    assert merged.affect_alpha is True


def test_blur_filter_merge_orthogonal_produces_isotropic_gaussian():
    """Valida se fusão de desfoque horizontal e vertical a 90 graus gera Gaussiana circular isotrópica."""
    blur_h = BlurFilter(radius=(10.0, 0.0), angle=0.0, mode=BlurMode.GAUSSIAN)
    blur_v = BlurFilter(radius=(10.0, 0.0), angle=90.0, mode=BlurMode.GAUSSIAN)

    matrix = np.identity(3, dtype=np.float32)
    merged = blur_h.merge(blur_v, matrix)

    assert merged is not None
    assert isinstance(merged, BlurFilter)
    assert pytest.approx(merged.radius_x, 0.01) == 10.0
    assert pytest.approx(merged.radius_y, 0.01) == 10.0


def test_blur_filter_merge_with_different_base_matrices():
    """Valida se fusão de filtros com matrizes de base distintas calcula a covariância relativa correta."""
    rad90 = math.radians(90.0)
    mat_rot90 = np.array(
        [
            [math.cos(rad90), -math.sin(rad90), 0.0],
            [math.sin(rad90), math.cos(rad90), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # blur1 local horizontal em base identidade (0 graus na tela)
    blur1 = BlurFilter(radius=(6.0, 0.0), angle=0.0)
    # blur2 local horizontal
    blur2 = BlurFilter(radius=(8.0, 0.0), angle=0.0)

    # Funde blur2 em blur1 aplicando a matriz de rotação relativa de 90 graus
    merged = blur1.merge(blur2, mat_rot90)

    assert merged is not None
    # No espaço unificado, as variâncias ortogonais são 6^2 e 8^2, gerando raios 8 e 6
    assert pytest.approx(merged.radius_x, 0.01) == 8.0
    assert pytest.approx(merged.radius_y, 0.01) == 6.0


def test_blur_filter_merge_incompatible_returns_none():
    """Valida se merge retorna None quando os modos ou configurações forem incompatíveis."""
    blur_gaussian = BlurFilter(radius=3.0, mode=BlurMode.GAUSSIAN)
    blur_box = BlurFilter(radius=3.0, mode=BlurMode.BOX)

    matrix = np.identity(3, dtype=np.float32)
    assert blur_gaussian.merge(blur_box, matrix) is None


def test_blur_filter_1d_horizontal_only_blurs_x():
    """Valida se radius=(r, 0) desfoca exclusivamente no eixo horizontal mantendo colunas perfeitas."""
    data = np.zeros((30, 30, 4), dtype=np.uint8)
    data[..., -1] = 255
    data[:, 15, :3] = 255
    img = Image(data, ImageFormat.RGBA)

    blur = BlurFilter(radius=(5.0, 0.0), mode=BlurMode.GAUSSIAN)
    ident = np.identity(3, dtype=np.float32)
    result = blur.apply(img, ident)

    assert result[15, 14, 0] > 0
    assert result[15, 16, 0] > 0
    np.testing.assert_array_equal(result[0, :, 0], result[29, :, 0])


def test_blur_filter_1d_vertical_only_blurs_y():
    """Valida se radius=(0, r) desfoca exclusivamente no eixo vertical mantendo linhas perfeitas."""
    data = np.zeros((30, 30, 4), dtype=np.uint8)
    data[..., -1] = 255
    data[15, :, :3] = 255
    img = Image(data, ImageFormat.RGBA)

    blur = BlurFilter(radius=(0.0, 5.0), mode=BlurMode.GAUSSIAN)
    ident = np.identity(3, dtype=np.float32)
    result = blur.apply(img, ident)

    assert result[14, 15, 0] > 0
    assert result[16, 15, 0] > 0
    np.testing.assert_array_equal(result[:, 0, 0], result[:, 29, 0])


def test_blur_filter_directional_angle_diffuses_along_diagonal():
    """Valida se angle=45.0 projeta a difusão no sentido diagonal."""
    data = np.zeros((31, 31, 4), dtype=np.uint8)
    data[..., -1] = 255
    data[15, 15, :3] = 255
    img = Image(data, ImageFormat.RGBA)

    blur = BlurFilter(radius=8.0, angle=45.0, mode=BlurMode.GAUSSIAN)
    ident = np.identity(3, dtype=np.float32)
    result = blur.apply(img, ident)

    assert result[17, 17, 0] > 0
    assert result[13, 13, 0] > 0
    assert result[25, 5, 0] == 0


def test_blur_filter_apply_with_rotated_matrix_adjusts_effective_angle():
    """Valida se uma matriz rotacionada em 45 graus ajusta automaticamente o ângulo de desfoque."""
    data = np.zeros((31, 31, 4), dtype=np.uint8)
    data[..., -1] = 255
    data[15, 15, :3] = 255
    img = Image(data, ImageFormat.RGBA)

    blur = BlurFilter(radius=8.0, angle=0.0, mode=BlurMode.GAUSSIAN)

    rad = math.radians(45.0)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    matrix = np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = blur.apply(img, matrix)

    assert result[17, 17, 0] > 0
    assert result[13, 13, 0] > 0
    assert result[25, 5, 0] == 0


def test_render_layer_with_blur_filter_in_pipeline():
    """Valida a renderização completa de uma camada com BlurFilter aplicado via CanvasRender."""
    canvas = Canvas.from_size(100, 100)
    data = np.zeros((40, 40, 4), dtype=np.uint8)
    data[:] = [255, 0, 0, 255]
    layer = Layer(Image(data, ImageFormat.RGBA))
    layer.transform.translate(30, 30)

    blur = BlurFilter(radius=4.0, affect_alpha=True)
    layer.add_effect(blur)

    renderer = CanvasRender()
    result = renderer.render_scene([layer], canvas)

    assert result[28, 50, 3] > 0


def test_render_layer_with_masked_blur_filter():
    """Valida a aplicação seletiva de BlurFilter restrito a uma máscara em BaseLayer."""
    canvas = Canvas.from_size(50, 50)
    data = np.zeros((40, 40, 4), dtype=np.uint8)
    data[:20, :20] = [255, 255, 255, 255]
    data[20:, 20:] = [255, 255, 255, 255]
    layer = Layer(Image(data, ImageFormat.RGBA))

    # Máscara branca na metade superior (y < 20) e preta na inferior
    mask_data = np.zeros((40, 40, 1), dtype=np.uint8)
    mask_data[:20, :] = 255
    mask = Mask(
        Image(mask_data, ImageFormat.GRAY),
        Region.from_size(40, 40),
        np.identity(3, dtype=np.float32),
    )

    blur = BlurFilter(radius=6.0)
    layer.bind_effect(blur, mask=mask)

    renderer = CanvasRender()
    result = renderer.render_scene([layer], canvas)

    # Na parte superior mascarada (y=10), o centro (x=20) foi suavizado
    assert result[10, 20, 0] > 0
    # Na parte inferior não mascarada (y=30), o centro nítido permaneceu preto (0)
    assert result[30, 10, 0] == 0
