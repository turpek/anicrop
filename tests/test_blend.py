from anicrop.blend import blend_normal
from anicrop.image import Image, ImageFormat
from anicrop.blend import hard_masking
import numpy as np
import pytest


def test_hard_masking_size_mismatch():
    """Garante que a função rejeite matrizes de tamanhos físicos diferentes."""
    base = Image(np.zeros((10, 10, 3), dtype=np.uint8), ImageFormat.RGB)
    overlay = Image(np.zeros((5, 5, 3), dtype=np.uint8), ImageFormat.RGB)

    with pytest.raises(ValueError, match="Size mismatch"):
        hard_masking(base, overlay)


def test_hard_masking_color_space_mismatch():
    """Garante que a função rejeite misturas de espaços de cor não suportados (ex: RGB com CMYK)."""
    base = Image(np.zeros((10, 10, 3), dtype=np.uint8), ImageFormat.RGB)
    overlay = Image(np.zeros((10, 10, 4), dtype=np.uint8), ImageFormat.CMYK)

    with pytest.raises(NotImplementedError, match="Format mismatch"):
        hard_masking(base, overlay)


def test_hard_masking_opaque_to_opaque():
    """Testa a cópia direta entre duas imagens opacas do mesmo shape (RGB -> RGB)."""
    base_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    over_arr = np.ones((10, 10, 3), dtype=np.uint8) * 255  # Overlay todo branco

    base = Image(base_arr, ImageFormat.RGB)
    overlay = Image(over_arr, ImageFormat.RGB)

    result = hard_masking(base, overlay)

    # A base inteira deve ter sido substituída pelo overlay
    assert np.array_equal(result._data, over_arr)


def test_hard_masking_alpha_to_alpha():
    """Testa o uso da máscara booleana entre imagens do mesmo shape (RGBA -> RGBA)."""
    base_arr = np.zeros((10, 10, 4), dtype=np.uint8)

    # Overlay: Metade esquerda 100% transparente, metade direita vermelho sólido
    over_arr = np.zeros((10, 10, 4), dtype=np.uint8)
    over_arr[:, 5:, 0] = 255  # Canal R (Red) na direita
    over_arr[:, 5:, 3] = 255  # Canal Alpha na direita

    base = Image(base_arr, ImageFormat.RGBA)
    overlay = Image(over_arr, ImageFormat.RGBA)

    result = hard_masking(base, overlay)

    # Metade esquerda deve continuar preta/zerada (não foi copiada)
    assert np.all(result._data[:, :5] == 0)
    # Metade direita deve ser igual ao overlay (vermelha)
    assert np.all(result._data[:, 5:, 0] == 255)
    assert np.all(result._data[:, 5:, 3] == 255)


def test_hard_masking_transparent_overlay_on_opaque_base():
    """Testa colar uma imagem com Alpha em um Canvas sem Alpha (RGBA -> RGB)."""
    # Base: 10x10 verde sólido RGB
    base_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    base_arr[..., 1] = 255

    # Overlay: 10x10 RGBA (Apenas um pixel no canto superior esquerdo é Vermelho, o resto é transparente)
    over_arr = np.zeros((10, 10, 4), dtype=np.uint8)
    over_arr[0, 0] = [255, 0, 0, 255]  # Pixel [0,0] = Vermelho Sólido

    base = Image(base_arr, ImageFormat.RGB)
    overlay = Image(over_arr, ImageFormat.RGBA)

    result = hard_masking(base, overlay)

    # O pixel [0,0] da base deve ter virado vermelho
    assert np.array_equal(result._data[0, 0], [255, 0, 0])
    # Um pixel qualquer fora da máscara (ex: [1,1]) deve continuar verde
    assert np.array_equal(result._data[1, 1], [0, 255, 0])
    # O shape da base deve continuar 3D com 3 canais
    assert result.shape == (10, 10, 3)


def test_hard_masking_opaque_overlay_on_transparent_base():
    """Testa colar uma imagem sem Alpha em um Canvas com Alpha (RGB -> RGBA)."""
    # Base: 10x10 transparente
    base_arr = np.zeros((10, 10, 4), dtype=np.uint8)

    # Overlay: 10x10 azul sólido RGB
    over_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    over_arr[..., 2] = 255

    base = Image(base_arr, ImageFormat.RGBA)
    overlay = Image(over_arr, ImageFormat.RGB)

    result = hard_masking(base, overlay)

    # Os canais RGB da base devem ter recebido o azul
    assert np.all(result._data[..., 2] == 255)
    # O canal Alpha da base deve ficar totalmente opaco (255),
    # pois a sobreposição opaca substitui a transparência.
    assert np.all(result._data[..., 3] == 255)


def test_hard_masking_grayscale():
    """Testa a integração com tons de cinza forçados para 3D (GRAY -> GRAY_ALPHA)."""
    # Base: 10x10 GRAY_ALPHA (2 canais)
    base_arr = np.zeros((10, 10, 2), dtype=np.uint8)

    # Overlay: 10x10 GRAY (inicializado nativamente como 2D pelo usuário/OpenCV)
    over_arr = np.ones((10, 10), dtype=np.uint8) * 128

    base = Image(base_arr, ImageFormat.GRAY_ALPHA)
    # A inicialização da Image deve converter o over_arr para (10, 10, 1) automaticamente
    overlay = Image(over_arr, ImageFormat.GRAY)

    result = hard_masking(base, overlay)

    # O canal de cor (índice 0) deve ser 128
    assert np.all(result._data[..., 0] == 128)
    # O canal alpha (índice 1) deve ficar opaco (255)
    assert np.all(result._data[..., 1] == 255)


# --- Testes para blend_normal ---


def test_blend_normal_size_mismatch_clipping():
    """
    Testa se blend_normal recorta (clip) corretamente quando as imagens
    têm tamanhos diferentes, operando na interseção (min_w, min_h).
    """
    # Base: 10x10 Azul
    base_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    base_arr[..., 2] = 255
    base = Image(base_arr, ImageFormat.RGB)

    # Edit: 20x20 Vermelho (Sò vai usar os primeiros 10x10)
    edit_arr = np.zeros((20, 20, 3), dtype=np.uint8)
    edit_arr[..., 0] = 255
    edit = Image(edit_arr, ImageFormat.RGB)

    blend_normal(base, edit)

    # Base inteira (10x10) deve virar vermelha
    assert np.all(base._data[..., 0] == 255)
    assert np.all(base._data[..., 2] == 0)


def test_blend_normal_alpha_blending_50_percent():
    """
    Testa a matemática do blend: 50% Vermelho sobre Azul = Roxo Escuro/Misturado.
    Formula: Out = Src * alpha + Dst * (1 - alpha)
    """
    # Base: Azul sólido (0, 0, 255)
    base_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    base_arr[..., 2] = 255
    base = Image(base_arr, ImageFormat.RGB)

    # Edit: Vermelho com 50% de Alpha (255, 0, 0, 128)
    # Alpha ~0.502 (128/255)
    edit_arr = np.zeros((10, 10, 4), dtype=np.uint8)
    edit_arr[..., 0] = 255
    edit_arr[..., 3] = 128
    edit = Image(edit_arr, ImageFormat.RGBA)

    blend_normal(base, edit)

    # Calculo esperado:
    # R: 255 * 0.502 + 0 * 0.498 = ~128
    # G: 0
    # B: 0 * 0.502 + 255 * 0.498 = ~127
    px = base._data[0, 0]
    assert 126 <= px[0] <= 129  # Margem de erro de arredondamento
    assert px[1] == 0
    assert 126 <= px[2] <= 129


def test_blend_normal_full_transparency_skipped():
    """Garante que áreas com Alpha 0 no edit não alteram a base."""
    # Base: Branco
    base_arr = np.ones((10, 10, 3), dtype=np.uint8) * 255
    base = Image(base_arr, ImageFormat.RGB)

    # Edit: Preto totalmente transparente
    edit_arr = np.zeros((10, 10, 4), dtype=np.uint8)
    # Alpha já é 0
    edit = Image(edit_arr, ImageFormat.RGBA)

    blend_normal(base, edit)

    # Base continua branca
    assert np.all(base._data == 255)


def test_blend_normal_sets_base_alpha_to_opaque():
    """
    Se a base tem canal Alpha, blend_normal deve forçar o Alpha da base
    para 255 nas áreas afetadas, garantindo opacidade do resultado.
    """
    # Base: Transparente (0,0,0,0)
    base_arr = np.zeros((10, 10, 4), dtype=np.uint8)
    base = Image(base_arr, ImageFormat.RGBA)

    # Edit: Vermelho Opaco
    edit_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    edit_arr[..., 0] = 255
    edit = Image(edit_arr, ImageFormat.RGB)

    blend_normal(base, edit)

    # Base deve ser Vermelha e Opaca
    assert np.all(base._data[..., 0] == 255)
    assert np.all(base._data[..., 3] == 255)


@pytest.mark.parametrize(
    "base_fmt, edit_fmt, edit_alpha, opacity, expected_copyto",
    [
        pytest.param(ImageFormat.RGBA, ImageFormat.RGBA, 255, 1.0, True,
                     id="rgba_solido_com_opacidade_total_usa_copyto"),
        pytest.param(ImageFormat.RGB, ImageFormat.RGB, None, 1.0, True,
                     id="rgb_solido_com_opacidade_total_usa_copyto"),
        pytest.param(ImageFormat.RGBA, ImageFormat.RGBA, 128, 1.0,
                     False, id="rgba_semi_transparente_nao_usa_copyto"),
        pytest.param(ImageFormat.RGBA, ImageFormat.RGBA, 255, 0.5, False,
                     id="rgba_solido_com_opacidade_reduzida_nao_usa_copyto"),
    ],
)
def test_blend_normal_fast_path_copyto(mocker, base_fmt, edit_fmt, edit_alpha, opacity, expected_copyto):
    """Valida se o fast-path de np.copyto é ativado exclusivamente para camadas sólidas com opacidade 1.0."""
    spy_copyto = mocker.spy(np, "copyto")
    base_channels = 4 if base_fmt == ImageFormat.RGBA else 3
    edit_channels = 4 if edit_fmt == ImageFormat.RGBA else 3

    base = Image(np.zeros((10, 10, base_channels), dtype=np.uint8), base_fmt)
    edit_arr = np.full((10, 10, edit_channels), 200, dtype=np.uint8)
    if edit_alpha is not None and edit_channels == 4:
        edit_arr[..., 3] = edit_alpha
    edit = Image(edit_arr, edit_fmt)

    blend_normal(base, edit, opacity=opacity)

    assert spy_copyto.called is expected_copyto


@pytest.mark.parametrize(
    "base_fmt, edit_fmt, edit_color, expected_pixel",
    [
        pytest.param(ImageFormat.RGBA, ImageFormat.RGB, [255, 0, 0], [
                     255, 0, 0, 255], id="base_rgba_edit_rgb"),
        pytest.param(ImageFormat.RGB, ImageFormat.RGBA, [0, 255, 0, 255], [
                     0, 255, 0], id="base_rgb_edit_rgba"),
        pytest.param(ImageFormat.GRAY, ImageFormat.RGB, [
                     255, 0, 0], [76], id="base_gray_edit_red"),
        pytest.param(ImageFormat.RGB, ImageFormat.GRAY, [128], [
                     128, 128, 128], id="base_rgb_edit_gray"),
        pytest.param(ImageFormat.GRAY, ImageFormat.GRAY, [
                     200], [200], id="base_gray_edit_gray"),
        pytest.param(ImageFormat.GRAY_ALPHA, ImageFormat.GRAY_ALPHA, [
                     200, 255], [200, 255], id="base_gray_alpha_edit_gray_alpha"),
        pytest.param(ImageFormat.RGBA, ImageFormat.GRAY_ALPHA, [100, 255], [
                     100, 100, 100, 255], id="base_rgba_edit_gray_alpha"),
        pytest.param(ImageFormat.GRAY_ALPHA, ImageFormat.RGB, [
                     255, 0, 0], [76, 255], id="base_gray_alpha_edit_rgb"),
    ],
)
def test_blend_normal_formatos_mistos(base_fmt, edit_fmt, edit_color, expected_pixel):
    """Valida se blend_normal harmoniza e converte formatos mistos corretamente."""
    base = Image(np.zeros((10, 10, base_fmt.channels), dtype=np.uint8), base_fmt)
    edit_arr = np.zeros((10, 10, edit_fmt.channels), dtype=np.uint8)
    edit_arr[:] = edit_color
    edit = Image(edit_arr, edit_fmt)

    blend_normal(base, edit, opacity=1.0)

    np.testing.assert_array_equal(base[0, 0], expected_pixel)
