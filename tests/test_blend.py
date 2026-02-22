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
    # O canal Alpha da base deve ter permanecido zerado (intacto),
    # pois a sobreposição opaca não injeta Alpha.
    assert np.all(result._data[..., 3] == 0)


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
    # O canal alpha (índice 1) deve continuar 0
    assert np.all(result._data[..., 1] == 0)


# --- Testes para blend_normal ---
from anicrop.blend import blend_normal


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

