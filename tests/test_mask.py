import numpy as np
import pytest

from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import EditLayer
from anicrop.mask import Mask
from anicrop.spatial import Region


def make_target_image(w: int = 20, h: int = 20, alpha: int = 255, fmt: ImageFormat = ImageFormat.RGBA) -> Image:
    channels = 4 if fmt == ImageFormat.RGBA else 2
    data = np.full((h, w, channels), 200, dtype=np.uint8)
    data[..., -1] = alpha
    return Image(data, fmt)


def make_mask_image(w: int = 20, h: int = 20, value: int = 255, fmt: ImageFormat = ImageFormat.GRAY) -> Image:
    channels = fmt.channels
    data = np.full((h, w, channels), value, dtype=np.uint8)
    return Image(data, fmt)


def test_mask_inherits_from_edit_layer():
    """Valida se a classe Mask herda corretamente de EditLayer e preserva atributos espaciais."""
    mask_img = make_mask_image(30, 40)
    region = Region.from_rect(5, 5, 30, 40)
    matrix = np.identity(3, dtype=np.float32)
    mask = Mask(mask_img, region=region, matrix=matrix, name="CustomMask")

    assert isinstance(mask, EditLayer)
    assert mask.image.size == (30, 40)
    assert mask.region == Region.from_rect(5, 5, 30, 40)
    assert mask.name == "CustomMask"
    assert mask.get_padding() == (0, 0, 0, 0)
    assert mask.invert is False


@pytest.mark.parametrize(
    "mask_value, initial_alpha, invert, expected_alpha",
    [
        pytest.param(255, 255, False, 255, id="mascara_branca_mantem_opacidade_total"),
        pytest.param(0, 255, False, 0, id="mascara_preta_torna_totalmente_transparente"),
        pytest.param(128, 255, False, 128,
                     id="mascara_cinza_reduz_opacidade_pela_metade"),
        pytest.param(128, 100, False, 50, id="mascara_cinza_modula_alfa_preexistente"),
        pytest.param(0, 255, True, 255, id="mascara_preta_invertida_mantem_opacidade"),
        pytest.param(255, 255, True, 0,
                     id="mascara_branca_invertida_torna_transparente"),
    ],
)
def test_mask_apply_rgba_modulation(mask_value, initial_alpha, invert, expected_alpha):
    """Valida modulação do canal alfa em imagens RGBA de acordo com o valor da máscara e flag invert."""
    target = make_target_image(10, 10, alpha=initial_alpha, fmt=ImageFormat.RGBA)
    mask_img = make_mask_image(10, 10, value=mask_value, fmt=ImageFormat.GRAY)
    region = Region.from_size(10, 10)
    matrix = np.identity(3, dtype=np.float32)
    mask = Mask(mask_img, region, matrix, invert=invert)

    result = mask.apply_modulation(target, mask_img)

    assert result is target
    np.testing.assert_array_equal(result[..., -1], expected_alpha)


@pytest.mark.parametrize(
    "mask_format",
    [
        pytest.param(ImageFormat.GRAY, id="mascara_gray_1_canal"),
        pytest.param(ImageFormat.GRAY_ALPHA, id="mascara_gray_alpha_2_canais"),
        pytest.param(ImageFormat.RGB, id="mascara_rgb_3_canais"),
        pytest.param(ImageFormat.RGBA, id="mascara_rgba_4_canais"),
    ],
)
def test_mask_apply_supports_all_mask_formats(mask_format):
    """Valida se Mask extrai corretamente a luminância de qualquer formato de imagem suportado."""
    target = make_target_image(10, 10, alpha=255, fmt=ImageFormat.RGBA)
    mask_img = make_mask_image(10, 10, value=0, fmt=mask_format)
    region = Region.from_size(10, 10)
    matrix = np.identity(3, dtype=np.float32)
    mask = Mask(mask_img, region, matrix)

    result = mask.apply_modulation(target, mask_img)

    np.testing.assert_array_equal(result[..., -1], 0)


def test_mask_merge_with_another_mask_unifies_regions():
    """Valida se merge de duas instâncias de Mask unifica suas regiões e combina suas imagens."""
    mask_img1 = make_mask_image(10, 10, value=255)
    mask1 = Mask(mask_img1, Region.from_rect(0, 0, 10, 10), np.identity(3))

    mask_img2 = make_mask_image(10, 10, value=128)
    mask2 = Mask(mask_img2, Region.from_rect(10, 0, 10, 10), np.identity(3))

    matrix = np.identity(3, dtype=np.float32)
    merged = mask1.merge(mask2, matrix)

    assert merged is not None
    assert merged.region == Region.from_rect(0, 0, 20, 10)
    assert merged.image.size == (20, 10)
    assert merged.matrix is matrix
    np.testing.assert_array_equal(merged.image[:10, :10, 0], 255)
    np.testing.assert_array_equal(merged.image[:10, 10:20, 0], 128)


def test_mask_merge_with_incompatible_effect_returns_none():
    """Valida se merge retorna None quando o efeito fornecido não for uma Mask."""
    class FakeEffect:
        def __init__(self):
            self.visible = True
            self.matrix = np.identity(3, dtype=np.float32)

        def prepare(self, frame): pass
        def get_padding(self): return (0, 0, 0, 0)
        def apply(self, img, mat=None): return img
        def merge(self, other, mat): return None

    mask_img = make_mask_image(10, 10)
    mask = Mask(mask_img, Region.from_size(10, 10), np.identity(3))

    result = mask.merge(FakeEffect(), np.identity(3))

    assert result is None


def test_mask_indexing_supports_slice_ellipsis_and_region():
    """Valida se Mask.__getitem__ e __setitem__ operam com slices, Ellipsis e Region."""
    mask_img = make_mask_image(20, 20, value=255)
    mask = Mask(mask_img, Region.from_size(20, 20), np.identity(3))

    # 1. Slicing convencional
    mask[0:10, 0:10] = 0
    assert mask[5, 5, 0] == 0
    assert mask[15, 15, 0] == 255

    # 2. Indexação com Region
    sub_region = Region.from_rect(10, 10, 5, 5)
    mask[sub_region] = 50
    assert mask[12, 12, 0] == 50

    # 3. Ellipsis
    mask[...] = 100
    np.testing.assert_array_equal(mask[...], 100)


def test_mask_apply_when_invisible_returns_original_image():
    """Valida se Mask.apply com visible=False retorna a imagem intacta sem modular."""
    target = make_target_image(10, 10, alpha=255, fmt=ImageFormat.RGBA)
    mask_img = make_mask_image(10, 10, value=0, fmt=ImageFormat.GRAY)
    mask = Mask(mask_img, Region.from_size(10, 10), np.identity(3), visible=False)

    result = mask.apply(target, np.identity(3))

    assert result is target
    np.testing.assert_array_equal(result[..., -1], 255)
