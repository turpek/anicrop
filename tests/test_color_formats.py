from __future__ import annotations

import numpy as np
import pytest

from anicrop.color import convert_image_format
from anicrop.enums import ImageFormat
from anicrop.image import Image


def test_image_format_enums_properties() -> None:
    """Valida as propriedades de canais e alfa dos formatos PRGBA e RGBX."""
    assert ImageFormat.PRGBA.has_alpha is True
    assert ImageFormat.PRGBA.is_premultiplied is True
    assert ImageFormat.PRGBA.channels == 4

    assert ImageFormat.RGBX.has_alpha is False
    assert ImageFormat.RGBX.is_premultiplied is False
    assert ImageFormat.RGBX.channels == 4


def test_rgba_to_prgba_and_back_opaque() -> None:
    """Valida que pixels 100% opacos preservam as cores exatas no roundtrip RGBA <-> PRGBA."""
    raw = np.array([[[200, 100, 50, 255]]], dtype=np.uint8)
    prgba = convert_image_format(raw, ImageFormat.RGBA, ImageFormat.PRGBA)
    rgba = convert_image_format(prgba, ImageFormat.PRGBA, ImageFormat.RGBA)

    np.testing.assert_array_equal(prgba, raw)
    np.testing.assert_array_equal(rgba, raw)


def test_rgba_to_prgba_and_back_translucent() -> None:
    """Valida a multiplicação e desmultiplicação de cores com 50% de transparência."""
    raw = np.array([[[200, 100, 50, 128]]], dtype=np.uint8)
    prgba = convert_image_format(raw, ImageFormat.RGBA, ImageFormat.PRGBA)

    # 200 * 128 / 255 = 100.39 -> 100
    # 100 * 128 / 255 = 50.19 -> 50
    # 50 * 128 / 255 = 25.09 -> 25
    expected_prgba = np.array([[[100, 50, 25, 128]]], dtype=np.uint8)
    np.testing.assert_array_equal(prgba, expected_prgba)

    rgba = convert_image_format(prgba, ImageFormat.PRGBA, ImageFormat.RGBA)
    # 100 * 255 / 128 = 199.2 -> 199
    # 50 * 255 / 128 = 99.6 -> 100
    # 25 * 255 / 128 = 49.8 -> 50
    expected_rgba = np.array([[[199, 100, 50, 128]]], dtype=np.uint8)
    np.testing.assert_array_equal(rgba, expected_rgba)


def test_rgba_to_prgba_fully_transparent() -> None:
    """Valida que pixels totalmente transparentes não geram divisão por zero ao desmultiplicar."""
    raw = np.array([[[255, 128, 64, 0]]], dtype=np.uint8)
    prgba = convert_image_format(raw, ImageFormat.RGBA, ImageFormat.PRGBA)
    rgba = convert_image_format(prgba, ImageFormat.PRGBA, ImageFormat.RGBA)

    np.testing.assert_array_equal(prgba, np.array([[[0, 0, 0, 0]]], dtype=np.uint8))
    np.testing.assert_array_equal(rgba, np.array([[[0, 0, 0, 0]]], dtype=np.uint8))


def test_rgb_to_rgbx_and_back() -> None:
    """Valida o roundtrip entre RGB e RGBX adicionando e removendo o canal de padding."""
    raw = np.array([[[10, 20, 30]]], dtype=np.uint8)
    rgbx = convert_image_format(raw, ImageFormat.RGB, ImageFormat.RGBX)
    rgb = convert_image_format(rgbx, ImageFormat.RGBX, ImageFormat.RGB)

    np.testing.assert_array_equal(rgbx, np.array([[[10, 20, 30, 255]]], dtype=np.uint8))
    np.testing.assert_array_equal(rgb, raw)


@pytest.mark.parametrize(
    "src_fmt,dst_fmt,in_shape,out_shape",
    [
        (ImageFormat.RGB, ImageFormat.PRGBA, (10, 10, 3), (10, 10, 4)),
        (ImageFormat.PRGBA, ImageFormat.RGB, (10, 10, 4), (10, 10, 3)),
        (ImageFormat.RGBX, ImageFormat.PRGBA, (10, 10, 4), (10, 10, 4)),
        (ImageFormat.PRGBA, ImageFormat.RGBX, (10, 10, 4), (10, 10, 4)),
        (ImageFormat.GRAY, ImageFormat.PRGBA, (10, 10, 1), (10, 10, 4)),
        (ImageFormat.PRGBA, ImageFormat.GRAY, (10, 10, 4), (10, 10, 1)),
        (ImageFormat.GRAY, ImageFormat.RGBX, (10, 10, 1), (10, 10, 4)),
        (ImageFormat.RGBX, ImageFormat.GRAY, (10, 10, 4), (10, 10, 1)),
        (ImageFormat.GRAY_ALPHA, ImageFormat.PRGBA, (10, 10, 2), (10, 10, 4)),
        (ImageFormat.PRGBA, ImageFormat.GRAY_ALPHA, (10, 10, 4), (10, 10, 2)),
    ],
    ids=[
        "rgb_to_prgba",
        "prgba_to_rgb",
        "rgbx_to_prgba",
        "prgba_to_rgbx",
        "gray_to_prgba",
        "prgba_to_gray",
        "gray_to_rgbx",
        "rgbx_to_gray",
        "gray_alpha_to_prgba",
        "prgba_to_gray_alpha",
    ],
)
def test_all_format_conversion_shapes(
    src_fmt: ImageFormat,
    dst_fmt: ImageFormat,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> None:
    """Valida que a conversão entre todos os pares de formatos produz a dimensão correta de canais."""
    data = np.full(in_shape, 128, dtype=np.uint8)
    converted = convert_image_format(data, src_fmt, dst_fmt)
    assert converted.shape == out_shape


def test_image_to_format_method() -> None:
    """Valida o método público Image.to_format convertendo uma instância para PRGBA e RGBX."""
    raw = np.full((20, 20, 4), 200, dtype=np.uint8)
    img_rgba = Image(raw, ImageFormat.RGBA)

    img_prgba = img_rgba.to_format(ImageFormat.PRGBA)
    assert img_prgba.format == ImageFormat.PRGBA
    assert img_prgba.shape == (20, 20, 4)

    img_rgbx = img_prgba.to_format(ImageFormat.RGBX)
    assert img_rgbx.format == ImageFormat.RGBX
    assert img_rgbx.shape == (20, 20, 4)
