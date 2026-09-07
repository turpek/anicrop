from __future__ import annotations

import cv2
import numpy as np
import pytest

from anicrop.blend import solid_fill
from anicrop.canvas import Canvas
from anicrop.config import config
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.render import CanvasRender


@pytest.fixture(autouse=True)
def reset_config():
    """Garante a restauração do config antes e depois de cada teste."""
    config.reset()
    yield
    config.reset()


def test_config_dtype_default_is_uint8():
    """Valida que o dtype padrão do config é uint8."""
    assert config.dtype == np.uint8


@pytest.mark.parametrize(
    "input_val,expected",
    [
        (np.uint8, np.uint8),
        ("uint8", np.uint8),
        (np.uint16, np.uint16),
        ("uint16", np.uint16),
        (np.float32, np.float32),
        ("float32", np.float32),
    ],
    ids=["u8_type", "u8_str", "u16_type", "u16_str", "f32_type", "f32_str"],
)
def test_config_dtype_setter_valid(input_val, expected):
    """Valida a atribuição de dtypes suportados no config."""
    config.dtype = input_val
    assert config.dtype == expected


def test_config_dtype_setter_invalid_raises():
    """Valida que dtypes não suportados levantam ValueError."""
    with pytest.raises(ValueError, match="dtype não suportado"):
        config.dtype = np.int64


def test_config_dtype_scoped_context():
    """Valida o override temporário do dtype via context manager."""
    with config(dtype=np.uint16):
        assert config.dtype == np.uint16
    assert config.dtype == np.uint8


def test_image_to_dtype_u8_to_u16():
    """Valida a conversão e escala correta de uint8 para uint16."""
    img_u8 = Image(np.array([[[0, 128, 255, 255]]], dtype=np.uint8), ImageFormat.RGBA)
    img_u16 = img_u8.to_dtype(np.uint16)

    assert img_u16.dtype == np.uint16
    assert img_u16[0, 0, 0] == 0
    assert img_u16[0, 0, 3] == 65535


def test_image_to_dtype_u16_to_u8():
    """Valida a conversão e escala correta de uint16 para uint8."""
    img_u16 = Image(np.array([[[0, 32768, 65535, 65535]]], dtype=np.uint16), ImageFormat.RGBA)
    img_u8 = img_u16.to_dtype(np.uint8)

    assert img_u8.dtype == np.uint8
    assert img_u8[0, 0, 0] == 0
    assert img_u8[0, 0, 3] == 255


def test_image_to_uint8_shortcut():
    """Valida o método de conveniência to_uint8 em imagem uint16."""
    img_u16 = Image(np.full((4, 4, 3), 65535, dtype=np.uint16), ImageFormat.RGB)
    img_u8 = img_u16.to_uint8()

    assert img_u8.dtype == np.uint8
    assert np.all(img_u8[...] == 255)


def test_canvas_inherits_config_dtype_by_default():
    """Valida que o Canvas herda o dtype do config na sua instanciação."""
    canvas_default = Canvas.from_size(100, 100)
    assert canvas_default.dtype == np.uint8

    with config(dtype=np.uint16):
        canvas_u16 = Canvas.from_size(100, 100)
        assert canvas_u16.dtype == np.uint16


def test_canvas_explicit_dtype_overrides_config():
    """Valida que a passagem explícita de dtype no Canvas tem precedência sobre o config."""
    canvas = Canvas.from_size(100, 100, dtype=np.uint16)
    assert canvas.dtype == np.uint16


def test_solid_fill_u16():
    """Valida o modo solid_fill operando nativamente em uint16."""
    base_data = np.zeros((10, 10, 4), dtype=np.uint16)
    base_data[:, :5] = [0, 0, 65535, 65535]  # Esquerda: Azul sólido 16-bit
    base = Image(base_data, ImageFormat.RGBA)

    over_data = np.full((10, 10, 4), [65535, 0, 0, 65535], dtype=np.uint16)  # Vermelho sólido
    overlay = Image(over_data, ImageFormat.RGBA)

    solid_fill(base, overlay)

    np.testing.assert_array_equal(base[5, 2], [0, 0, 65535, 65535])
    np.testing.assert_array_equal(base[5, 8], [65535, 0, 0, 65535])


def test_render_scene_u16_canvas_produces_u16_image():
    """Valida que renderizar uma cena sobre um Canvas uint16 produz uma imagem uint16."""
    canvas = Canvas.from_size(20, 20, dtype=np.uint16)
    img_data = np.full((10, 10, 4), 65535, dtype=np.uint16)
    layer = Layer(Image(img_data, ImageFormat.RGBA))

    rendered = CanvasRender().render_scene([layer], canvas)

    assert rendered.dtype == np.uint16
    assert np.all(rendered[:10, :10] == 65535)


def test_render_scene_harmonizes_u8_layer_into_u16_canvas():
    """Valida que uma camada uint8 é harmonizada sem erros ao ser desenhada em Canvas uint16."""
    canvas = Canvas.from_size(20, 20, dtype=np.uint16)
    img_u8 = np.full((10, 10, 4), 255, dtype=np.uint8)
    layer = Layer(Image(img_u8, ImageFormat.RGBA))

    rendered = CanvasRender().render_scene([layer], canvas)

    assert rendered.dtype == np.uint16
    assert np.all(rendered[:10, :10] == 65535)


def test_image_open_16bit_converts_to_default_u8(tmp_path):
    """Valida que Image.open converte PNG 16-bit para uint8 por padrão."""
    img_u16 = np.full((10, 10, 3), 65535, dtype=np.uint16)
    file_path = tmp_path / "test16.png"
    cv2.imwrite(str(file_path), img_u16)

    img = Image.open(file_path)

    assert img.dtype == np.uint8
    assert np.all(img[...] == 255)


def test_image_open_16bit_preserves_u16_when_configured(tmp_path):
    """Valida que Image.open preserva uint16 quando config.dtype == np.uint16."""
    img_u16 = np.full((10, 10, 3), 65535, dtype=np.uint16)
    file_path = tmp_path / "test16.png"
    cv2.imwrite(str(file_path), img_u16)

    with config(dtype=np.uint16):
        img = Image.open(file_path)

    assert img.dtype == np.uint16
    assert np.all(img[...] == 65535)


def test_image_open_dtype_none_preserves_native_dtype(tmp_path):
    """Valida que Image.open com dtype=None preserva o dtype nativo do arquivo."""
    img_u16 = np.full((10, 10, 3), 65535, dtype=np.uint16)
    file_path = tmp_path / "test16.png"
    cv2.imwrite(str(file_path), img_u16)

    img = Image.open(file_path, dtype=None)

    assert img.dtype == np.uint16
    assert np.all(img[...] == 65535)
