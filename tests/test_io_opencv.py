from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anicrop.enums import ImageFormat
from anicrop.interfaces.io import SaveOptions
from anicrop.io import (
    OpenCVBackend,
    get_backend,
)
from anicrop.io.opencv import (
    _convert_to_requested_format,
    _cv2_imread_windows,
    _cv2_imwrite_windows,
)
from anicrop.spatial import Region


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_registry_backend_lookup():
    """Valida se o registro de backends resolve corretamente o OpenCVBackend."""
    backend = get_backend("opencv")
    assert isinstance(backend, OpenCVBackend)


def test_registry_unknown_backend_raises_error():
    """Valida se buscar um backend desconhecido lanca KeyError."""
    with pytest.raises(KeyError, match="não encontrado"):
        get_backend("non_existent_backend")


def test_opencv_get_size(temp_dir: Path):
    """Valida se get_size extrai as dimensoes corretas lendo o cabecalho."""
    img_path = temp_dir / "test_size.png"
    data = np.zeros((150, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), data)

    backend = OpenCVBackend()
    size = backend.get_size(img_path)

    assert size == (200, 150)


@pytest.mark.parametrize(
    "shape, is_color, expected_fmt",
    [
        pytest.param((40, 50, 4), True, ImageFormat.RGBA, id="auto_detect_rgba"),
        pytest.param((40, 50, 3), True, ImageFormat.RGB, id="auto_detect_rgb"),
        pytest.param((40, 50), False, ImageFormat.GRAY, id="auto_detect_gray"),
    ],
)
def test_opencv_read_auto_detect(temp_dir: Path, shape, is_color, expected_fmt):
    """Valida auto-deteccao de formatos na leitura com format=None."""
    img_path = temp_dir / f"test_{expected_fmt.name}.png"
    data = np.full(shape, 128, dtype=np.uint8)
    cv2.imwrite(str(img_path), data)

    backend = OpenCVBackend()
    loaded_data, resolved_fmt, orig_size = backend.read(img_path, format=None)

    assert resolved_fmt == expected_fmt
    assert orig_size == (50, 40)
    assert loaded_data.shape[:2] == (40, 50)


def test_opencv_read_explicit_conversion_rgb_to_rgba(temp_dir: Path):
    """Valida conversao explicita de RGB para RGBA adicionando canal alfa opaco."""
    img_path = temp_dir / "test_rgb.png"
    rgb_data = np.full((30, 40, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(img_path), rgb_data)

    backend = OpenCVBackend()
    loaded_data, resolved_fmt, _ = backend.read(img_path, format=ImageFormat.RGBA)

    assert resolved_fmt == ImageFormat.RGBA
    assert loaded_data.shape == (30, 40, 4)
    assert np.all(loaded_data[:, :, 3] == 255)


def test_opencv_read_with_shrink(temp_dir: Path):
    """Valida se o parametro shrink reduz proporcionalmente as dimensoes da imagem lida."""
    img_path = temp_dir / "test_shrink.png"
    data = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), data)

    backend = OpenCVBackend()
    loaded_data, _, orig_size = backend.read(img_path, format=ImageFormat.RGB, shrink=2)

    assert orig_size == (200, 100)
    assert loaded_data.shape == (50, 100, 3)


def test_opencv_read_with_roi(temp_dir: Path):
    """Valida se o parametro roi recorta a regiao espacial especificada."""
    img_path = temp_dir / "test_roi.png"
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    data[10:30, 20:50] = (255, 0, 0)

    backend = OpenCVBackend()
    backend.write(img_path, data, format=ImageFormat.RGB)

    roi = Region.from_rect(20, 10, 30, 20)
    loaded_data, _, orig_size = backend.read(img_path, format=ImageFormat.RGB, roi=roi)

    assert orig_size == (100, 100)
    assert loaded_data.shape == (20, 30, 3)
    assert np.all(loaded_data == (255, 0, 0))


def test_opencv_write_jpeg_flattens_alpha(temp_dir: Path):
    """Valida se salvar imagem RGBA transparente em JPEG mescla o alfa na cor de fundo."""
    img_path = temp_dir / "test_flatten.jpg"
    rgba_data = np.zeros((20, 20, 4), dtype=np.uint8)
    rgba_data[:, :] = (0, 255, 0, 0)  # Totalmente transparente

    backend = OpenCVBackend()
    options = SaveOptions(bg_color=(255, 0, 0))  # Fundo vermelho
    backend.write(img_path, rgba_data, format=ImageFormat.RGBA, options=options)

    # Reabre e verifica que o fundo transparente virou vermelho
    reloaded, _, _ = backend.read(img_path, format=ImageFormat.RGB)
    assert np.allclose(reloaded[0, 0], [255, 0, 0], atol=10)


def test_opencv_read_non_existent_file_raises_error(temp_dir: Path):
    """Valida se tentar ler arquivo inexistente lanca FileNotFoundError."""
    backend = OpenCVBackend()
    with pytest.raises(FileNotFoundError):
        backend.read(temp_dir / "nao_existe.png")


def test_opencv_write_and_read_bgr(temp_dir: Path):
    """Valida gravação e leitura de imagens no formato BGR com OpenCV."""
    img_path = temp_dir / "test_bgr.png"
    data = np.zeros((30, 40, 3), dtype=np.uint8)
    data[:, :] = (200, 100, 50)

    backend = OpenCVBackend()
    backend.write(img_path, data, format=ImageFormat.BGR)

    loaded, fmt, _ = backend.read(img_path, format=ImageFormat.BGR)
    assert fmt == ImageFormat.BGR
    assert loaded.shape == (30, 40, 3)
    assert np.all(loaded[0, 0] == (200, 100, 50))


def test_opencv_write_and_read_bgra(temp_dir: Path):
    """Valida gravação e leitura de imagens no formato BGRA com OpenCV."""
    img_path = temp_dir / "test_bgra.png"
    data = np.zeros((30, 40, 4), dtype=np.uint8)
    data[:, :] = (200, 100, 50, 180)

    backend = OpenCVBackend()
    backend.write(img_path, data, format=ImageFormat.BGRA)

    loaded, fmt, _ = backend.read(img_path, format=ImageFormat.BGRA)
    assert fmt == ImageFormat.BGRA
    assert loaded.shape == (30, 40, 4)
    assert np.all(loaded[0, 0] == (200, 100, 50, 180))


def test_opencv_write_and_read_unicode_path(temp_dir: Path):
    """Valida gravação e leitura de imagens com caracteres especiais e acentos no caminho."""
    img_path = temp_dir / "álbum_férias_teste_📸.png"
    data = np.zeros((30, 40, 3), dtype=np.uint8)
    data[:, :] = (120, 150, 180)

    backend = OpenCVBackend()
    backend.write(img_path, data, format=ImageFormat.RGB)

    loaded, fmt, size = backend.read(img_path, format=ImageFormat.RGB)
    assert fmt == ImageFormat.RGB
    assert size == (40, 30)
    assert np.all(loaded[0, 0] == (120, 150, 180))


def test_opencv_windows_unicode_helpers(temp_dir: Path):
    """Valida as funções especialistas do Windows para leitura e gravação com caminhos Unicode."""
    img_path = temp_dir / "imagem_caminho_acentuado_ç_ã.png"
    data = np.full((20, 20, 3), 210, dtype=np.uint8)

    success = _cv2_imwrite_windows(img_path, data)
    assert success is True

    loaded = _cv2_imread_windows(img_path)
    assert loaded is not None
    assert loaded.shape == (20, 20, 3)
    assert np.all(loaded == 210)


def test_opencv_write_and_read_16bit_png(temp_dir: Path):
    """Valida gravação e leitura de imagens uint16 garantindo canal alfa opaco em 65535."""
    img_path = temp_dir / "test_16bit.png"
    data = np.full((30, 40, 3), 50000, dtype=np.uint16)

    backend = OpenCVBackend()
    backend.write(img_path, data, format=ImageFormat.RGB)

    loaded, fmt, size = backend.read(img_path, format=ImageFormat.RGBA)
    assert fmt == ImageFormat.RGBA
    assert size == (40, 30)
    assert loaded.dtype == np.uint16
    assert loaded.shape == (30, 40, 4)
    assert np.all(loaded[:, :, 3] == 65535)
    assert np.all(loaded[:, :, :3] == 50000)


def test_opencv_read_float_conversion_alpha():
    """Valida conversão de imagem float32 para RGBA garantindo canal alfa opaco em 1.0."""
    raw_bgr = np.full((20, 20, 3), 0.75, dtype=np.float32)

    converted = _convert_to_requested_format(raw_bgr, 3, ImageFormat.RGBA)
    assert converted.dtype == np.float32
    assert converted.shape == (20, 20, 4)
    assert np.all(converted[:, :, 3] == 1.0)
    assert np.all(converted[:, :, :3] == 0.75)
