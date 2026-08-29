from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import pytest

from anicrop.enums import ImageFormat
from anicrop.interfaces.io import SaveOptions
from anicrop.io import OpenCVBackend, get_backend, get_default_backend, register_backend, set_default_backend
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
