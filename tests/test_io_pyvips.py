from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from anicrop.enums import ImageFormat
from anicrop.interfaces.io import SaveOptions
from anicrop.io import PyvipsBackend, get_backend, get_default_backend, is_vips_available
from anicrop.spatial import Region

pytestmark = pytest.mark.skipif(not is_vips_available(), reason="pyvips não instalado")


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_vips_default_backend_when_available():
    """Valida se PyvipsBackend e definido como backend padrao quando pyvips esta disponivel."""
    backend = get_default_backend()
    assert isinstance(backend, PyvipsBackend)
    assert get_backend("vips") is backend


def test_vips_get_size(temp_dir: Path):
    """Valida se get_size extrai as dimensoes corretas lendo o cabecalho com pyvips."""
    img_path = temp_dir / "test_vips_size.png"
    backend = PyvipsBackend()
    data = np.zeros((150, 200, 3), dtype=np.uint8)
    backend.write(img_path, data, format=ImageFormat.RGB)

    size = backend.get_size(img_path)
    assert size == (200, 150)


@pytest.mark.parametrize(
    "shape, expected_fmt",
    [
        pytest.param((40, 50, 4), ImageFormat.RGBA, id="vips_auto_rgba"),
        pytest.param((40, 50, 3), ImageFormat.RGB, id="vips_auto_rgb"),
        pytest.param((40, 50, 1), ImageFormat.GRAY, id="vips_auto_gray"),
    ],
)
def test_vips_read_auto_detect(temp_dir: Path, shape, expected_fmt):
    """Valida auto-deteccao de formatos na leitura com format=None usando pyvips."""
    img_path = temp_dir / f"test_vips_{expected_fmt.name}.png"
    data = np.full(shape, 128, dtype=np.uint8)

    backend = PyvipsBackend()
    backend.write(img_path, data, format=expected_fmt)

    loaded_data, resolved_fmt, orig_size = backend.read(img_path, format=None)

    assert resolved_fmt == expected_fmt
    assert orig_size == (50, 40)
    assert loaded_data.shape[:2] == (40, 50)


def test_vips_read_explicit_conversion_rgb_to_rgba(temp_dir: Path):
    """Valida conversao explicita de RGB para RGBA adicionando canal alfa opaco com pyvips."""
    img_path = temp_dir / "test_vips_rgb.png"
    rgb_data = np.full((30, 40, 3), 200, dtype=np.uint8)

    backend = PyvipsBackend()
    backend.write(img_path, rgb_data, format=ImageFormat.RGB)

    loaded_data, resolved_fmt, _ = backend.read(img_path, format=ImageFormat.RGBA)

    assert resolved_fmt == ImageFormat.RGBA
    assert loaded_data.shape == (30, 40, 4)
    assert np.all(loaded_data[:, :, 3] == 255)


def test_vips_read_with_shrink(temp_dir: Path):
    """Valida se o parametro shrink reduz proporcionalmente as dimensoes no pyvips."""
    img_path = temp_dir / "test_vips_shrink.png"
    data = np.zeros((100, 200, 3), dtype=np.uint8)

    backend = PyvipsBackend()
    backend.write(img_path, data, format=ImageFormat.RGB)

    loaded_data, _, orig_size = backend.read(img_path, format=ImageFormat.RGB, shrink=2)

    assert orig_size == (200, 100)
    assert loaded_data.shape == (50, 100, 3)


def test_vips_read_with_roi(temp_dir: Path):
    """Valida se o parametro roi recorta a regiao espacial especificada no pyvips."""
    img_path = temp_dir / "test_vips_roi.png"
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    data[10:30, 20:50] = (255, 0, 0)

    backend = PyvipsBackend()
    backend.write(img_path, data, format=ImageFormat.RGB)

    roi = Region.from_rect(20, 10, 30, 20)
    loaded_data, _, orig_size = backend.read(img_path, format=ImageFormat.RGB, roi=roi)

    assert orig_size == (100, 100)
    assert loaded_data.shape == (20, 30, 3)
    assert np.all(loaded_data == (255, 0, 0))


def test_vips_write_jpeg_flattens_alpha(temp_dir: Path):
    """Valida se salvar imagem RGBA transparente em JPEG mescla o alfa na cor de fundo com pyvips."""
    img_path = temp_dir / "test_vips_flatten.jpg"
    rgba_data = np.zeros((20, 20, 4), dtype=np.uint8)
    rgba_data[:, :] = (0, 255, 0, 0)  # Totalmente transparente

    backend = PyvipsBackend()
    options = SaveOptions(bg_color=(255, 0, 0))  # Fundo vermelho
    backend.write(img_path, rgba_data, format=ImageFormat.RGBA, options=options)

    # Reabre e verifica que o fundo transparente virou vermelho
    reloaded, _, _ = backend.read(img_path, format=ImageFormat.RGB)
    assert np.allclose(reloaded[0, 0], [255, 0, 0], atol=10)


def test_vips_read_non_existent_file_raises_error(temp_dir: Path):
    """Valida se tentar ler arquivo inexistente no pyvips lanca FileNotFoundError."""
    backend = PyvipsBackend()
    with pytest.raises(FileNotFoundError):
        backend.read(temp_dir / "nao_existe.png")
