import gc
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from anicrop.buffer import MMapBuffer
from anicrop.image import Image, ImageFormat


def test_mmap_buffer_creation_from_array(tmp_path: Path):
    """Valida a criacao de MMapBuffer a partir de array NumPy com persistencia em disco."""
    arr = np.zeros((100, 200, 4), dtype=np.uint8)
    arr[10:20, 30:40] = 255
    target_path = tmp_path / "test.raw"

    buf = MMapBuffer.from_array(arr, file_path=target_path)

    assert buf.shape == (100, 200, 4)
    assert buf.width == 200
    assert buf.height == 100
    assert buf.channels == 4
    assert buf.ndim == 3
    assert np.array_equal(buf[10:20, 30:40], arr[10:20, 30:40])


def test_mmap_buffer_create_empty(tmp_path: Path):
    """Valida a alocacao de um MMapBuffer vazio e mutacao direta via slice."""
    target_path = tmp_path / "empty.raw"
    buf = MMapBuffer.create_empty((50, 60, 3), dtype=np.uint8, file_path=target_path)
    buf[10:20, 10:20] = 128
    buf.flush()

    assert buf.shape == (50, 60, 3)
    assert np.all(buf[10:20, 10:20] == 128)
    assert np.all(buf[0:5, 0:5] == 0)


def test_mmap_buffer_numpy_array_protocol(tmp_path: Path):
    """Valida a compatibilidade de MMapBuffer com o protocolo NumPy array e igualdade."""
    arr = np.ones((40, 40, 4), dtype=np.uint8) * 200
    buf = MMapBuffer.from_array(arr, file_path=tmp_path / "arr.raw")

    converted = np.asarray(buf)

    assert isinstance(converted, np.ndarray)
    assert np.array_equal(converted, arr)
    assert np.all(buf == arr)


def test_mmap_buffer_image_integration(tmp_path: Path):
    """Valida o encapsulamento de MMapBuffer dentro de Image e renderizacao."""
    arr = np.full((100, 100, 4), 255, dtype=np.uint8)
    buf = MMapBuffer.from_array(arr, file_path=tmp_path / "img.raw")
    img = Image(buf, ImageFormat.RGBA)

    assert img.width == 100
    assert img.height == 100
    assert img.format == ImageFormat.RGBA
    assert np.array_equal(img[...], arr)


@pytest.mark.parametrize("level, expected_factor", [(0, 1.0), (1, 0.5), (2, 0.25)])
def test_mmap_buffer_get_lod(tmp_path: Path, level: int, expected_factor: float):
    """Valida a geracao sob demanda de niveis de resolucao reduzidos (LOD)."""
    arr = np.zeros((100, 200, 4), dtype=np.uint8)
    buf = MMapBuffer.from_array(arr, file_path=tmp_path / "lod.raw")

    lod_buf = buf.get_lod(level)

    expected_w = int(200 * expected_factor)
    expected_h = int(100 * expected_factor)
    assert lod_buf.width == expected_w
    assert lod_buf.height == expected_h


def test_mmap_buffer_auto_remove_on_gc():
    """Valida a remocao automatica do arquivo temporario no disco ao ser coletado pelo Garbage Collector."""
    buf = MMapBuffer.create_empty((100, 100, 4), dtype=np.uint8)
    file_path = buf.file_path

    assert file_path is not None
    assert file_path.exists()
    del buf
    gc.collect()

    assert not file_path.exists()


def test_mmap_buffer_auto_remove_on_explicit_close():
    """Valida a remocao imediata do arquivo temporario no disco ao invocar close explicitamente."""
    buf = MMapBuffer.create_empty((100, 100, 4), dtype=np.uint8)
    file_path = buf.file_path

    assert file_path is not None
    assert file_path.exists()
    buf.close()

    assert not file_path.exists()


def test_mmap_buffer_preserve_custom_file_path(tmp_path: Path):
    """Valida que caminhos de arquivo customizados fornecidos pelo usuario nao sao removidos automaticamente."""
    target_path = tmp_path / "custom.raw"
    arr = np.ones((50, 50, 3), dtype=np.uint8)
    buf = MMapBuffer.from_array(arr, file_path=target_path)

    assert buf.file_path == target_path
    assert target_path.exists()
    del buf
    gc.collect()

    assert target_path.exists()


def test_mmap_buffer_open_existing_never_removes(tmp_path: Path):
    """Valida que arquivos abertos via open_existing sao sempre preservados apos fechamento e GC."""
    target_path = tmp_path / "existing.raw"
    arr = np.zeros((40, 40, 4), dtype=np.uint8)
    initial_buf = MMapBuffer.from_array(arr, file_path=target_path)
    initial_buf.flush()

    opened_buf = MMapBuffer.open_existing(target_path, (40, 40, 4), dtype=np.uint8)
    opened_buf.close()
    del opened_buf
    gc.collect()

    assert target_path.exists()


def test_mmap_buffer_create_empty_raises_when_insufficient_space(tmp_path: Path):
    """Valida que create_empty lanca OSError preventivo e nao cria arquivo quando falta espaco."""
    target_path = tmp_path / "too_big.raw"
    mock_usage = MagicMock(free=10 * 1024 * 1024)

    with patch("shutil.disk_usage", return_value=mock_usage):
        with pytest.raises(OSError, match="Espaço insuficiente em disco"):
            MMapBuffer.create_empty((10000, 10000, 4), dtype=np.uint8, file_path=target_path)

    assert not target_path.exists()


def test_mmap_buffer_from_array_raises_when_insufficient_space(tmp_path: Path):
    """Valida que from_array lanca OSError preventivo e nao grava quando o espaco e insuficiente."""
    target_path = tmp_path / "too_big_arr.raw"
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    mock_usage = MagicMock(free=1024)

    with patch("shutil.disk_usage", return_value=mock_usage):
        with pytest.raises(OSError, match="Espaço insuficiente em disco"):
            MMapBuffer.from_array(arr, file_path=target_path)

    assert not target_path.exists()


def test_mmap_buffer_create_empty_cleans_up_on_np_memmap_failure():
    """Valida que arquivos residuais sao removidos imediatamente se np.memmap falhar durante criacao."""
    created_paths = []

    def fail_memmap(*args, **kwargs):
        path = Path(args[0])
        created_paths.append(path)
        path.write_bytes(b"corrupted_partial")
        raise OSError(122, "Disk quota exceeded")

    with patch("numpy.memmap", side_effect=fail_memmap):
        with pytest.raises(OSError, match="Disk quota exceeded"):
            MMapBuffer.create_empty((50, 50, 4), dtype=np.uint8)

    assert len(created_paths) == 1
    assert not created_paths[0].exists()


def test_mmap_buffer_from_array_cleans_up_on_np_memmap_failure():
    """Valida que arquivos temporarios de from_array sao expurgados se np.memmap falhar."""
    created_paths = []

    def fail_memmap(*args, **kwargs):
        path = Path(args[0])
        created_paths.append(path)
        path.write_bytes(b"corrupted_partial")
        raise OSError(122, "Disk quota exceeded")

    arr = np.zeros((20, 20, 4), dtype=np.uint8)
    with patch("numpy.memmap", side_effect=fail_memmap):
        with pytest.raises(OSError, match="Disk quota exceeded"):
            MMapBuffer.from_array(arr)

    assert len(created_paths) == 1
    assert not created_paths[0].exists()
