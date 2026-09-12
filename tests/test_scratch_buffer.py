from __future__ import annotations

import numpy as np

from anicrop.config import config
from anicrop.enums import ImageFormat
from anicrop.scratch import ScratchBuffer
from anicrop.spatial import Region


def test_scratch_buffer_lazy_initialization():
    """Valida que a configuracao do buffer nao aloca memoria imediatamente."""
    buf = ScratchBuffer()
    buf.configure((200, 100), ImageFormat.RGBA)

    assert buf._image is None


def test_scratch_buffer_getitem_allocates_and_slices():
    """Valida que o acesso por slice aloca sob demanda e retorna array NumPy com dimensoes corretas."""
    buf = ScratchBuffer()
    buf.configure((200, 100), ImageFormat.RGBA)

    slice_arr = buf[Region.from_rect(10, 20, 50, 40)]

    assert isinstance(slice_arr, np.ndarray)
    assert slice_arr.shape == (40, 50, 4)
    assert buf._image is not None
    assert buf._image.width >= 200
    assert buf._image.height >= 100


def test_scratch_buffer_reuses_existing_allocation():
    """Valida que requisicoes menores reutilizam o mesmo buffer de memoria sem realocacao."""
    buf = ScratchBuffer()
    buf.configure((200, 100), ImageFormat.RGBA)
    _ = buf[Region.from_rect(0, 0, 50, 50)]
    original_ptr = buf._image._data.__array_interface__["data"][0]

    buf.configure((100, 50), ImageFormat.RGBA)
    _ = buf[Region.from_rect(0, 0, 30, 30)]
    reused_ptr = buf._image._data.__array_interface__["data"][0]

    assert original_ptr == reused_ptr


def test_scratch_buffer_grows_with_growth_factor():
    """Valida que requisicoes maiores expandem as dimensoes do buffer com fator multiplicador."""
    buf = ScratchBuffer()
    buf.configure((100, 100), ImageFormat.RGBA)
    _ = buf[Region.from_rect(0, 0, 50, 50)]
    initial_w = buf._image.width
    initial_h = buf._image.height

    buf.configure((300, 300), ImageFormat.RGBA)
    _ = buf[Region.from_rect(0, 0, 50, 50)]

    assert buf._image.width >= int(initial_w * 1.5)
    assert buf._image.height >= int(initial_h * 1.5)


def test_scratch_buffer_reallocates_on_format_change():
    """Valida que alterar o formato de cor forca a recriacao do buffer com o novo numero de canais."""
    buf = ScratchBuffer()
    buf.configure((100, 100), ImageFormat.RGBA)
    _ = buf[Region.from_rect(0, 0, 50, 50)]

    buf.configure((100, 100), ImageFormat.RGB)
    slice_rgb = buf[Region.from_rect(0, 0, 50, 50)]

    assert buf._image.format == ImageFormat.RGB
    assert slice_rgb.shape == (50, 50, 3)


def test_scratch_buffer_was_used_flag_lifecycle():
    """Valida se a flag was_used inicia como False, vira True apos __getitem__ e reseta no configure."""
    buf = ScratchBuffer()
    assert buf.was_used is False

    buf.configure((100, 100), ImageFormat.RGBA)
    assert buf.was_used is False

    _ = buf[Region.from_size(50, 50)]
    assert buf.was_used is True

    buf.configure((100, 100), ImageFormat.RGBA)
    assert buf.was_used is False


def test_scratch_buffer_growth_frees_previous_mmap_buffer():
    """Valida se a expansao do ScratchBuffer desaloca imediatamente o arquivo temporario anterior no disco."""
    with config(memory_threshold=1000):
        buf = ScratchBuffer()
        buf.configure((40, 40), ImageFormat.RGBA)
        _ = buf[Region.from_size(40, 40)]
        path_first = buf._image._data.file_path

        assert path_first is not None
        assert path_first.exists()

        buf.configure((80, 80), ImageFormat.RGBA)
        _ = buf[Region.from_size(80, 80)]
        path_second = buf._image._data.file_path

        assert not path_first.exists()
        assert path_second is not None
        assert path_second.exists()

        buf.close()
        assert not path_second.exists()
