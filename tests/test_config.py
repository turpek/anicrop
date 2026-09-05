from __future__ import annotations

import pytest

from anicrop.buffer import ArrayBuffer, MMapBuffer
from anicrop.config import config
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.io.registry import get_default_backend


@pytest.fixture(autouse=True)
def reset_config():
    """Garante que todas as configurações sejam restauradas antes e depois de cada teste."""
    config.reset()
    yield
    config.reset()


def test_config_default_values():
    """Valida os valores padrão de fábrica do singleton config."""
    assert config.backend == "opencv"
    assert config.memory_threshold == 8192 * 8192


def test_config_set_backend():
    """Valida a alteração direta do backend padrão via config.backend."""
    config.backend = "vips"
    assert config.backend == "vips"
    assert get_default_backend().__class__.__name__ == "PyvipsBackend"


def test_config_set_backend_invalid_raises_error():
    """Valida que atribuir um backend não registrado lança exceção."""
    with pytest.raises((KeyError, ValueError)):
        config.backend = "backend_inexistente"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none_disables_disk"),
        pytest.param(100_000_000, id="custom_threshold"),
    ],
)
def test_config_set_memory_threshold(value):
    """Valida a alteração do limite de memória em RAM via config.memory_threshold."""
    config.memory_threshold = value
    assert config.memory_threshold == value


@pytest.mark.parametrize("invalid_val", [0, -10], ids=["zero", "negative"])
def test_config_set_memory_threshold_invalid_raises_error(invalid_val):
    """Valida que valores não positivos de memory_threshold lançam ValueError."""
    with pytest.raises(ValueError, match="memory_threshold deve ser positivo"):
        config.memory_threshold = invalid_val


def test_config_context_manager_temporary_override():
    """Valida que config(...) altera temporariamente os valores dentro do bloco with."""
    initial_backend = config.backend
    initial_threshold = config.memory_threshold

    with config(backend="vips", memory_threshold=None):
        assert config.backend == "vips"
        assert config.memory_threshold is None

    assert config.backend == initial_backend
    assert config.memory_threshold == initial_threshold


def test_config_context_manager_restores_on_exception():
    """Valida que config(...) restaura os valores originais mesmo se uma exceção ocorrer no bloco with."""
    initial_backend = config.backend

    with pytest.raises(RuntimeError, match="Erro simulado"):
        with config(backend="vips"):
            assert config.backend == "vips"
            raise RuntimeError("Erro simulado")

    assert config.backend == initial_backend


def test_config_context_manager_invalid_key_raises_attribute_error():
    """Valida que passar um atributo inexistente no context manager lança AttributeError."""
    with pytest.raises(AttributeError, match="Opção de configuração inválida"):
        with config(inexistente="foo"):
            pass


def test_config_reset_restores_defaults():
    """Valida que config.reset() redefine todas as opções para os valores padrão."""
    config.backend = "vips"
    config.memory_threshold = None

    config.reset()

    assert config.backend == "opencv"
    assert config.memory_threshold == 8192 * 8192


def test_config_integration_with_image_new():
    """Valida se Image.new consulta o threshold do config dentro de um bloco with."""
    with config(memory_threshold=1_000_000):
        img = Image.new((2000, 2000), ImageFormat.RGBA)
        assert isinstance(img._data, MMapBuffer)

    with config(memory_threshold=None):
        img2 = Image.new((2000, 2000), ImageFormat.RGBA)
        assert isinstance(img2._data, ArrayBuffer)
