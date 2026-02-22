from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer, EditLayer
from anicrop.proxy import ProxyLayer
from anicrop.history import GlobalHistory
from anicrop.command import SetAttributeCommand
from anicrop.spatial import Region
import numpy as np
import pytest
from unittest.mock import Mock


def make_img(w=10, h=10):
    return Image(np.zeros((h, w, 4), dtype=np.uint8), ImageFormat.RGBA)


@pytest.fixture
def history():
    return Mock(spec=GlobalHistory)


@pytest.fixture
def layer():
    return Layer(make_img())


@pytest.fixture
def proxy(layer, history):
    return ProxyLayer(layer, history)


def test_ProxyLayer_delegacao_de_leitura(proxy, layer):
    """Testa se o proxy permite ler atributos do layer original."""
    layer.name = "Original Name"
    assert proxy.name == "Original Name"
    assert proxy.opacity == 1.0


def test_ProxyLayer_interceptacao_de_escrita(proxy, layer, history):
    """Testa se a escrita em atributos passa pelo histórico."""
    proxy.name = "New Name"

    # Verifica se o comando foi enviado para o histórico
    history.push.assert_called_once()
    args, _ = history.push.call_args
    command_class, attr_name, target, value = args

    assert command_class == SetAttributeCommand
    assert attr_name == "name"
    assert target is layer
    assert value == "New Name"

    # O comando SetAttributeCommand normalmente aplica a mudança quando executado pelo histórico.
    # Como estamos mockando o histórico, o valor no layer não deve mudar automaticamente
    # a menos que o mock execute o comando (o que não faz).
    # Se o Proxy aplicasse a mudança E mandasse pro histórico, teríamos duplicidade ou comportamento diferente.
    # O padrão Command geralmente implica que o histórico executa.
    # Vamos assumir por enquanto que o Proxy APENAS manda pro histórico.
    assert layer.name != "New Name"


def test_ProxyLayer_delegacao_de_metodos(proxy, layer):
    """Testa se métodos chamados no proxy são executados no layer."""
    region = Region.from_size(5, 5)
    img = make_img(5, 5)

    # Chama add_edit através do proxy
    proxy.add_edit(img, region)

    # Verifica se o efeito ocorreu no layer
    assert len(layer._edits) == 2
    assert isinstance(layer._edits[1], EditLayer)
    assert layer._edits[1].region == region


def test_ProxyLayer_acesso_a_atributos_inexistentes(proxy):
    with pytest.raises(AttributeError):
        _ = proxy.atributo_que_nao_existe


def test_ProxyLayer_dir_contem_atributos_do_layer(proxy):
    attrs = dir(proxy)
    assert "name" in attrs
    assert "opacity" in attrs
    assert "add_edit" in attrs
