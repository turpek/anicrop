import numpy as np
import pytest

from anicrop.command import BaseLayerCommand
from anicrop.container import GroupLayer
from anicrop.history import GlobalHistory
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.proxy import GroupProxy, ProxyLayer, is_property_with_setter


def make_img(w=10, h=10):
    return Image(np.zeros((h, w, 4), dtype=np.uint8), ImageFormat.RGBA)


@pytest.fixture
def history(mocker):
    return mocker.MagicMock(spec=GlobalHistory)


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
    """Testa se a escrita em atributos passa pelo histórico usando LayerImageCommand."""
    proxy.name = "New Name"

    # Verifica se o comando foi enviado para o histórico
    assert history.start_action.call_count == 1
    args, _ = history.start_action.call_args
    command_class, attr_name, target, value = args

    assert command_class == BaseLayerCommand
    assert attr_name == "name"
    assert target is proxy


def test_ProxyLayer_integration_with_real_history(layer):
    """Garante o funcionamento do Undo/Redo real com o ProxyLayer usando LayerImageCommand."""
    real_history = GlobalHistory()
    proxy = ProxyLayer(layer, real_history)

    # 1. Testando propriedade atômica
    proxy.name = "Modified Name"
    assert layer.name == "Modified Name"

    real_history.undo()
    assert layer.name == "Layer"

    real_history.redo()
    assert layer.name == "Modified Name"


def test_ProxyLayer_encadeamento_retorna_o_proprio_composer(proxy, layer):
    """Garante que chamadas encadeadas na transformação devolvem o Composer do Layer."""
    ret = proxy.transform.translate(10, 20)

    # O método devolve o Composer para permitir o encadeamento das matrizes
    assert ret is layer.transform

    # A transformação real deve ter sido aplicada
    assert layer.transform.matrix[0, 2] == 10
    assert layer.transform.matrix[1, 2] == 20


def test_ProxyLayer_dir_contem_atributos_do_layer(proxy):
    attrs = dir(proxy)
    assert "name" in attrs
    assert "opacity" in attrs
    assert "transform" in attrs


def test_ProxyLayer_atribuicao_composta_chama_push_uma_unica_vez(proxy, history):
    """Garante que a atribuição composta (+=) chama o push do histórico exatamente uma vez."""
    proxy.opacity = 0.5
    history.reset_mock()

    proxy.opacity += 0.2
    assert history.start_action.call_count == 1
    assert proxy.opacity == 0.7


def test_is_property_with_setter():
    class Dummy:
        @property
        def prop_ok(self):
            return 1

        @prop_ok.setter
        def prop_ok(self, v):
            pass

        @property
        def prop_readonly(self):
            return 1

        normal_attr = 10

    assert is_property_with_setter(Dummy, "prop_ok") is True
    assert is_property_with_setter(Dummy, "prop_readonly") is False
    assert is_property_with_setter(Dummy, "normal_attr") is False
    assert is_property_with_setter(Dummy, "inexistent") is False


def test_ProxyLayer_passa_atributos_dinamicos_livremente(proxy, layer):
    """Atributos não mapeados no _ACTION_ROUTER devem ser lidos e escritos no target sem erro."""
    proxy.foo = "bar"
    assert layer.foo == "bar"
    assert proxy.foo == "bar"


def test_ProxyLayer_transparencia_de_tipo(proxy, layer):
    """Garante que o Proxy se comporte como o próprio Layer em isinstance e tipo."""
    assert isinstance(proxy, Layer)
    assert isinstance(proxy, ProxyLayer)


def test_ProxyLayer_bypass_de_historico_desativado(proxy, layer, history):
    """Garante que acões encadeadas não registram se o histórico estiver desativado."""
    history.is_active = False

    ret = proxy.transform.translate(50, 50)

    # Devolve o Composer
    assert ret is layer.transform

    # Ação gravou no objeto real
    assert layer.transform.matrix[0, 2] == 50

    # Histórico não gravou nada
    history.start_action.assert_not_called()


def test_ProxyLayer_short_circuit_quando_history_desativado(proxy, layer, history):
    """Garante que se o histórico estiver desativado, o proxy não realiza lookup e repassa direto."""
    # Simulamos o histórico desativado
    history.is_active = False

    # Essa ação está no _ACTION_ROUTER
    proxy.name = "Teste Bypass"

    # O histórico não deve tentar gravar
    history.start_action.assert_not_called()

    # A variável altera normalmente
    assert layer.name == "Teste Bypass"


def test_proxy_parent_property_returns_parent_proxy():
    """Garante que o acesso à propriedade .parent de um filho retorna a instância do Proxy do pai."""
    hist = GlobalHistory()
    parent_group = GroupProxy(GroupLayer(name="Parent"), hist)
    child_group = GroupProxy(GroupLayer(name="Child"), hist)

    parent_group.append(child_group)

    # .parent do filho DEVE retornar a instância do Proxy do pai (parent_group)
    assert child_group.parent is parent_group


def test_prevenir_ciclo_ao_adicionar_pai_como_filho_no_proxy():
    """Garante que adicionar um container pai como filho através do Proxy lança ValueError."""
    hist = GlobalHistory()
    g1 = GroupProxy(GroupLayer(name="g1"), hist)
    g2 = GroupProxy(GroupLayer(name="g2"), hist)

    g1.append(g2)
    assert g2.parent is g1

    with pytest.raises(ValueError, match="Cannot add an ancestor container to a child container"):
        g2.append(g1)


def test_proxy_registry_single_instance_and_clean_target():
    """Garante que a mesma camada devolve a mesma instância de Proxy e o target não é poluído."""
    hist = GlobalHistory()
    img = Image(np.zeros((10, 10, 4), dtype=np.uint8), ImageFormat.RGBA)
    raw_layer = Layer(img, name="RawL")

    p1 = ProxyLayer(raw_layer, hist)
    p2 = ProxyLayer(raw_layer, hist)

    # Garante instância única de Proxy por target (Flyweight Identity Map)
    assert p1 is p2

    # Garante que o objeto real NÃO tem atributo _proxy pendurado nele (domínio limpo)
    assert not hasattr(raw_layer, "_proxy")


def test_proxy_parent_read_only_and_dir_target_only():
    """Garante que .parent não aceita atribuição direta e dir(proxy) devolve apenas atributos do target."""
    hist = GlobalHistory()
    img = Image(np.zeros((10, 10, 4), dtype=np.uint8), ImageFormat.RGBA)
    raw_layer = Layer(img, name="L1")
    proxy = ProxyLayer(raw_layer, hist)

    # 1. Atribuição direta a .parent lança AttributeError
    with pytest.raises(AttributeError, match="Direct assignment to 'parent' is not supported"):
        proxy.parent = "Novo Pai"

    # 2. dir(proxy) devolve exatamente os atributos do target real
    assert dir(proxy) == dir(raw_layer)
