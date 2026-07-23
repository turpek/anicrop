from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer, EditLayer
from anicrop.proxy import ProxyLayer, is_property_with_setter
from anicrop.history import GlobalHistory
from anicrop.command import SnapshotCommand
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
    """Testa se a escrita em atributos passa pelo histórico usando SnapshotCommand."""
    proxy.name = "New Name"

    # Verifica se o comando foi enviado para o histórico
    assert history.push.call_count == 2
    args, _ = history.push.call_args
    command_class, attr_name, target, value = args

    assert command_class == SnapshotCommand
    assert attr_name == "name"
    assert target is layer
    assert isinstance(value, dict)
    # A primeira chamada mandou o old_state, a segunda o new_state
    args_old, _ = history.push.call_args_list[0]
    args_new, _ = history.push.call_args_list[1]
    
    assert args_old[3]["name"] == "Layer"
    assert args_new[3]["name"] == "New Name"

    # O ProxyLayer aplica a mudança diretamente no layer real
    assert layer.name == "New Name"


def test_ProxyLayer_integration_with_real_history(layer):
    """Garante o funcionamento do Undo/Redo real com o ProxyLayer usando SnapshotCommand."""
    real_history = GlobalHistory()
    proxy = ProxyLayer(layer, real_history)

    # 1. Testando propriedade atômica
    proxy.name = "Modified Name"
    assert layer.name == "Modified Name"

    real_history.undo()
    assert layer.name == "Layer"  # Volta ao original

    real_history.redo()
    assert layer.name == "Modified Name"  # Refaz

    # 2. Testando transformações encadeadas via sub-proxy GenericProxy (proxy.transform.rotate(90))
    proxy.transform.rotate(90).translate(50, 50)

    pt = np.array([0, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(layer.transform.matrix @ pt, [60, 50, 1], atol=1e-4)

    # Undo da rotação e translação juntas (mescladas no mesmo comando "transform")
    real_history.undo()
    assert layer.transform_used is False

    # Redo restaura a transformação completa
    real_history.redo()
    np.testing.assert_allclose(layer.transform.matrix @ pt, [60, 50, 1], atol=1e-4)


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


def test_ProxyLayer_atribuicao_composta_chama_push_uma_unica_vez(proxy, history):
    """Garante que a atribuição composta (+=) chama o push do histórico exatamente uma vez."""
    proxy.opacity = 0.5
    history.reset_mock()

    proxy.opacity += 0.2

    assert history.push.call_count == 3



# --- Testes Unitários de Cenários para is_property_with_setter ---

class ParentFake:
    @property
    def parent_readonly_prop(self) -> int:
        return 1

    @property
    def parent_writable_prop(self) -> int:
        return 2

    @parent_writable_prop.setter
    def parent_writable_prop(self, val: int) -> None:
        pass


class ChildFake(ParentFake):
    def __init__(self):
        self.instance_var = 10

    @property
    def child_readonly_prop(self) -> str:
        return "readonly"

    @property
    def child_writable_prop(self) -> str:
        return "writable"

    @child_writable_prop.setter
    def child_writable_prop(self, val: str) -> None:
        pass

    def some_method(self) -> None:
        pass


@pytest.mark.parametrize("name, expected", [
    ("child_writable_prop", True),       # Cenário 1: Property com setter definida na própria classe
    ("child_readonly_prop", False),      # Cenário 2: Property sem setter (somente getter) na própria classe
    ("parent_writable_prop", True),      # Cenário 3: Property com setter herdada
    ("parent_readonly_prop", False),     # Cenário 4: Property sem setter herdada
    ("instance_var", False),             # Cenário 5: Atributo normal de instância
    ("some_method", False),              # Cenário 6: Método normal da classe
    ("non_existent_attribute", False),   # Cenário 7: Atributo inexistente
])
def test_is_property_with_setter_cenarios(name, expected):
    assert is_property_with_setter(ChildFake, name) is expected


