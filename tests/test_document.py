import numpy as np
import pytest

from anicrop.document import DirectDocumentPolicy, Document, ReactiveDocumentPolicy
from anicrop.enums import ImageFormat
from anicrop.geometry import FitGeometry
from anicrop.history import GlobalHistory
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.proxy import GroupProxy, LayerStackProxy, ProxyLayer


def make_img(w: int = 10, h: int = 10) -> Image:
    return Image(np.zeros((h, w, 4), dtype=np.uint8), ImageFormat.RGBA)


def test_document_reactive_mode():
    """Valida inicialização do documento em modo reativo com histórico e proxies."""
    doc = Document("TestDoc", 100, 100, wrap_proxy=True)

    assert isinstance(doc.history, GlobalHistory)
    assert isinstance(doc.stack, LayerStackProxy)

    raw_layer = Layer(make_img(), name="L1")
    added_layer = doc.add(raw_layer)

    assert isinstance(added_layer, ProxyLayer)
    assert doc[0] is added_layer

    added_layer.name = "Renamed L1"
    assert raw_layer.name == "Renamed L1"

    doc.history.undo()
    assert raw_layer.name == "L1"


def test_document_direct_mode():
    """Valida inicialização do documento em modo direto de alta performance sem proxies."""
    doc = Document("TestDoc", 100, 100, wrap_proxy=False)

    assert doc.history is None
    assert not isinstance(doc.stack, LayerStackProxy)

    raw_layer = Layer(make_img(), name="L1")
    added_layer = doc.add(raw_layer)

    assert not isinstance(added_layer, ProxyLayer)
    assert isinstance(added_layer, Layer)
    assert doc[0] is raw_layer


def test_document_canvas_properties_and_resize():
    """Valida propriedades de dimensão no objeto Canvas do Document."""
    from anicrop.spatial import Region
    doc = Document("TestDoc", 800, 600)

    assert doc.canvas.width == 800
    assert doc.canvas.height == 600
    assert doc.canvas.size == (800, 600)

    doc.canvas.region = Region.from_size(1920, 1080)

    assert doc.canvas.width == 1920
    assert doc.canvas.height == 1080
    assert doc.canvas.size == (1920, 1080)


def test_document_add_and_duplicate_name_error():
    """Valida se a adição de camada com nome duplicado levanta ValueError estrito."""
    doc = Document("TestDoc", 100, 100)
    l1 = Layer(make_img(), name="camada1")
    l2 = Layer(make_img(), name="camada1")

    doc.add(l1)

    with pytest.raises(ValueError, match="A layer named 'camada1' already exists"):
        doc.add(l2)


def test_document_add_group_and_duplicate_name_error():
    """Valida criação de grupo e prevenção de colisão de nomes com grupos."""
    doc = Document("TestDoc", 100, 100)
    doc.add_group("grupo1")

    with pytest.raises(ValueError, match="A layer named 'grupo1' already exists"):
        doc.add_group("grupo1")


def test_document_container_sequence_protocol():
    """Valida protocolo de sequência (__len__, __getitem__, __iter__, __contains__)."""
    doc = Document("TestDoc", 100, 100)
    l1 = doc.add(Layer(make_img(), name="fundo"))
    l2 = doc.add(Layer(make_img(), name="texto"))
    l3 = doc.add(Layer(make_img(), name="overlay"))

    assert len(doc) == 3
    assert doc[0] is l1
    assert doc[1] is l2
    assert doc[-1] is l3
    assert list(doc) == [l1, l2, l3]
    assert "fundo" in doc
    assert "inexistente" not in doc
    assert l2 in doc


def test_document_getitem_by_name_and_keyerror():
    """Valida acesso a camadas por nome via string e KeyError para chaves inexistentes."""
    doc = Document("TestDoc", 100, 100)
    l1 = doc.add(Layer(make_img(), name="fundo"))

    assert doc["fundo"] is l1

    with pytest.raises(KeyError, match="Layer named 'nao_existe' not found"):
        _ = doc["nao_existe"]


def test_document_find_layer_recursive():
    """Valida busca de camadas aninhadas dentro de grupos usando find()."""
    doc = Document("TestDoc", 100, 100)
    group = doc.add_group("grupo_pai")
    filho = Layer(make_img(), name="filho_aninhado")
    group.append(filho)

    encontrado = doc.find("filho_aninhado", recursive=True)

    assert encontrado is not None
    assert getattr(encontrado, "name", "") == "filho_aninhado"
    assert doc.find("filho_aninhado", recursive=False) is None


def test_document_delitem_by_name_and_index():
    """Valida remoção de camadas via del doc[name] e del doc[index]."""
    doc = Document("TestDoc", 100, 100)
    l1 = doc.add(Layer(make_img(), name="l1"))
    l2 = doc.add(Layer(make_img(), name="l2"))
    l3 = doc.add(Layer(make_img(), name="l3"))

    del doc["l2"]
    assert len(doc) == 2
    assert "l2" not in doc

    del doc[0]
    assert len(doc) == 1
    assert doc[0] is l3


def test_document_remove_and_pop_and_clear():
    """Valida métodos de mutação da pilha remove, pop e clear."""
    doc = Document("TestDoc", 100, 100)
    l1 = doc.add(Layer(make_img(), name="l1"))
    l2 = doc.add(Layer(make_img(), name="l2"))
    l3 = doc.add(Layer(make_img(), name="l3"))

    doc.remove("l1")
    assert len(doc) == 2
    assert "l1" not in doc

    popped = doc.pop(0)
    assert popped is l2
    assert len(doc) == 1

    doc.clear()
    assert len(doc) == 0


def test_document_render_in_memory():
    """Valida renderização em alta resolução direto para objeto Image em memória."""
    doc = Document("TestDoc", 50, 50)
    img_data = np.ones((50, 50, 4), dtype=np.uint8) * 200
    doc.add(Layer(Image(img_data, ImageFormat.RGBA), name="l1"))

    rendered_img = doc.render()

    assert isinstance(rendered_img, Image)
    assert rendered_img.size == (50, 50)
    assert rendered_img.format == ImageFormat.RGBA


def test_document_backward_compatibility_aliases():
    """Valida funcionamento dos aliases legados de compatibilidade."""
    doc = Document("TestDoc", 100, 100)
    l1 = doc.add_layer(Layer(make_img(), name="l1"))
    g1 = doc.create_group("g1")

    assert doc.get_bottom_layer() is g1
    assert len(doc) == 2
    assert doc["l1"] is l1
