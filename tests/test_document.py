import numpy as np
import pytest
from anicrop.document import DirectDocumentPolicy, Document, ReactiveDocumentPolicy
from anicrop.history import GlobalHistory
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.proxy import LayerStackProxy, ProxyLayer


def make_img(w=10, h=10):
    return Image(np.zeros((h, w, 4), dtype=np.uint8), ImageFormat.RGBA)


def test_document_reactive_mode():
    """Testa modo reativo (wrap_proxy=True, padrão)."""
    doc = Document("TestDoc", 100, 100, wrap_proxy=True)

    assert isinstance(doc.history, GlobalHistory)
    assert isinstance(doc.stack, LayerStackProxy)

    raw_layer = Layer(make_img(), name="L1")
    added_layer = doc.add_layer(raw_layer)

    assert isinstance(added_layer, ProxyLayer)
    assert doc.stack[0] is added_layer

    # Modifica propriedade e testa histórico
    added_layer.name = "Renamed L1"
    assert raw_layer.name == "Renamed L1"

    doc.history.undo()
    assert raw_layer.name == "L1"


def test_document_direct_mode():
    """Testa modo direto/stateless (wrap_proxy=False)."""
    doc = Document("TestDoc", 100, 100, wrap_proxy=False)

    assert doc.history is None
    assert not isinstance(doc.stack, LayerStackProxy)

    raw_layer = Layer(make_img(), name="L1")
    added_layer = doc.add_layer(raw_layer)

    assert not isinstance(added_layer, ProxyLayer)
    assert isinstance(added_layer, Layer)
    assert doc.stack[0] is raw_layer
