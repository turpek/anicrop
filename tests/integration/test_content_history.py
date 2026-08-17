import numpy as np
import pytest

from anicrop.document import Document
from anicrop.enums import ImageFormat
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.spatial import Region


def test_crop_undo_single_step_restores_and_empties_history():
    """Valida se uma única chamada a doc.history.undo desfaz o crop e esvazia a pilha de undo."""
    doc = Document("TestCropHistory", 100, 100, history=True)
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = doc.add(Layer(Image(data, ImageFormat.RGBA), name="L1"))

    # Limpa a acao inicial de criacao/adicao da camada na pilha
    doc.history._undo_stack.clear()

    initial_region = layer.global_region
    initial_edits_count = len(layer._edits)

    doc.content.crop(layer, (20, 20, 40, 40))

    # Desfaz a operação de crop
    doc.history.undo()

    # O histórico deve estar vazio e a camada totalmente restaurada
    assert doc.history.undo_empty()
    assert layer.global_region == initial_region
    assert len(layer._edits) == initial_edits_count
