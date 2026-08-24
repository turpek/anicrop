import numpy as np
import pytest

from anicrop.content import FitContext
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


def test_resize_undo_single_step_restores_and_empties_history():
    """Valida se uma única chamada a doc.history.undo desfaz o resize completamente."""
    doc = Document("TestResizeHistory", 200, 200, history=True)
    data = np.full((50, 50, 4), 255, dtype=np.uint8)
    layer = doc.add(Layer(Image(data, ImageFormat.RGBA), name="L1"))

    doc.history._undo_stack.clear()
    initial_region = layer.global_region

    doc.content.resize(layer, 150, 120)
    assert layer.global_region.size == (150, 120)

    doc.history.undo()
    assert doc.history.undo_empty()
    assert layer.global_region == initial_region


def test_fit_undo_single_step_restores_and_empties_history():
    """Valida se uma única chamada a doc.history.undo desfaz o fit de forma atomica."""
    doc = Document("TestFitHistory", 400, 400, history=True)
    data = np.full((100, 200, 4), 255, dtype=np.uint8)
    layer = doc.add(Layer(Image(data, ImageFormat.RGBA), name="L1"))

    doc.history._undo_stack.clear()
    initial_region = layer.global_region

    doc.content.fit(layer, doc.canvas)
    assert layer.global_region == Region.from_size(400, 400)

    doc.history.undo()
    assert doc.history.undo_empty()
    assert layer.global_region == initial_region


def test_fit_helper_contain_undo_single_step_restores_and_empties_history():
    """Valida se uma única chamada a doc.history.undo desfaz o fit via FitContext helper de forma atomica."""
    doc = Document("DocContentFitHelperTest", 100, 100, history=True)
    data = np.full((100, 200, 4), 255, dtype=np.uint8)
    layer = doc.add(Layer(Image(data, ImageFormat.RGBA), name="L1"))

    doc.history._undo_stack.clear()
    initial_region = layer.global_region

    doc.content.fit(FitContext(layer, doc.canvas).fit_contain)
    assert layer.global_region == Region.from_rect(0, 25, 100, 50)

    doc.history.undo()
    assert doc.history.undo_empty()
    assert layer.global_region == initial_region


def test_flip_x_undo_single_step_restores_and_empties_history():
    """Valida se uma única chamada a doc.history.undo desfaz o flip_x de forma atomica."""
    doc = Document("TestFlipXHistory", 100, 100, history=True)
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = doc.add(Layer(Image(data, ImageFormat.RGBA), name="L1"))

    doc.history._undo_stack.clear()

    doc.content.flip_x(layer)
    assert layer.matrix[0, 0] == -1.0

    doc.history.undo()
    assert doc.history.undo_empty()
    assert layer.matrix[0, 0] == 1.0


def test_flip_y_undo_single_step_restores_and_empties_history():
    """Valida se uma única chamada a doc.history.undo desfaz o flip_y de forma atomica."""
    doc = Document("TestFlipYHistory", 100, 100, history=True)
    data = np.full((100, 100, 4), 255, dtype=np.uint8)
    layer = doc.add(Layer(Image(data, ImageFormat.RGBA), name="L1"))

    doc.history._undo_stack.clear()

    doc.content.flip_y(layer)
    assert layer.matrix[1, 1] == -1.0

    doc.history.undo()
    assert doc.history.undo_empty()
    assert layer.matrix[1, 1] == 1.0
