from anicrop.history import GlobalHistory
from pytest import fixture
import pytest


class FakeCommand:

    def __init__(self, layer, value):
        self.update_value_count = 0
        self.execute_count = 0

    def execute(self):
        self.execute_count += 1
        ...

    def undo(self):
        ...

    def update_value(self, value):
        self.update_value_count += 1
        ...

    def can_merge(self, command_cls, layer):
        return False


@fixture
def history():
    return GlobalHistory()


def test_GlobalHistory_vazia(history):
    assert history.undo_empty()
    assert history.redo_empty()


def test_GlobalHistory_undo_nao_vazia(mocker, history):
    mock_layer = mocker.MagicMock()
    history.push(FakeCommand, mock_layer, 45)
    assert not history.undo_empty()

    cmd = history._undo_stack[-1]
    assert cmd.execute_count == 1


def test_GlobalHistory_redo_nao_vazia(mocker, history):
    mock_layer = mocker.MagicMock()
    history.push(FakeCommand, mock_layer, 45)
    history.undo()
    assert not history.redo_empty()


def test_GlobalHistory_push_mesmo_comando_mesmo_layer(mocker, history):
    layer = object()

    history.push(FakeCommand, layer, 10)

    cmd = history._undo_stack[-1]
    spy_update = mocker.spy(cmd, "update_value")
    mocker.patch.object(cmd, "can_merge", return_value=True)

    history.push(FakeCommand, layer, 20)

    assert len(history._undo_stack) == 1
    spy_update.assert_called_once_with(20)
    assert cmd.execute_count == 2


def test_GlobalHistory_push_mesmo_comando_mas_layer_diferente(mocker, history):
    layer1, layer2 = object(), object()
    history.push(FakeCommand, layer1, 10)
    history.push(FakeCommand, layer2, 20)
    assert len(history._undo_stack) == 2


def test_GlobalHistory_push_mesmo_layer_sem_merge(mocker, history):
    layer = object()

    history.push(FakeCommand, layer, 10)
    cmd = history._undo_stack[-1]
    mocker.patch.object(cmd, "can_merge", return_value=False)
    history.push(FakeCommand, layer, 20)
    assert len(history._undo_stack) == 2


def test_GlobalHistory_undo_com_undo_stack_vazia(mocker, history):
    with pytest.raises(IndexError, match="Undo stack is empty"):
        history.undo()


def test_GlobalHistory_undo(mocker, history):
    spy_undo = mocker.spy(FakeCommand, "undo")
    history.push(FakeCommand, object(), 10)
    history.undo()
    assert history.undo_empty()
    assert spy_undo.call_count == 1
    assert not history.redo_empty()


def test_GlobalHistory_redo_com_undo_stack_vazia(mocker, history):
    with pytest.raises(IndexError, match="Redo stack is empty"):
        history.redo()


def test_GlobalHistory_redo_apos_undo(mocker, history):
    history.push(FakeCommand, object(), 10)
    cmd = history._undo_stack[-1]
    spy_execute = mocker.spy(cmd, "execute")
    history.undo()
    history.redo()
    assert spy_execute.call_count == 1


def test_GlobalHistory_redo_empty_apos_um_undo_seguido_de_push(mocker, history):
    history.push(FakeCommand, object(), 10)
    history.undo()
    history.push(FakeCommand, object(), 20)
    assert history.redo_empty()


def test_GlobalHistory_merge_nao_cria_novo_command(mocker, history):
    layer = object()

    spy_init = mocker.spy(FakeCommand, "__init__")

    history.push(FakeCommand, layer, 10)
    cmd = history._undo_stack[-1]

    mocker.patch.object(cmd, "can_merge", return_value=True)

    history.push(FakeCommand, layer, 20)

    # __init__ deve ter sido chamado apenas uma vez
    assert spy_init.call_count == 1
