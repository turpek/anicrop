from anicrop.history import GlobalHistory
from pytest import fixture
import pytest


class FakeCommand:

    def __init__(self, layer, value):
        self.update_value_count = 0
        self.execute_count = 0
        self._sealed = False
        self.value = value
        self.state = layer.state
        self.layer = layer

    def __repr__(self) -> str:
        return f"{type(self).__name__}(value={self.value})"

    def execute(self):
        self.execute_count += 1
        self.layer.state = self.value

    def seal(self):
        self._sealed = True

    def is_sealed(self):
        return self._sealed

    def undo(self):
        self.layer.state = self.state

    def update_value(self, value):
        self.update_value_count += 1
        self.value = value

    def can_merge(self, command_cls, layer):
        if self.is_sealed():
            return False
        return isinstance(self, command_cls) and self.layer == layer


class FakeRotationCommand(FakeCommand):
    ...


class FakeScaleCommand(FakeCommand):
    ...


class FakeTranslateCommand(FakeCommand):
    ...


@fixture
def history():
    return GlobalHistory()


def make_layer():
    obj = type("FakeLayer", (), {"state": 0})()
    return obj


def test_GlobalHistory_vazia(history):
    assert history.undo_empty()
    assert history.redo_empty()


def test_GlobalHistory_undo_nao_vazia(mocker, history):
    mock_layer = make_layer()
    history.push(FakeCommand, mock_layer, 45)
    assert not history.undo_empty()

    cmd = history._undo_stack[-1]
    assert cmd.execute_count == 1


def test_GlobalHistory_redo_nao_vazia(mocker, history):
    mock_layer = make_layer()
    history.push(FakeCommand, mock_layer, 45)
    history.undo()
    assert not history.redo_empty()


def test_GlobalHistory_push_mesmo_comando_mesmo_layer(mocker, history):

    layer = make_layer()
    history.push(FakeCommand, layer, 10)

    cmd = history._undo_stack[-1]
    spy_update = mocker.spy(cmd, "update_value")
    mocker.patch.object(cmd, "can_merge", return_value=True)

    history.push(FakeCommand, layer, 20)

    assert len(history._undo_stack) == 1
    spy_update.assert_called_once_with(20)
    assert cmd.execute_count == 2


def test_GlobalHistory_push_mesmo_comando_mas_layer_diferente(mocker, history):
    layer1 = make_layer()
    layer2 = make_layer()
    history.push(FakeCommand, layer1, 10)
    history.push(FakeCommand, layer2, 20)
    assert len(history._undo_stack) == 2


def test_GlobalHistory_push_mesmo_layer_sem_merge(mocker, history):
    layer = object()

    layer = make_layer()
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
    layer = make_layer()
    history.push(FakeCommand, layer, 10)
    history.undo()
    assert history.undo_empty()
    assert spy_undo.call_count == 1
    assert not history.redo_empty()


def test_GlobalHistory_redo_com_undo_stack_vazia(mocker, history):
    with pytest.raises(IndexError, match="Redo stack is empty"):
        history.redo()


def test_GlobalHistory_redo_apos_undo(mocker, history):
    layer = make_layer()
    history.push(FakeCommand, layer, 10)
    cmd = history._undo_stack[-1]
    spy_execute = mocker.spy(cmd, "execute")
    history.undo()
    history.redo()
    assert spy_execute.call_count == 1


def test_GlobalHistory_redo_empty_apos_um_undo_seguido_de_push(mocker, history):
    layer1 = make_layer()
    layer2 = make_layer()
    history.push(FakeCommand, layer1, 10)
    history.undo()
    history.push(FakeCommand, layer2, 20)
    assert history.redo_empty()


def test_GlobalHistory_merge_nao_cria_novo_command(mocker, history):
    layer = make_layer()

    spy_init = mocker.spy(FakeCommand, "__init__")

    history.push(FakeCommand, layer, 10)
    cmd = history._undo_stack[-1]

    mocker.patch.object(cmd, "can_merge", return_value=True)

    history.push(FakeCommand, layer, 20)

    # __init__ deve ter sido chamado apenas uma vez
    assert spy_init.call_count == 1


def test_GlobalHistory_commit_apos_mudar_comando(mocker, history):
    layer = make_layer()
    history.push(FakeScaleCommand, layer, 1.5)
    history.push(FakeRotationCommand, layer, 10)
    history.push(FakeRotationCommand, layer, 30)
    history.push(FakeScaleCommand, layer, 2)
    cmd = history._undo_stack[-2]
    assert cmd.is_sealed()


def test_GlobalHistory_commit_manualmente(mocker, history):
    layer = make_layer()
    history.push(FakeScaleCommand, layer, 1.5)
    assert history.commit()

    cmd = history._undo_stack[-1]
    assert cmd.is_sealed()


def test_GlobalHistory_commit_manualmente_com_undo_vazio(mocker, history):
    assert not history.commit()


def test_GlobalHistory_comandos_iguais_apos_undo(mocker, history):
    layer = make_layer()
    history.push(FakeRotationCommand, layer, 10)
    history.push(FakeRotationCommand, layer, 20)
    history.push(FakeScaleCommand, layer, 2)
    history.undo()
    history.push(FakeRotationCommand, layer, 30)
    cmds = history._undo_stack
    assert cmds[1].value == 30
    assert cmds[0].value == 20
