from anicrop.history import GlobalHistory, NormalPolicy
from pytest import fixture
import pytest
from typing import Any


class FakeCommand:

    def __init__(self, name: str, item: Any, value: Any = None):
        self.execute_count = 0
        self._sealed = False
        self.state = item.state
        self.value = value
        self.layer = item
        self.name = name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(value={self.value})"

    def execute(self):
        self.execute_count += 1
        if self.value is not None:
            self.layer.state = self.value

    def seal(self):
        if not self._sealed:
            self.value = self.layer.state
            self._sealed = True

    def is_sealed(self):
        return self._sealed

    def undo(self):
        if not self._sealed:
            self.seal()
        self.layer.state = self.state

    def has_changes(self) -> bool:
        return True

    def can_merge(self, name: str, layer: object) -> bool:
        if self.is_sealed():
            return False
        return self.name == name and self.layer == layer


class FakeRotationCommand(FakeCommand):
    pass


class FakeScaleCommand(FakeCommand):
    pass


class FakeTranslateCommand(FakeCommand):
    pass


class FakeNoChangeCommand(FakeCommand):

    def has_changes(self) -> bool:
        return False


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
    history.start_action(FakeCommand, 'fake', mock_layer)
    assert not history.undo_empty()


def test_GlobalHistory_redo_nao_vazia(mocker, history):
    mock_layer = make_layer()
    history.start_action(FakeCommand, 'fake', mock_layer)
    history.undo()
    assert not history.redo_empty()


def test_GlobalHistory_undo_com_undo_stack_vazia(mocker, history):
    with pytest.raises(IndexError, match="Undo stack is empty"):
        history.undo()


def test_GlobalHistory_undo(mocker, history):
    spy_undo = mocker.spy(FakeCommand, "undo")
    layer = make_layer()
    history.start_action(FakeCommand, 'fake', layer)
    history.undo()
    assert history.undo_empty()
    assert spy_undo.call_count == 1
    assert not history.redo_empty()


def test_GlobalHistory_redo_com_undo_stack_vazia(mocker, history):
    with pytest.raises(IndexError, match="Redo stack is empty"):
        history.redo()


def test_GlobalHistory_redo_apos_undo(mocker, history):
    layer = make_layer()
    history.start_action(FakeCommand, 'fake', layer)
    cmd = history._undo_stack[-1]
    spy_execute = mocker.spy(cmd, "execute")
    history.undo()
    history.redo()
    assert spy_execute.call_count == 1


def test_GlobalHistory_redo_empty_apos_um_undo_seguido_de_push(mocker, history):
    layer1 = make_layer()
    layer2 = make_layer()
    history.start_action(FakeCommand, 'fake', layer1)
    history.undo()
    history.start_action(FakeCommand, 'fake', layer2)
    assert history.redo_empty()


def test_GlobalHistory_commit_apos_mudar_comando(mocker, history):
    layer = make_layer()
    history.start_action(FakeScaleCommand, 'scale', layer)
    history.start_action(FakeRotationCommand, 'rotation', layer)
    history.start_action(FakeRotationCommand, 'rotation', layer)
    history.start_action(FakeScaleCommand, 'scale', layer)
    cmd = history._undo_stack[-2]
    assert cmd.is_sealed()


def test_GlobalHistory_commit_manualmente(mocker, history):
    layer = make_layer()
    history.start_action(FakeScaleCommand, 'scale', layer)
    assert history.commit()

    cmd = history._undo_stack[-1]
    assert cmd.is_sealed()


def test_GlobalHistory_commit_manualmente_com_undo_vazio(mocker, history):
    assert not history.commit()


@pytest.mark.parametrize(
    'context_manager_name, expected_size', [
        ('transaction', 5),
        ('atomic', 1),
        ('merge_continuous', 4),
        ('group_action', 3),
        ('disabled', 0)
    ],
)
def test_GlobalHistory_context_modes(history, context_manager_name, expected_size):
    class FakeClass:
        def __init__(self):
            self.state = 0

    layer1 = FakeClass()
    layer2 = FakeClass()

    context_manager = getattr(history, context_manager_name)

    with context_manager():
        # Comando Fake 1, Classe 1
        history.commit()
        history.start_action(FakeRotationCommand, 'prop1', layer1)
        layer1.state = 1
        history.commit()

        # Comando Fake 1, Classe 1
        history.start_action(FakeRotationCommand, 'prop1', layer1)
        layer1.state = 2
        history.commit()

        # Comando Fake 1, Classe 1
        history.start_action(FakeRotationCommand, 'prop1-b', layer1)
        layer1.state = 3
        history.commit()

        # Comando Fake 2, Classe 2
        history.start_action(FakeScaleCommand, 'prop2', layer2)
        layer2.state = 1
        history.commit()

        # Comando Fake 1, Classe 1
        history.start_action(FakeRotationCommand, 'prop1', layer1)
        layer1.state = 4
        history.commit()

    assert len(history._undo_stack) == expected_size
    assert isinstance(history._policy, NormalPolicy)


def test_undo_discards_unmutated_command_at_top_of_stack(history):
    """Valida se undo() descarta comando sem alteração no topo e desfaz a ação real anterior."""
    class Target:
        state = 0

    target = Target()
    history.start_action(FakeCommand, 'real_action', target, value=10)
    target.state = 10
    history.commit()

    history.start_action(FakeNoChangeCommand, 'read_action', target)
    assert len(history._undo_stack) == 2

    history.undo()
    assert history.undo_empty()
    assert target.state == 0
