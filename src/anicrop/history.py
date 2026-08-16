from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from collections import deque
from contextlib import contextmanager

if TYPE_CHECKING:
    from anicrop.command import Command


class ActionPolicy(ABC):
    """Interface abstrata (Strategy/Policy) para os modos de ação do histórico."""

    @abstractmethod
    def start_action(self, history: GlobalHistory, command_cls: type[Command], name: str, target: Any, value: Any = None) -> None:
        ...

    @abstractmethod
    def commit(self, history: GlobalHistory) -> bool:
        ...


class NormalPolicy(ActionPolicy):
    """Política padrão: cria um novo comando e sela o anterior."""

    def start_action(self, history: GlobalHistory, command_cls: type[Command], name: str, target: Any, value: Any = None) -> None:
        history._clear_redo()
        history.commit()
        cmd = command_cls(name, target, value)
        history._undo_stack.append(cmd)

    def commit(self, history: GlobalHistory) -> bool:
        if not history.undo_empty():
            last_cmd = history._undo_stack[-1]
            if not last_cmd._sealed:
                last_cmd.seal()
                if not last_cmd.has_changes():
                    history._undo_stack.pop()
                return True
        return False


class MergeContinuousPolicy(ActionPolicy):
    """Política de mesclagem contínua: mescla ações de mesmo nome e objeto."""

    def start_action(self, history: GlobalHistory, command_cls: type[Command], name: str, target: Any, value: Any = None) -> None:
        if not history.undo_empty():
            last_cmd = history._undo_stack[-1]
            if type(last_cmd) is command_cls and last_cmd.can_merge(name, target):
                return

        history._clear_redo()
        history.commit()
        cmd = command_cls(name, target, value)
        history._undo_stack.append(cmd)

    def commit(self, history: GlobalHistory) -> bool:
        return False


class GroupActionPolicy(ActionPolicy):
    """Política de agrupamento: ignora ações consecutivas da mesma classe de comando."""

    def start_action(self, history: GlobalHistory, command_cls: type[Command], name: str, target: Any, value: Any = None) -> None:
        if not history.undo_empty():
            last_cmd = history._undo_stack[-1]
            if type(last_cmd) is command_cls:
                return

        history._clear_redo()
        history.commit()
        cmd = command_cls(name, target, value)
        history._undo_stack.append(cmd)

    def commit(self, history: GlobalHistory) -> bool:
        return False


class DisabledPolicy(ActionPolicy):
    """Política silenciosa/desativada: ignora qualquer início de ação e commit."""

    def start_action(self, history: GlobalHistory, command_cls: type[Command], name: str, target: Any, value: Any = None) -> None:
        pass

    def commit(self, history: GlobalHistory) -> bool:
        return False


class GlobalHistory:

    def __init__(self) -> None:
        self._undo_stack: deque[Command] = deque()
        self._redo_stack: deque[Command] = deque()
        self._policy: ActionPolicy = NormalPolicy()

    @property
    def is_active(self) -> bool:
        return not isinstance(self._policy, DisabledPolicy)

    def _clear_redo(self) -> None:
        self._redo_stack.clear()

    def commit(self) -> bool:
        return self._policy.commit(self)

    def start_action(self, command_cls: type[Command], name: str, target: Any, value: Any = None) -> None:
        """Abre uma nova transação usando a política ativa."""
        self._policy.start_action(self, command_cls, name, target, value)

    def undo(self) -> None:
        if self.undo_empty():
            raise IndexError("Undo stack is empty")

        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)

    def redo(self) -> None:
        if self.redo_empty():
            raise IndexError("Redo stack is empty")

        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)

    def undo_empty(self) -> bool:
        return len(self._undo_stack) == 0

    def redo_empty(self) -> bool:
        return len(self._redo_stack) == 0

    @contextmanager
    def use_policy(self, policy: ActionPolicy):
        old_policy = self._policy
        self._policy = policy
        try:
            yield
        finally:
            self._policy = old_policy
            self.commit()

    @contextmanager
    def transaction(self):
        """Contexto de transação padrão."""
        with self.use_policy(NormalPolicy()):
            yield

    @contextmanager
    def merge_continuous(self):
        """Contexto de mesclagem por nome."""
        with self.use_policy(MergeContinuousPolicy()):
            yield

    @contextmanager
    def group_action(self):
        """Contexto de agrupamento por classe de comando."""
        with self.use_policy(GroupActionPolicy()):
            yield

    @contextmanager
    def disabled(self):
        """Contexto que desativa temporariamente a gravação de ações no histórico."""
        with self.use_policy(DisabledPolicy()):
            yield
