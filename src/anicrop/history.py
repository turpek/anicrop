from __future__ import annotations
from anicrop.command import Command
from anicrop.layer import Layer
from collections import deque
from contextlib import contextmanager


class GlobalHistory:

    def __init__(self):
        self._undo_stack = deque()
        self._redo_stack = deque()
        self._current_start_action = self._start_action_normal
        self._current_commit = self._commit

    def _clear_redo(self) -> None:
        self._redo_stack.clear()

    def _commit(self) -> bool:
        """Sela a transação pendente capturando o snapshot final."""
        if not self.undo_empty():
            last_cmd = self._undo_stack[-1]
            if not last_cmd._sealed:
                last_cmd.seal()
                if not last_cmd.has_changes():
                    self._undo_stack.pop()
                return True
        return False

    def _no_commit(self) -> bool:
        ...

    def commit(self) -> bool:
        return self._current_commit()

    def start_action(self, command_cls: type[Command], name: str, layer: Layer) -> None:
        """Abre uma nova transação. Sela a anterior se houver."""
        self._current_start_action(command_cls, name, layer)

    def _start_action_normal(self, command_cls: type[Command], name: str, layer: Layer) -> None:
        self._clear_redo()
        self.commit()
        cmd = command_cls(name, layer)
        self._undo_stack.append(cmd)

    def _start_action_merge(self, command_cls: type[Command], name: str, layer: Layer) -> None:
        if not self.undo_empty():
            last_cmd = self._undo_stack[-1]
            if type(last_cmd) is command_cls and last_cmd.can_merge(name, layer):
                return

        self._clear_redo()
        self.commit()
        cmd = command_cls(name, layer)
        self._undo_stack.append(cmd)

    def _start_action_group(self, command_cls: type[Command], name: str, layer: Layer) -> None:
        if not self.undo_empty():
            last_cmd = self._undo_stack[-1]
            if type(last_cmd) is command_cls:
                return

        self._clear_redo()
        self.commit()
        cmd = command_cls(name, layer)
        self._undo_stack.append(cmd)

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
    def _with_strategy(self, strategy_method, commit_method):
        old_strategy = self._current_start_action
        old_commit = self._current_commit

        self._current_start_action = strategy_method
        self._current_commit = commit_method
        try:
            yield
        finally:
            self._current_start_action = old_strategy
            self._current_commit = old_commit
            self.commit()

    @contextmanager
    def transaction(self):
        """Opção 1: Fluxo atual, garantindo commit no final."""
        with self._with_strategy(self._start_action_normal, self._commit):
            yield

    @contextmanager
    def merge_continuous(self):
        """Opção 2: Merge por nome."""
        with self._with_strategy(self._start_action_merge, self._no_commit):
            yield

    @contextmanager
    def group_action(self):
        """Opção 3: Merge por tipo de comando."""
        with self._with_strategy(self._start_action_group, self._no_commit):
            yield
