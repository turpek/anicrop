from __future__ import annotations
from anicrop.command import Command
from anicrop.layer import Layer
from collections import deque
from typing import Any


class GlobalHistory:

    def __init__(self):
        self._undo_stack = deque()
        self._redo_stack = deque()

    def _clear_redo(self) -> None:
        self._redo_stack.clear()

    def _instantiate_command(self, command_cls: type[Command], layer: Layer, value: Any) -> Command:
        cmd = command_cls(layer, value)
        self._undo_stack.append(cmd)
        return cmd

    def _can_instantiate_command(self, command_cls: type[Command], layer: Layer) -> bool:
        return self.undo_empty() or not self._undo_stack[-1].can_merge(command_cls, layer)

    def _update_command(self, value: Any) -> Command:
        cmd = self._undo_stack[-1]
        cmd.update_value(value)
        return cmd

    def push(self, command_cls: type[Command], layer: Layer, value: Any) -> None:
        self._clear_redo()

        if self._can_instantiate_command(command_cls, layer):
            cmd = self._instantiate_command(command_cls, layer, value)
        else:
            cmd = self._update_command(value)
        cmd.execute()

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
