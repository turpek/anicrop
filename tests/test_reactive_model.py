from __future__ import annotations

from typing import Any

import pytest

from anicrop.command import Command
from anicrop.history import GlobalHistory
from anicrop.reactive import (
    BaseFluentProxy,
    BaseHistoryProxy,
    ProxyRegistry,
    StrategyProxy,
)


class DummyEntity:
    """Classe dummy para testar o BaseHistoryProxy genérico."""

    def __init__(self, name: str = "Init", value: int = 10):
        self._name = name
        self.value = value
        self.untracked = "raw"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        self._name = val


class DummyCustomProxy(BaseHistoryProxy[DummyEntity]):
    """Subclasse de proxy que declara atributos ignorados."""

    _IGNORED_ATTRIBUTES = frozenset({"untracked"})


class DummyChain:
    """Classe dummy com interface fluente (method chaining)."""

    def __init__(self, val: int = 0):
        self.val = val

    def add(self, amount: int) -> DummyChain:
        self.val += amount
        return self

    def multiply(self, factor: int) -> DummyChain:
        self.val *= factor
        return self


class DummyChainCommand(Command):
    """Comando simples para rastrear mutações fluentes do DummyChain."""

    def __init__(self, name: str, item: Any, value: Any = None):
        super().__init__(name, item, value)
        target = getattr(item, "_target", item)
        self._target = target
        self._old_val = target.val
        self._new_val = target.val

    def seal(self) -> None:
        self._new_val = self._target.val
        self._sealed = True

    def execute(self) -> None:
        self._target.val = self._new_val

    def undo(self) -> None:
        if not self._sealed:
            self.seal()
        self._target.val = self._old_val

    def has_changes(self) -> bool:
        return self._old_val != self._new_val


class DummyFluentProxy(BaseFluentProxy):
    """Proxy fluente especialista para DummyChain."""

    _MUTATING_METHODS = frozenset({"add", "multiply"})
    _COMMAND_CLASS = DummyChainCommand
    _COMMAND_NAME = "chain_op"


class DummyService:
    """Classe dummy de serviço/estratégia com métodos públicos."""

    def __init__(self, entity: DummyEntity):
        self.entity = entity

    def double_value(self) -> None:
        self.entity.value = self.entity.value * 2

    def complex_operation(self, sub_service: DummyService) -> None:
        self.double_value()
        sub_service.double_value()
        self.entity.name = "Processed"

    def failing_operation(self) -> None:
        self.entity.value = 999
        raise RuntimeError("Operation aborted")


@pytest.fixture
def history():
    return GlobalHistory()


def test_base_history_proxy_tracks_property_setter(history):
    """Valida se o BaseHistoryProxy intercepta @property com setter e suporta Undo e Redo."""
    entity = DummyEntity(name="Initial")
    proxy = BaseHistoryProxy(entity, history)

    proxy.name = "Updated"

    assert entity.name == "Updated"
    assert len(history._undo_stack) == 1

    history.undo()
    assert entity.name == "Initial"

    history.redo()
    assert entity.name == "Updated"


def test_base_history_proxy_ignored_attributes_bypass_history(history):
    """Valida se atributos declarados em _IGNORED_ATTRIBUTES não geram comandos no histórico."""
    entity = DummyEntity()
    proxy = DummyCustomProxy(entity, history)

    proxy.untracked = "modified"

    assert entity.untracked == "modified"
    assert history.undo_empty()


def test_base_history_proxy_noop_does_not_pollute_history(history):
    """Valida se atribuir o mesmo valor não gera comando de alteração no histórico."""
    entity = DummyEntity(name="Same")
    proxy = BaseHistoryProxy(entity, history)

    proxy.name = "Same"

    assert history.undo_empty()


def test_base_fluent_proxy_single_line_chaining_creates_single_command(history):
    """Valida se múltiplas chamadas encadeadas na mesma linha geram exatamente 1 comando no histórico."""
    chain = DummyChain(val=10)
    proxy = DummyFluentProxy(chain, history)

    proxy.add(5).multiply(2)
    del proxy

    assert chain.val == 30
    assert len(history._undo_stack) == 1

    history.undo()
    assert chain.val == 10

    history.redo()
    assert chain.val == 30


def test_base_fluent_proxy_passive_read_does_not_touch_history(history):
    """Valida se inspecionar atributos do objeto fluente não dispara ações no histórico."""
    chain = DummyChain(val=42)
    proxy = DummyFluentProxy(chain, history)

    _ = proxy.val
    del proxy

    assert history.undo_empty()


def test_strategy_proxy_wraps_public_methods_in_single_atomic_command(history):
    """Valida se chamadas de serviço via StrategyProxy geram uma unidade atômica no histórico."""
    entity = DummyEntity(value=5)
    proxy_entity = BaseHistoryProxy(entity, history)
    service = DummyService(proxy_entity)
    proxy_service = StrategyProxy(service, history)

    proxy_service.double_value()

    assert entity.value == 10
    assert len(history._undo_stack) == 1

    history.undo()
    assert entity.value == 5


def test_strategy_proxy_nested_calls_accumulate_into_single_macro(history):
    """Valida se serviços aninhados são absorvidos no mesmo macro atômico sem commit prematuro."""
    e1 = DummyEntity(name="E1", value=10)
    e2 = DummyEntity(name="E2", value=20)
    p_e1 = BaseHistoryProxy(e1, history)
    p_e2 = BaseHistoryProxy(e2, history)

    s1 = DummyService(p_e1)
    s2 = DummyService(p_e2)
    p_s1 = StrategyProxy(s1, history)
    p_s2 = StrategyProxy(s2, history)

    p_s1.complex_operation(p_s2)

    assert e1.value == 20
    assert e2.value == 40
    assert e1.name == "Processed"
    assert len(history._undo_stack) == 1

    history.undo()
    assert e1.value == 10
    assert e2.value == 20
    assert e1.name == "E1"


def test_strategy_proxy_rollback_on_failure(history):
    """Valida se uma falha no método de serviço desfaz mutações parciais e descarta o comando."""
    entity = DummyEntity(value=100)
    proxy_entity = BaseHistoryProxy(entity, history)
    service = DummyService(proxy_entity)
    proxy_service = StrategyProxy(service, history)

    with pytest.raises(RuntimeError, match="Operation aborted"):
        proxy_service.failing_operation()

    assert entity.value == 100
    assert history.undo_empty()


def test_proxy_registry_identity_map_reuses_instances(history):
    """Valida se o ProxyRegistry retorna a mesma instância de proxy para o mesmo target."""
    registry = ProxyRegistry(history)
    entity = DummyEntity()

    p1 = registry.get_or_create(entity)
    p2 = registry.get_or_create(entity)

    assert p1 is p2
    assert not hasattr(entity, "_proxy")


def test_strategy_proxy_custom_history_context_hook(history):
    """Valida se uma subclasse de StrategyProxy pode mudar o contexto de histórico via hook."""
    class SilentStrategyProxy(StrategyProxy):
        def _history_context(self, action_name: str) -> Any:
            return self._history.disabled()

    entity = DummyEntity(value=42)
    proxy_entity = BaseHistoryProxy(entity, history)
    service = DummyService(proxy_entity)
    silent_proxy = SilentStrategyProxy(service, history)

    silent_proxy.double_value()

    assert entity.value == 84
    assert history.undo_empty()
