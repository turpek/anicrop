from __future__ import annotations

import weakref
from typing import Any

import numpy as np


def unwrap_target(obj: Any) -> Any:
    """Extrai o objeto real de domínio se obj for um Proxy."""
    return getattr(obj, "_target", obj)


def unwrap_call_args(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Desempacota proxies de argumentos posicionais e nomeados para chamadas no domínio."""
    clean_args = tuple(unwrap_target(a) for a in args)
    clean_kwargs = {k: unwrap_target(v) for k, v in kwargs.items()}
    return clean_args, clean_kwargs


def is_property_with_setter(cls: type, name: str) -> bool:
    """Retorna True se o atributo na classe cls for uma property com setter."""
    class_attr = getattr(cls, name, None)
    return isinstance(class_attr, property) and class_attr.fset is not None


def is_readonly_property(cls: type, name: str) -> bool:
    """Retorna True se o atributo na classe cls for uma property sem setter (read-only)."""
    class_attr = getattr(cls, name, None)
    return isinstance(class_attr, property) and class_attr.fset is None


class ProxyRegistry:
    """Identity Map usando WeakValueDictionary para garantir instância única de Proxy por target."""

    _TYPE_REGISTRY: list[tuple[type, type]] = []
    _DEFAULT_PROXY_CLS: type | None = None

    def __init__(self, history: Any):
        self._history = history
        self._cache: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()

    @classmethod
    def register(cls, domain_cls: type, proxy_cls: type) -> None:
        """Registra uma associação entre classe de domínio e classe de proxy."""
        cls._TYPE_REGISTRY.append((domain_cls, proxy_cls))

    @classmethod
    def set_default_proxy(cls, proxy_cls: type) -> None:
        """Define a classe padrão de proxy para objetos não mapeados explicitamente."""
        cls._DEFAULT_PROXY_CLS = proxy_cls

    def get_or_create(self, target: Any) -> Any:
        """Retorna o proxy correspondente para o target ou o cria via Identity Map."""
        if target is None:
            return None

        # Se já for um proxy ou se for NullContainer neutro
        if hasattr(target, "_target") or type(target).__name__ == "NullContainer":
            return target

        target_id = id(target)
        if target_id in self._cache:
            return self._cache[target_id]

        # Busca na lista de registros
        for domain_type, proxy_type in self._TYPE_REGISTRY:
            if isinstance(target, domain_type):
                return proxy_type(target, self._history, registry=self)

        if self._DEFAULT_PROXY_CLS is not None:
            return self._DEFAULT_PROXY_CLS(target, self._history, registry=self)

        return target


def wrap_domain_result(
    result: Any, history: Any, registry: ProxyRegistry
) -> Any:
    """Empacota o resultado do domínio no Proxy correspondente via Identity Map."""
    if result is None or isinstance(
        result, (int, float, str, bool, bytes, np.ndarray, tuple, list, set, dict)
    ):
        return result
    return registry.get_or_create(result)


def get_registry_for_history(history: Any) -> ProxyRegistry:
    """Recupera ou instancia o ProxyRegistry vinculado à instância de GlobalHistory."""
    reg = getattr(history, "_proxy_registry", None)
    if isinstance(reg, ProxyRegistry):
        return reg
    reg = ProxyRegistry(history)
    try:
        history._proxy_registry = reg
    except Exception:
        pass
    return reg
