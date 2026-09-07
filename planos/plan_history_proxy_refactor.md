# Guia Definitivo de Refatoração: Histórico & Proxies no anicrop

> **Status:** Fase 1 Concluída (Commit `7697b1d`) | 1059 testes passando  
> **Branch Atual:** `refactor/history-proxy`  
> **Arquivo Local de Handoff:** Este documento NÃO deve ser comitado no Git (deixe como untracked para guiar a continuação em uma sessão limpa).

---

## 1. O Que NÃO Fazer (Linhas Vermelhas e Armadilhas Evitadas)

Durante a discussão técnica e o diagnóstico arquitetural, delimitamos princípios estritos sobre **o que NÃO fazer**:

1. **NUNCA violar a Pureza do Domínio (Regra de Ouro):**
   * **PROIBIDO:** Colocar qualquer menção a `history`, `_history`, `Proxy` ou `isinstance(..., Proxy)` dentro dos módulos de domínio ou serviços (`Layout`, `Content`, `Layer`, `Container`, `Mask`, `Canvas`, etc.).
   * **CORRETO:** O domínio deve ser 100% cego e agnóstico à infraestrutura. Toda e qualquer orquestração de histórico, transações ou envelopamento deve residir **exclusivamente dentro do módulo `proxy.py`**.
2. **NUNCA desmontar `ReparentCommand` nem `ContainerSnapshot`:**
   * **PROIBIDO:** Tentar simplificar ou refatorar o `ReparentCommand` e as classes `ContainerSnapshot` / `NodeContainerSnapshot` / `NullContainer`.
   * **MOTIVO:** Essa estrutura relacional foi desenvolvida para resolver bugs sutis e críticos de prevenção de ciclos na árvore (`GroupLayer._check_ancestor`) e preservação exata da matriz espacial inversa (`_parent_inverse`). Ela já está 100% coberta e estabilizada por testes. Deixe-a intacta.
3. **NUNCA quebrar o encadeamento de `layer.transform` em micro-comandos:**
   * **PROIBIDO:** Criar micro-comandos separados para `rotate`, `scale`, `translate` dentro de uma cadeia fluente.
   * **MOTIVO:** A sintaxe `layer.transform.scale(2, 2).translate(10, 0).rotate(45)` na mesma linha DEVE gerar **exatamente 1 comando no histórico**. Quebrar isso exigiria merges caóticos de comandos heterogêneos. Chamadas em linhas separadas devem gerar comandos separados.
4. **NUNCA limpar o `_redo_stack` prematuramente em `start_action`:**
   * **PROIBIDO:** Chamar `history._clear_redo()` logo na abertura de uma ação.
   * **MOTIVO:** Se a ação for uma leitura passiva ou uma atribuição sem alteração real (`layer.opacity = layer.opacity`), o comando será descartado no commit, mas o redo stack teria sido destruído para sempre. O redo só deve ser limpo quando uma ação com delta real (`has_changes() is True`) for efetivamente confirmada.
5. **NUNCA usar lambdas na tupla de deltas do `AdaptiveCommand`:**
   * **PROIBIDO:** Armazenar `(old, new, lambda target, v: setattr(...))` em cada propriedade.
   * **MOTIVO:** É *over-engineering* e desperdício de memória criando closures efêmeras. Como todas as properties do `BaseLayer` possuem `@setter` nativo, basta armazenar `(old_value, new_value)` e usar `setattr(self._target, prop_name, val)` diretamente no `undo()` e `execute()`.
6. **NUNCA deixar o sistema em estado aberto/pendente ("dangling") para operações normais:**
   * **PROIBIDO:** Deixar `layer.opacity = 0.5` ou `group.append(layer)` abertos na pilha esperando a próxima linha de código.
   * **CORRETO:** 95% das operações do sistema são discretas e começam e terminam no mesmo turno. Elas abrem, executam e **selam imediatamente**. Apenas a cadeia fluente de `transform` usa o ciclo do `ProxyComposer`.

---

## 2. Anatomia do Proxy Atual (Como Funciona e Onde Está a Bagunça)

No código atual de [`src/anicrop/proxy.py`](file:///home/gui/python/anicrop/src/anicrop/proxy.py):

1. **Monolito de Introspecção (`BaseHistoryProxy.__getattribute__`):**
   * Possui mais de 100 linhas cheias de condicionais em cascata (`if name == "parent"`, `if name in chainable`, `if callable and name in action_router`, etc.).
   * Existe um bloco idêntico de código de ~30 linhas **duplicado** em dois lugares: dentro do closure `method_wrapper` e no retorno de atributos passivos (linhas 211-238 e 243-266).
   * Lista hardcoded de 8 classes de domínio sendo verificadas no corpo do proxy.
2. **O Hack de Leitura em `.transform` (`_CHAINABLE_PROPERTIES`):**
   * Como `proxy.transform` não tinha um proxy próprio (retornava o `ComposerRel` cru), o autor colocou `_CHAINABLE_PROPERTIES = ("transform",)`.
   * Sempre que alguém lia `proxy.transform`, o proxy chamava `history.start_action(...)` imediatamente.
   * Consequência: inspecionar `_ = layer.transform.matrix` iniciava um comando no histórico e apagava o redo stack!
3. **Acoplamento Invertido em `command.py` (`_create_snapshot`):**
   * Em `command.py:28`: `elif not isinstance(obj, NullContainer): raise TypeError("Expected a Proxy or neutral object")`.
   * Os comandos recusavam objetos puros de domínio, forçando que fossem Proxies.
4. **Comandos sem Selamento Imediato:**
   * No `__setattr__`, o proxy chamava `start_action`, depois `setattr(target, name, value)` e não chamava `commit()`.
   * O comando ficava eternamente aberto na pilha de undo com `_sealed = False`, esperando a próxima ação ou um `history.undo()` forçar o selamento.

---

## 3. Como o Sistema DEVERÁ Funcionar (A Nova Arquitetura)

### 3.1. Funções Auxiliares Puras (Modularidade Máxima)
Em vez de condicionais inline e closures duplicadas, extrair funções auxiliares no topo de `proxy.py`:

```python
def unwrap_target(obj: Any) -> Any:
    """Extrai o objeto real de domínio se obj for um Proxy."""
    return getattr(obj, "_target", obj)

def unwrap_call_args(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Desempacota proxies de argumentos posicionais e nomeados para chamadas no domínio."""
    clean_args = tuple(unwrap_target(a) for a in args)
    clean_kwargs = {k: unwrap_target(v) for k, v in kwargs.items()}
    return clean_args, clean_kwargs

def wrap_domain_result(result: Any, history: GlobalHistory, registry: ProxyRegistry) -> Any:
    """Empacota o resultado do domínio no Proxy correspondente via Identity Map."""
    if result is None or isinstance(result, (int, float, str, bool, tuple, np.ndarray)):
        return result
    return registry.get_or_create(result)

def execute_silent_mutation(history: GlobalHistory, func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Executa a mutação no objeto real sob history.disabled() para evitar loops."""
    with history.disabled():
        return func(*args, **kwargs)
```

### 3.2. As 4 Classes de Proxy Especializadas

```
                     ┌───────────────────────────────┐
                     │       BaseHistoryProxy        │
                     │        (Proxy Geral)          │
                     │  - Intercepta __setattr__     │
                     │  - Leitura passiva limpa      │
                     └───────────────┬───────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  StrategyProxy   │        │  ProxyComposer   │        │    ProxyMask     │
│ (layout/content) │        │   (transform)    │        │     (pixels)     │
│- with transação  │        │- Fluent chain    │        │- __setitem__     │
│  automática      │        │- __del__ seal    │        │- Micro-snapshot  │
└──────────────────┘        └──────────────────┘        └──────────────────┘
         │
         └───────────────────────────┬───────────────────────────┐
                                     ▼                           ▼
                            ┌──────────────────┐        ┌──────────────────┐
                            │    ProxyLayer    │        │BaseContainerProxy│
                            │ - add_edit       │        │ - append/remove  │
                            │ - mask link      │        │ - ReparentCommand│
                            └────────┬─────────┘        └────────┬─────────┘
                                     │                           │
                                     └─────────────┬─────────────┘
                                                   ▼
                                          ┌──────────────────┐
                                          │    GroupProxy    │
                                          │(Layer+Container) │
                                          └──────────────────┘
```

1. **`BaseHistoryProxy` (Geral):**
   * `__setattr__`: se `name in _ACTION_ROUTER`, chama `history.start_action(AdaptiveCommand, name, self, value)`, aplica via `setattr` sob `disabled` e **chama `history.commit()` imediatamente** (selando no ato).
   * `__getattribute__`: leitura passiva pura sem histórico. Se o retorno for entidade, passa por `wrap_domain_result`.
2. **`StrategyProxy` (para `layer.layout` e `layer.content`):**
   * Envolve métodos públicos em `with history.transaction(name):`.
   * Quando `doc.layout.align(...)` ou `doc.content.crop(...)` roda:
     * O `StrategyProxy` abre a transação reentrante.
     * Mutações internas (`region`, `frame`, `add_edit`) são absorvidas na mesma transação.
     * Ao encerrar o método, consolida tudo em **1 único passo de Undo**.
     * O motor `Layout` e `Content` não sabe que histórico existe!
3. **`ProxyComposer` (para `layer.transform` - Padrão C):**
   * Envolve `layer.transform` (`ComposerRel`).
   * `.matrix`, `.size`, `.region`: leitura passiva pura.
   * `.rotate()`, `.scale()`, `.translate()`:
     * Abrem o comando na primeira chamada mutadora.
     * Executam no `ComposerRel` real.
     * Retornam `self` (o próprio `ProxyComposer`).
   * `__del__`: se tiver comando aberto, sela.
4. **`ProxyMask`:**
   * Intercepta `__setitem__` (`mask[10:20, 10:20] = 0`) para gerar `MaskCommand` com micro-snapshot do slice.
5. **`BaseContainerProxy` / `GroupProxy`:**
   * Métodos `append`, `insert`, `remove`, `move`, `pop` continuam gerando `ReparentCommand` com integridade de `_parent_inverse`.

---

## 4. O Que Já Foi Feito (Fase 1 - Concluída)

* **Commit:** `7697b1d` na branch `refactor/history-proxy`.
* **Arquivo alterado:** [`src/anicrop/command.py`](file:///home/gui/python/anicrop/src/anicrop/command.py)
  * Implementado `AdaptiveCommand` com `_deltas: dict[str, tuple[Any, Any]]` em $O(1)$ sem lambdas.
  * `BaseLayerCommand` herda de `AdaptiveCommand`. Suporta deltas adaptativos e fallback para snapshots antigos.
  * Suporte a arrays NumPy com `np.copy` em `record_change`, `undo` e `execute`.
* **Arquivo alterado:** [`tests/test_command.py`](file:///home/gui/python/anicrop/tests/test_command.py)
  * 5 novos testes unitários cobrindo mutação individual, acúmulo de propriedades distintas, merge contínuo (slider), no-change e arrays NumPy.
* **Testes:** 1059/1059 testes passando no `pytest`.

---

## 5. Roteiro Passo a Passo para as Próximas Fases

### Fase 2: `src/anicrop/history.py` (Concluída - 1063 testes passando)
1. **Redo Seguro:**
   * Em `NormalPolicy.commit(history)`:
     * Se `last_cmd.has_changes() is True`: executa `history._clear_redo()`.
     * Se `last_cmd.has_changes() is False`: dá pop no comando sem limpar o redo stack.
   * Removido `history._clear_redo()` prematuro de `start_action`.
   * `redo_empty()` agora chama `self.commit()` antes de inspecionar a pilha de redo.
2. **Controle de Profundidade em `use_policy` e Atomic Seguro com Rollback:**
   * `GlobalHistory._policy_depth`: centraliza no `use_policy` a regra de ouro: apenas o contexto raiz (`_policy_depth == 0`) executa `self.commit()`.
   * Chamadas internas subordinadas (ex: `content` chamando `layout` ou mutações sob `disabled()`) não trocam o macro pai nem comitam antes da hora.
   * `atomic`: se `already_atomic`, não cria macros redundantes. Em caso de exceção na raiz, desfaz as ações registradas e descarta o macro da pilha.
3. **Testes em `tests/test_history.py`:**
   * `test_history_safe_redo_preserves_stack_on_no_change_action`: preservação de redo após leitura/no-change.
   * `test_history_atomic_nested_accumulates_into_single_macro`: reentrância agregando no macro raiz.
   * `test_history_atomic_rollback_on_exception`: rollback automático e descarte do macro.
   * `test_history_disabled_within_atomic_does_not_commit_prematurely`: `disabled()` interno sem commit prematuro.

### Fase 3: `src/anicrop/proxy.py` (Proxies Limpos e Especialistas)
1. Inserir as 4 funções auxiliares (`unwrap_target`, `unwrap_call_args`, etc.).
2. Implementar `StrategyProxy` e usá-lo para `layout` e `content`.
3. Implementar `ProxyComposer` para `transform`.
4. Refatorar `BaseHistoryProxy.__setattr__` para gravar via `AdaptiveCommand` e selar imediatamente.
5. Limpar duplicações de `__getattribute__`.
6. Validar com:
   * `uv run pytest tests/test_proxy.py`
   * `uv run pytest tests/integration/test_layout_history.py`
   * `uv run pytest tests/integration/test_content_history.py`

### Fase 4: Validação Global e Qualidade
1. Rodar suíte completa: `uv run pytest`.
2. Rodar linter e formatador estrito:
   ```bash
   uv run ruff check src/ tests/ --fix && uv run autopep8 --in-place --recursive --max-line-length 89 --ignore E501,E402,W503,W504 src/ tests/
   ```
