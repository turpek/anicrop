# Guia do Sistema de Histórico (`anicrop.history`)

O módulo `anicrop.history` implementa o motor central de histórico (Undo/Redo) da biblioteca `anicrop`, utilizando os padrões de projeto **Command**, **Memento** e **Strategy**. Ele permite desfazer e refazer operações geométricas, ajustes de parâmetros escalares, mutações na árvore hierárquica e edições locais de pixels com precisão atômica e consumo mínimo de memória.

---

## 1. Visão Geral e Arquitetura

O histórico é gerenciado pela classe `GlobalHistory`. Ele desacopla a intenção de modificação da sua aplicação direta, permitindo registrar mutações que podem ser desfeitas e refeitas a qualquer momento.

### Princípios de Design:
- **Duas Pilhas Reativas**: Mantém internamente duas pilhas (`_undo_stack` e `_redo_stack`) gerenciadas com `collections.deque`.
- **Limpeza Automática de Redo**: Sempre que uma nova ação for executada fora do fluxo de redo, a pilha de redo é imediatamente esvaziada (`_clear_redo()`), mantendo a linha do tempo estritamente determinística e linear.
- **Transações Atômicas de 1 Passo**: Operações compostas de alto nível (como `crop`, `fit` ou `resize_bounds`) agrupam todos os seus sub-comandos internos em um único comando composto (`MacroCommand`), permitindo reverter tudo com **exatamente 1 chamada de `undo()`**.
- **Desacoplamento do Domínio**: As classes puras de domínio (`Layer`, `GroupLayer`, `Canvas`, `Mask`) não conhecem o histórico. O rastreamento de ações é feito de forma não-invasiva através de proxies reativos (`ProxyLayer`, `GroupProxy`, `ProxyCanvas`, `ProxyMask`).

---

## 2. Políticas de Histórico (`ActionPolicy`)

O comportamento de registro e agrupamento de ações é governado dinamicamente por instâncias de `ActionPolicy`. Isso permite que o histórico altere seu modo de operação em tempo de execução sem acoplamento rígido.

| Política | Classe | Comportamento Principal | Cenário Típico de Uso |
| :--- | :--- | :--- | :--- |
| **Normal** | `NormalPolicy` | Registra cada comando individualmente na pilha. Sela o comando anterior caso não suporte fusão contínua (`can_merge`). | Atribuições diretas de propriedades (ex: `layer.opacity = 0.5`, `layer.name = "Novo"`). |
| **Atômica** | `AtomicPolicy` | Absorve todos os sub-comandos gerados internamente em um único `MacroCommand`. Reverte tudo em 1 único Undo. Possui rollback automático em caso de exceção. | Operações de alto nível do motor (ex: `layer.layout.align()`, `layer.content.crop()`, `doc.layout.resize_bounds()`). |
| **Fusão Contínua** | `MergeContinuousPolicy` | Agrupa mutações consecutivas com o mesmo nome e mesmo alvo no mesmo comando aberto, atualizando o delta sem criar novos passos. | Arrastar controles em uma GUI, sliders de opacidade contínua, pan/zoom interativo. |
| **Agrupamento de Ação** | `GroupActionPolicy` | Ignora o início de novas ações se o topo da pilha de undo pertencer à mesma classe de comando (`type(last_cmd) is command_cls`). | Agrupamento de comandos semelhantes em lote. |
| **Desativada / Silenciosa** | `DisabledPolicy` | Ignora silenciosamente qualquer chamada a `start_action` e `commit`. Nenhuma ação é gravada. | Renderização de cena, cálculos de bounding box e execução interna de `undo()` / `redo()`. |

---

## 3. Controle de Profundidade e Context Managers

Para permitir que funções chamem outras funções sem corromper o estado das políticas, o `GlobalHistory` implementa **controle reentrante de profundidade** via pilha de políticas:

```python
with history.use_policy(AtomicPolicy()):
    # Política ativa é AtomicPolicy
    with history.use_policy(DisabledPolicy()):
        # Temporariamente desativada
        pass
    # Restaura com segurança para AtomicPolicy
# Ao sair da raiz (_policy_depth == 0), executa history.commit()
```

### Context Managers Utilitários da Classe `GlobalHistory`:

#### 1. `with history.atomic(name: str = "Atomic"):`
Garante que todas as mutações que ocorrerem no interior do bloco sejam seladas juntas em um único `MacroCommand`. Se ocorrer qualquer erro não tratado dentro do bloco, o histórico desfaz automaticamente os sub-comandos já aplicados e descarta o macro-comando (auto-rollback transacional).

```python
with doc.history.atomic("ComplexSetup"):
    doc.canvas.region = Region.from_size(1920, 1080)
    layer.opacity = 0.8
    layer.layout.align(doc.canvas.region, 0.5, 0.5)

# Apenas 1 undo reverte todas as 3 operações acima juntas:
doc.history.undo()
```

#### 2. `with history.merge_continuous():`
Ativa a `MergeContinuousPolicy`. Útil para mesclar alterações contínuas em um único estado inicial e final.

```python
with doc.history.merge_continuous():
    for opacity in [0.9, 0.8, 0.7, 0.6, 0.5]:
        layer.opacity = opacity

# 1 undo restaura a opacidade original diretamente:
doc.history.undo()
```

#### 3. `with history.disabled():`
Desativa a gravação temporariamente. Todas as mutações aplicadas no bloco ocorrem no domínio, mas não deixam rastros na pilha de undo.

```python
with doc.history.disabled():
    # Modificação silenciosa/volátil
    layer.visible = False
```

#### 4. `with history.transaction(name: str | None = None):`
Contexto polimórfico: se `name` for fornecido, delega para `history.atomic(name)`; caso contrário, executa sob `NormalPolicy`.

---

## 4. Como Usar na Prática

### 4.1. Uso Integrado através do `Document` (Recomendado)

Ao instanciar um `Document` com o parâmetro `history=True`, o documento ativa a política reativa (`ReactiveDocumentPolicy`), instanciando um `GlobalHistory` e empacotando automaticamente o Canvas e todas as camadas adicionadas em proxies correspondentes:

```python
from anicrop.document import Document
from anicrop.layer import Layer
from anicrop.image import Image
from anicrop.enums import ImageFormat
from anicrop.spatial import Region
import numpy as np

# 1. Cria o documento com histórico ativado
doc = Document(name="MeuProjeto", width=1000, height=1000, history=True)

# 2. Adiciona camada (retorna automaticamente um ProxyLayer)
img = Image(np.full((200, 200, 4), 255, dtype=np.uint8), ImageFormat.RGBA)
layer = doc.add(Layer(img, name="Camada1"))

# 3. Realiza mutações rastreadas
layer.opacity = 0.5
layer.layout.align(doc.canvas.region, anchor_x=1.0, anchor_y=1.0)
doc.canvas.layout.resize_bounds(1200, 800)

# 4. Desfaz os passos individualmente
doc.history.undo()  # Desfaz o resize do Canvas
doc.history.undo()  # Desfaz o alinhamento da camada
doc.history.undo()  # Desfaz a alteração de opacidade (retorna a 1.0)

# 5. Refaz os passos
doc.history.redo()  # Opacidade volta para 0.5
doc.history.redo()  # Alinhamento é reaplicado
```

### 4.2. Consulta de Estado do Histórico

A classe `GlobalHistory` expõe métodos expressivos para consultar as pilhas:

```python
# Verifica se há ações para desfazer ou refazer
if not doc.history.undo_empty():
    doc.history.undo()

if not doc.history.redo_empty():
    doc.history.redo()

# Verifica se o histórico está ativo para gravação no momento
if doc.history.is_active:
    print("Histórico gravando ações normalmente.")

# Inspeciona a quantidade de passos na pilha de undo
total_passos = len(doc.history._undo_stack)
```

### 4.3. Uso Avulso / Standalone de `GlobalHistory`

Você também pode utilizar o `GlobalHistory` diretamente com contêineres e camadas avulsas sem a fachada `Document`:

```python
from anicrop.history import GlobalHistory
from anicrop.reactive import ProxyRegistry, ProxyLayer
from anicrop.layer import Layer
from anicrop.image import Image
from anicrop.enums import ImageFormat
import numpy as np

# Cria histórico e registry
history = GlobalHistory()
registry = ProxyRegistry(history)

# Envolve a camada no ProxyLayer
raw_layer = Layer(Image(np.zeros((100, 100, 4), dtype=np.uint8), ImageFormat.RGBA))
proxy_layer = registry.get_or_create(raw_layer)

# Operações são rastreadas
proxy_layer.opacity = 0.3
assert raw_layer.opacity == 0.3

history.undo()
assert raw_layer.opacity == 1.0
```

---

## 5. Resumo das Boas Práticas

1. **Nunca modifique o `_target` diretamente** se desejar rastrear histórico; aplique as alterações sempre no objeto Proxy (`doc.canvas`, `layer`, `group`).
2. **Use `with history.atomic(...)`** sempre que orquestrar uma sequência lógica de operações que o usuário enxerga como uma única ação na interface.
3. **Use `with history.disabled()`** em rotinas internas de cálculo, pré-visualização ou carregamento inicial de templates para evitar passos desnecessários na pilha.
4. **Verifique `undo_empty()` / `redo_empty()`** antes de disparar chamadas de Undo/Redo na interface para manter botões habilitados/desabilitados de forma reativa.
