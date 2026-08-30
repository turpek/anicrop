# Guia de Contêineres de Camadas e Hierarquia (`container.py`)

O módulo `anicrop.container` define a estrutura hierárquica e a árvore espacial de elementos do documento. Utiliza o padrão de projeto **Composite**, permitindo que contêineres e camadas individuais sejam manipulados através de interfaces unificadas.

---

## 1. Protocolo de Nó Espacial: `NodeContainerProtocol`

O `NodeContainerProtocol` é o protocolo formal para qualquer elemento da árvore espacial/hierárquica do `anicrop`.

### Requisitos do Protocolo:
- **`parent` (`Container`)**: Atributo apontando para o contêiner pai. Nos objetos raiz ou recém-criados, é inicializado apontando para `_NULL_CONTAINER`.
- **`_parent_inverse` (`np.ndarray`)**: Matriz 3x3 com a inversa da transformação do pai, garantindo a recomposição exata de coordenadas relativas na hierarquia.

---

## 2. Abstração Base: `BaseLayer` (`anicrop.container.BaseLayer`)

A `BaseLayer` é a classe abstrata herdada por **`Layer`** e **`GroupLayer`**. Ela estende o comportamento de ambos para oferecer suporte completo a transformações matriciais 2D e propriedades visuais.

### Principais Atributos e Métodos de `BaseLayer`:
- **`opacity` (`float`)**: Opacidade global do elemento (0.0 a 1.0).
- **`blend_mode` (`BlendMode`)**: Modo de mesclagem (NORMAL, MULTIPLY, SCREEN, etc.).
- **`visible` (`bool`)**: Flag de visibilidade do elemento no pipeline de renderização.
- **`transform` (`Composer`)**: Compositor mutável de transformações matriciais acumuladas (instanciado automaticamente no `__init__`).
- **`layout` (`GeometryStrategy`)**: Setter e getter da estratégia de disposição espacial (utilizado internamente pelo módulo `Layout`).
- **`set_transform(transform: Transform, reference: Optional[Canvas | Layer] = None) -> None`**: Aplica um objeto de transformação (`Transform`) substituindo/ajustando o compositor local.
- **`transform_clear() -> None`**: Limpa as transformações acumuladas, retornando à matriz identidade.
- **`_parent_inverse` (`np.ndarray`)**: Matriz inversa da transformação do contêiner pai (`mat_inverse(parent.matrix)`).

---

## 3. `LayerStack` (`anicrop.container.LayerStack`)

A `LayerStack` é o contêiner raiz (*Root Container*) da árvore de elementos de um `Document`.

### Características Principais
- Herda da classe base `Container`, oferecendo comportamento de lista ordenada de camadas (do fundo para o topo: índice `0` é a base/fundo e o último elemento ou `append` é o topo visível).
- É um objeto de nível raiz e **não pode** ser adicionado como filho de nenhum outro contêiner ou grupo (tentativas lançam `TypeError`).
- Retorna uma matriz de transformação identidade (`identity(3)`) como sua matriz base.

---

## 4. `GroupLayer` (`anicrop.container.GroupLayer`)

A classe `GroupLayer` representa um grupo de camadas. Ela herda simultaneamente de **`Container`** (podendo ter elementos filhos) e de **`BaseLayer`** (possuindo opacidade, modo de mesclagem, visibilidade e transformações matriciais próprias).

Ao aplicar transformações ou alterar a opacidade de um `GroupLayer`, todos os seus elementos filhos (sejam `Layer`s ou sub-`GroupLayer`s) herdam e combinam esse estado dinamicamente durante a renderização.

### Uso Direto das Transformações no Grupo:
Como qualquer sub-classe de `BaseLayer`, o `GroupLayer` já vem com a propriedade `transform` pronta para uso:

```python
group = doc.add_group(name="Acessorios")
group.append(layer1)
group.append(layer2)

# Transforma todas as camadas do grupo simultaneamente:
group.transform.rotate(30).scale(1.2, 1.2).translate(50, 0)
```

---

### Principais Métodos e Propriedades de `GroupLayer` e `Container`

#### `__init__(opacity: float = 1.0, blend_mode: BlendMode = BlendMode.NORMAL, name: str = 'BaseLayer')`
- **Descrição**: Inicializa um novo grupo de camadas vazio com opacidade e modo de mesclagem configuráveis. Herda de `Container` e `BaseLayer`, definindo `parent = _NULL_CONTAINER`.
- **Parâmetros**:
  - `opacity` (`float`): Opacidade global do grupo (0.0 a 1.0).
  - `blend_mode` (`BlendMode`): Modo de mesclagem do grupo com a cena.
  - `name` (`str`): Nome identificador do grupo.
- **Retorno**: Instância de `GroupLayer`.

#### `append(item: GroupLayer | Layer) -> None`
- **Descrição**: Adiciona um `Layer` ou outro `GroupLayer` como filho ao final do grupo. Remove automaticamente o item do seu contêiner pai anterior, atualiza o atributo `item.parent = self` e recalcula `item._parent_inverse = mat_inverse(self.matrix)`. Valida para evitar inclusões cíclicas.
- **Parâmetros**:
  - `item` (`GroupLayer | Layer`): O elemento a ser inserido no grupo.
- **Lança**: `ValueError` se o item já estiver no grupo ou se for um ancestral.
- **Retorno**: `None`.

#### `insert(index: int, item: GroupLayer | Layer) -> None`
- **Descrição**: Insere um elemento filho em um índice específico da pilha do grupo.
- **Parâmetros**:
  - `index` (`int`): Índice de destino.
  - `item` (`GroupLayer | Layer`): Elemento a inserir.
- **Retorno**: `None`.

#### `remove(item: GroupLayer | Layer) -> None`
- **Descrição**: Remove o elemento filho do grupo, redefinindo `item.parent = _NULL_CONTAINER` e recarregando `item._parent_inverse`.
- **Parâmetros**:
  - `item` (`GroupLayer | Layer`): Elemento a ser removido.
- **Retorno**: `None`.

#### `move(item: BaseLayer, new_index: int) -> None`
- **Descrição**: Altera a ordem de empilhamento (*z-order*) de um elemento filho existente dentro do contêiner para um índice absoluto sem desanexá-lo.
- **Parâmetros**:
  - `item` (`BaseLayer`): Elemento a ser reordenado.
  - `new_index` (`int`): Novo índice de posição.
- **Retorno**: `None`.

#### `move_relative(item: BaseLayer, steps: int) -> None`
- **Descrição**: Desloca a posição de um filho relativamente na pilha (passos positivos movem em direção ao topo/índices maiores, passos negativos movem em direção à base/índices menores) com limitação segura (*clamping*) nos limites `[0, len - 1]`.
- **Parâmetros**:
  - `item` (`BaseLayer`): Elemento a ser movido.
  - `steps` (`int`): Quantidade de posições relativas a avançar ou recuar.
- **Retorno**: `None`.

#### `move_to_front(item: BaseLayer) -> None`
- **Descrição**: Move o elemento diretamente para o topo absoluto da pilha do contêiner (`len - 1`).
- **Parâmetros**:
  - `item` (`BaseLayer`): Elemento a ser enviado para o topo.
- **Retorno**: `None`.

#### `move_to_back(item: BaseLayer) -> None`
- **Descrição**: Move o elemento diretamente para a base absoluta da pilha do contêiner (índice `0`).
- **Parâmetros**:
  - `item` (`BaseLayer`): Elemento a ser enviado para o fundo.
- **Retorno**: `None`.

#### `swap(item_a: BaseLayer, item_b: BaseLayer) -> None`
- **Descrição**: Troca diretamente a ordem de empilhamento de dois elementos contidos no mesmo contêiner.
- **Parâmetros**:
  - `item_a` (`BaseLayer`): Primeiro elemento.
  - `item_b` (`BaseLayer`): Segundo elemento.
- **Retorno**: `None`.

#### `reverse(recursive: bool = False) -> None`
- **Descrição**: Inverte in-place a ordem dos filhos contidos no contêiner. Quando `recursive=True`, desce recursivamente invertendo também a ordem interna dos nós de qualquer `GroupLayer` aninhado.
- **Parâmetros**:
  - `recursive` (`bool`): Se `True`, inverte recursivamente toda a árvore de ramos descendentes (padrão: `False`).
- **Retorno**: `None`.

#### `clear() -> None`
- **Descrição**: Remove todos os filhos contidos no grupo.
- **Retorno**: `None`.

#### `pop(index: int = -1) -> Container | Layer`
- **Descrição**: Remove e retorna o filho localizado no índice especificado (por padrão, o último).
- **Retorno**: `Container | Layer`.

#### `@property matrix -> np.ndarray`
- **Descrição**: Calcula a matriz de transformação 3x3 acumulada do grupo no espaço global, combinando a matriz do pai, a inversa do pai e a transformação local do grupo: `parent.matrix @ _parent_inverse @ transform.matrix`.
- **Retorno**: `np.ndarray` — Matriz 3x3 (`float32`).

#### `@property global_region -> Region`
- **Descrição**: Calcula o AABB (*Axis-Aligned Bounding Box*) global do grupo no espaço do Canvas. Se o grupo contiver filhos, retorna a união (`|`) das regiões globais de todos os filhos visíveis.
- **Retorno**: `Region` — Região delimitadora global.

#### `@property region -> Region`
- **Descrição**: Retorna a união das regiões locais dos filhos que pertencem a este grupo.
- **Retorno**: `Region`.
