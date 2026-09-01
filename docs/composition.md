# Guia de Composição de Camadas — Módulo `Composition` (`composition.py`)

O módulo `anicrop.composition` fornece operações puras e desacopladas para manipulação e reestruturação da hierarquia gráfica, incluindo clonagem profunda isolada, agrupamento não-destrutivo (`merge`) e rasterização plana (`flatten`).

---

## 1. Princípios Arquiteturais da Composição

- **Isolamento Total de Mutação**: Operações de clonagem (`clone_layer`, `clone_group`, `clone_node`) duplicam inteiramente matrizes afins 3x3, compositores (`Composer`), buffers de máscara e instâncias de efeitos (`BoundEffect`), garantindo que alterações no clone não reflitam no nó original.
- **Eficiência de Memória (Zero-Copy)**: Embora a estrutura de metadados e matrizes seja duplicada, os buffers pesados de pixels dos patches (`EditLayer.image`) são compartilhados por referência de forma segura e imutável.
- **Preservação de Efeitos Individuais (`merge`)**: Diferente de achatar edits em uma única camada (o que destruiria filtros individuais e modos de mesclagem), `merge` encapsula as camadas em um `GroupLayer` desacoplado, preservando cada efeito, máscara e opacidade individual.
- **Rasterização Fiel (`flatten`)**: Utiliza `CanvasRender.render_container` para assar a cena completa em uma única imagem rasterizada com suporte a modos de interpolação, formato de cores e cor de fundo.

---

## 2. Clonagem de Nós Gráficos

### `clone_layer(layer: Layer) -> Layer`
- **Descrição**: Cria uma cópia profunda e isolada de uma camada folha `Layer`. Duplica matrizes 3x3, cópias dos buffers de máscara, novos `BoundEffect` e enquadramentos ativos de layout.
- **Parâmetros**:
  - `layer` (`Layer`): A camada a ser clonada.
- **Retorno**: `Layer` — Uma nova camada independente.

```python
from anicrop.composition import clone_layer

cloned = clone_layer(original_layer)
cloned.transform.rotate(45).translate(100, 50)  # Não afeta original_layer
```

---

### `clone_group(group: GroupLayer) -> GroupLayer`
- **Descrição**: Cria uma cópia profunda de um `GroupLayer` e recursivamente de todos os seus filhos (`Layer` ou subgrupos).
- **Parâmetros**:
  - `group` (`GroupLayer`): O grupo a ser clonado.
- **Retorno**: `GroupLayer` — Um novo grupo desacoplado.

---

### `clone_node(node: BaseLayer) -> BaseLayer`
- **Descrição**: Função polimórfica que despacha automaticamente para `clone_layer` ou `clone_group` com base no tipo concreto do nó.
- **Parâmetros**:
  - `node` (`BaseLayer`): Qualquer nó gráfico (`Layer` ou `GroupLayer`).
- **Retorno**: `BaseLayer` — Clone do nó.

---

## 3. Agrupamento e Fusão de Camadas

### `merge(layers: Sequence[BaseLayer] | Container, name: str = "Group") -> GroupLayer`
- **Descrição**: Cria uma nova composição agrupada não-destrutiva (`GroupLayer`) contendo cópias desacopladas das camadas fornecidas. O `GroupLayer` gerencia nativamente a região delimitadora dinâmica de seus filhos.
- **Parâmetros**:
  - `layers` (`Sequence[BaseLayer] | Container`): Lista ou contêiner de camadas a serem agrupadas.
  - `name` (`str`, opcional): Nome do novo grupo criado (padrão: `"Group"`).
- **Retorno**: `GroupLayer` — O grupo resultante contendo as camadas clonadas.

```python
from anicrop.composition import merge

# Agrupa múltiplas camadas em um GroupLayer desacoplado
group = merge([layer1, layer2, sub_group], name="ComposicaoPersonagem")
```

---

#### `flatten(layers: Sequence[BaseLayer] | Container, name: str = "Layer", format: ImageFormat | None = None, interp: InterpMode = InterpMode.LANCZOS, bg_color: tuple[int, ...] | None = None) -> Layer`
- **Descrição**: Rasteriza o conjunto ou contêiner de camadas calculando automaticamente a região global delimitadora (*ROI*) e gera um `Layer` plano.
- **Herança de Propriedades**:
  - `format`: Se `None` (padrão), infere automaticamente o formato da camada superior (`layers[-1].format`).
  - `blend_mode`: Herda o modo de mesclagem da camada base inferior (`layers[0].blend_mode`) ou do grupo (`group.blend_mode`).
  - `visible`: Herda o estado de visibilidade da camada de topo (`layers[-1].visible`) ou do grupo.
  - `opacity`: Se o alvo for um `GroupLayer`, preserva a opacidade global do grupo.
- **Parâmetros**:
  - `layers` (`Sequence[BaseLayer] | Container`): Lista ou contêiner de camadas a serem rasterizadas.
  - `name` (`str`, opcional): Nome da camada plana resultante (padrão: `"Layer"`).
  - `format` (`ImageFormat | None`, opcional): Formato de cores de saída (padrão: infere do topo).
  - `interp` (`InterpMode`, opcional): Modo de interpolação na renderização (padrão: `InterpMode.LANCZOS`).
  - `bg_color` (`tuple[int, ...] | None`, opcional): Cor de fundo opcional para preenchimento.
- **Retorno**: `Layer` — Uma única camada folha contendo a imagem rasterizada.

```python
from anicrop.composition import flatten
from anicrop.enums import ImageFormat, InterpMode

# Achata as camadas em uma única camada rasterizada
flat_layer = flatten(
    [layer_fundo, layer_efeito], name="FundoAssado", interp=InterpMode.LANCZOS
)
```

---

## 4. Fachada Estática: `LayerComposition`

Para maior comodidade, a classe `LayerComposition` reúne os métodos de composição como métodos estáticos:

```python
from anicrop.composition import LayerComposition

# Clonagem
cloned = LayerComposition.clone(node)

# Agrupamento
group = LayerComposition.merge([layer1, layer2], name="NovoGrupo")

# Rasterização
flat = LayerComposition.flatten([layer1, layer2], name="LayerPlano")
```

---

## 5. Serviço Acoplado ao Documento: `doc.combine` (`Combine`)

O serviço `doc.combine` orquestra operações de mescla e fusão descendente (*"Merge Down"*) no contexto da árvore de camadas do documento, garantindo validação de mesmo nível e controle de substituição na pilha.

### `doc.combine.merge(target: BaseLayer | str, name: str, count: int = 1, remove_source: bool = True) -> GroupLayer`
- **Descrição**: Mescla a camada `target` com até `count` camadas visíveis imediatamente abaixo dela em seu contêiner pai, gerando um novo `GroupLayer`.
- **Parâmetros**:
  - `target` (`BaseLayer | str`): Camada de topo (instância ou nome).
  - `name` (`str`): Nome obrigatório do novo grupo criado.
  - `count` (`int`): Quantidade de camadas visíveis abaixo a incluir (padrão: `1`).
  - `remove_source` (`bool`): Se `True` (padrão), remove as camadas de origem e insere o grupo na posição mais baixa.
- **Retorno**: `GroupLayer` — O grupo criado.

```python
# Mescla a camada 'detalhe' com 1 camada visível abaixo:
grupo = doc.combine.merge("detalhe", name="GrupoDetalhe", count=1)
```

---

### `doc.combine.flatten(target: BaseLayer | str, name: str, count: int = 1, format: ImageFormat | None = None, interp: InterpMode = InterpMode.LANCZOS, bg_color: tuple[int, ...] | None = None, remove_source: bool = True) -> Layer`
- **Descrição**: Rasteriza a camada `target` com até `count` camadas visíveis abaixo dela em uma única camada `Layer` plana. A camada resultante herda o `blend_mode` da camada base inferior (`sequence[0]`) e o `ImageFormat` e visibilidade da camada `target`.
- **Parâmetros**:
  - `target` (`BaseLayer | str`): Camada de topo.
  - `name` (`str`): Nome obrigatório da camada plana resultante.
  - `count` (`int`): Quantidade de camadas visíveis abaixo a achatar (padrão: `1`).
  - `format` (`ImageFormat | None`): Formato de cor (se omitido, herda `target.format`).
  - `interp` (`InterpMode`): Modo de interpolação na renderização.
  - `bg_color` (`tuple[int, ...] | None`): Cor de fundo opcional.
  - `remove_source` (`bool`): Se `True` (padrão), substitui as camadas na pilha.
- **Retorno**: `Layer` — A camada plana resultante.

```python
# Achata 'efeitos' com 2 camadas visíveis abaixo preservando o formato de cor:
flat = doc.combine.flatten("efeitos", name="EfeitosAssados", count=2)
```

---

### `doc.combine.bake(target: GroupLayer | str, name: str | None = None, format: ImageFormat | None = None, interp: InterpMode = InterpMode.LANCZOS, bg_color: tuple[int, ...] | None = None) -> Layer`
- **Descrição**: Assa os filhos internos de um `GroupLayer` em uma única camada `Layer` plana, substituindo o grupo original em seu container pai.
- **Parâmetros**:
  - `target` (`GroupLayer | str`): Grupo alvo a ser assado (instância ou nome).
  - `name` (`str | None`): Nome da camada resultante (se omitido, herda o nome do grupo).
  - `format` (`ImageFormat | None`): Formato de cor (se omitido, herda `target.format`).
  - `interp` (`InterpMode`): Modo de interpolação na renderização.
  - `bg_color` (`tuple[int, ...] | None`): Cor de fundo opcional.
- **Retorno**: `Layer` — A camada plana resultante inserida na mesma posição do grupo.

```python
# Assa o grupo 'Personagem' diretamente em uma camada plana:
camada_assada = doc.combine.bake("Personagem")
```

---

### `doc.combine.bake_stack(name: str = "Layer", format: ImageFormat = ImageFormat.RGBA, interp: InterpMode = InterpMode.LANCZOS, bg_color: tuple[int, ...] | None = None) -> Layer`
- **Descrição**: Método especialista para assar toda a pilha (`doc.stack`) do documento em uma única camada `Layer` plana, limpando as camadas anteriores e inserindo o resultado.
- **Parâmetros**:
  - `name` (`str`): Nome da camada resultante (padrão: `"Layer"`).
  - `format` (`ImageFormat`): Formato de cor (padrão: `ImageFormat.RGBA`).
  - `interp` (`InterpMode`): Modo de interpolação na renderização.
  - `bg_color` (`tuple[int, ...] | None`): Cor de fundo opcional.
- **Retorno**: `Layer` — A camada plana resultante que passa a ser o único nó na pilha do documento.

```python
# Achata a pilha inteira do documento:
camada_final = doc.combine.bake_stack(name="CenaCompleta")
```
