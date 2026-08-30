# Guia de Camadas (`layer.py`)

O módulo `anicrop.layer` define as abstrações de camadas individuais da biblioteca, separando a classe abstrata base de camadas (`BaseLayer`), as camadas folha com pixels (`Layer`) e os patches de edições locais (`EditLayer`).

---

## 1. Classe Base de Camadas: `BaseLayer` (`anicrop.container.BaseLayer`)

A `BaseLayer` é a classe abstrata da qual tanto `Layer` quanto `GroupLayer` herdam. Ela fornece o comportamento fundamental compartilhado por todos os elementos gráficos da hierarquia:

- **Atributos de Composição**: `opacity` (0.0 a 1.0), `blend_mode` (enum `BlendMode`), `visible` (`bool`) e `name` (`str`).
- **Transformações Matriciais**: Mantém a propriedade `transform` que retorna o compositor mutável (`ComposerRel`), já instanciado no `__init__`.
- **Estratégias de Layout**: Mantém o controlador `layout` (`@layout.setter`) para delegar a resolução da geometria para estratégias espaciais (ex: `FitGeometry`).
- **Navegação Hierárquica (`NodeContainerProtocol`)**: Mantém a referência `_parent_inverse` (matriz inversa do contêiner pai) e a interface do protocolo de nós da árvore espacial.

### Uso Idiomático da Propriedade `transform`

A propriedade `@property transform -> Composer` **já vem pronta e instanciada** desde o `__init__` da camada. Portanto, **NÃO é necessário** instanciar um objeto `Transform` separado ou chamar `set_transform()` para realizar transformações básicas.

Pode-se encadear rotações, escalas e translações diretamente na propriedade `.transform` do layer:

```python
# Uso Idiomático Direto (Recomendado):
layer = Layer(img, name="layer1")
layer.transform.rotate(45).scale(1.5, 1.5).translate(100, 50)
```

---

## 2. Camada Concreta: `Layer` (`anicrop.layer.Layer`)

A classe `Layer` representa uma camada de imagem folha (contendo pixels reais) na hierarquia.

### Herança e Protocolo de Nó (`NodeContainerProtocol`)
- Herda de **`BaseLayer`**.
- Implementa o **`NodeContainerProtocol`**: Inicializa o atributo `self.parent = _NULL_CONTAINER` (que é atualizado quando a camada é adicionada a um `Container` ou `GroupLayer`) e herda de `BaseLayer` o atributo `self._parent_inverse` para permitir navegação e cálculo das matrizes globais relativas ao pai.

Os pixels originais da imagem nunca são modificados diretamente. Toda alteração espacial é gerenciada dinamicamente via transformações matriciais no momento do render.

---

### Principais Propriedades e Métodos de `Layer`

#### Construtor Polimórfico (`__init__` via `@ovld`)
- **Descrição**: Inicializa uma nova camada `Layer` de forma polimórfica utilizando *multiple dispatch* nativo via `@ovld`, aceitando uma imagem base (`Image`), uma região delimitadora pura (`Region`) ou dimensões `size: tuple[int, int]`.
- **Sobrecargas (`@ovld`)**:
  - `Layer(image: Image, *, opacity: float = 1.0, blend_mode: BlendMode = BlendMode.NORMAL, name: str = 'Layer', format: ImageFormat | None = None)`: Cria camada com tamanho da imagem e popula o primeiro `EditLayer` com os pixels da imagem.
  - `Layer(region: Region, *, opacity: float = 1.0, blend_mode: BlendMode = BlendMode.NORMAL, name: str = 'Layer', format: ImageFormat = ImageFormat.RGBA)`: Cria camada pura apenas com moldura espacial (sem edits prévios).
  - `Layer(size: tuple, *, opacity: float = 1.0, blend_mode: BlendMode = BlendMode.NORMAL, name: str = 'Layer', format: ImageFormat = ImageFormat.RGBA)`: Cria camada pura com dimensões `(w, h)` (sem edits prévios).
- **Retorno**: Instância de `Layer`.

#### `@property transform -> Composer`
- **Descrição**: Acessa o compositor mutável de transformações da camada. Permite o encadeamento direto de operações espaciais (`rotate`, `scale`, `translate`).

#### `@property region -> Region` / `@region.setter`
- **Descrição**: Retorna ou define a área/região bounding box local da camada (`Region`).
- **Deslocamento Idiomático**: Para mover a posição da camada sem alterar suas dimensões físicas, utilize a soma com tuplas:
  ```python
  layer.region += (100, 200)  # Desloca a regiao local em (dx, dy)
  ```
- **Retorno**: `Region` — A região delimitadora local da camada.

#### `@property global_region -> Region`
- **Descrição**: Calcula e retorna o Axis-Aligned Bounding Box (AABB) real da camada no espaço global do Canvas após a aplicação de todas as transformações (rotação, escala, translação e matrizes de grupos pais).
- **Retorno**: `Region` — A região delimitadora no espaço do Canvas.

#### `@property image -> Image`
- **Descrição**: Retorna o objeto `Image` do `EditLayer` principal (base) contido na camada.
- **Retorno**: `Image` — Os pixels base da camada.

#### `@property canvas_size -> tuple[int, int]`
- **Descrição**: Retorna o tamanho do Canvas associado ou, caso não haja Canvas definido, a dimensão da própria região da camada `(width, height)`.
- **Retorno**: `tuple[int, int]`.

#### `@property format -> ImageFormat`
- **Descrição**: Retorna o formato de imagem (RGBA, RGB, GRAY, etc.) dos dados contidos no layer.
- **Retorno**: `ImageFormat`.

#### `add_edit(image: Image, region: Region, blend_mode: BlendMode = BlendMode.NORMAL) -> None`
- **Descrição**: Adiciona uma nova edição/patch (`EditLayer`) à fila da camada. Isso permite aplicar modificações em áreas locais da camada sem destruir a imagem original.
- **Parâmetros**:
  - `image` (`Image`): Imagem contendo os pixels do patch.
  - `region` (`Region`): Região local onde o patch deve ser aplicado.
  - `blend_mode` (`BlendMode`): Modo de mesclagem do patch com a camada base.
- **Retorno**: `None`.

---

## 3. Máscaras e Sistema de Efeitos (`anicrop.mask`, `anicrop.effect`, `anicrop.filter`)

Tanto `Layer` quanto `GroupLayer` (através de `BaseLayer`) suportam máscara não-destrutiva única e uma fila de efeitos de pós-processamento.

### 3.1. Gerenciamento de Máscara Única em `BaseLayer`

Cada camada possui no máximo uma máscara ativa associada, vinculada à sua geometria espacial no momento da criação:

- **`set_mask(image: Image, region: Region, invert: bool = False, visible: bool = True, name: str = 'Mask') -> Mask`**: Cria e atribui a máscara da camada, ancorando automaticamente a matriz inversa global da camada.
- **`remove_mask() -> None` / `clear_mask()`**: Remove a máscara ativa da camada.
- **`@property mask -> Mask | None`**: Retorna a máscara ativa ou `None` caso não haja.

#### Indexação Direta e Histórico Reativo (`ProxyMask`):
A classe `Mask` suporta fatiamento e mutação atômica de pixels via indexação direta:
```python
mask = layer.set_mask(img_mask, Region.from_size(200, 200))

# Modifica regioes de pixels diretamente (suporta slice, Region e tuplas):
mask[Region.from_rect(0, 0, 50, 50)] = 0
mask[:, :25] = 128

# Em documentos reativos, toda mutacao por indexacao ou troca de atributos gera micro-snapshots com Undo/Redo:
mask.invert = True
doc.history.undo()  # Desfaz a inversao
doc.history.undo()  # Restaura o buffer de pixels anterior
```

---

### 3.2. O Protocolo `Effect` (`anicrop.effect.Effect`)

O protocolo `Effect` é estritamente puro e desacoplado, sem estado espacial interno:

```python
@runtime_checkable
class Effect(Protocol):
    def get_padding(self) -> tuple[int, int, int, int]:
        """Retorna a margem de expansão (top, right, bottom, left) necessária."""
        ...

    def apply(self, image: Image, matrix: np.ndarray) -> Image:
        """Processa e transforma o buffer de imagem recebendo a matriz afim ativa."""
        ...

    def merge(self, other: Effect, matrix: np.ndarray) -> Effect | None:
        """Tenta combinar este efeito com outro compatível."""
        ...
```

---

### 3.3. Envelope Geométrico: `BoundEffect` (`anicrop.effect.BoundEffect`)

A classe `BoundEffect` implementa o protocolo `Effect` e atua como envelope para ancorar um efeito puro à camada:
- **`self.matrix`**: Armazena a matriz inversa da camada na vinculação.
- **Delta Matrix**: No método `apply`, calcula $\Delta M = M_{\text{render}} \cdot M_{\text{base\_inv}}$ e repassa para o efeito puro.
- **Modulação por Máscara**: Se `mask` for fornecida, modula o resultado automaticamente (`mask.modulate_blend`).
- **Controle de Visibilidade**: Respeita a flag `visible`.

---

### 3.4. Filtro de Desfoque Anisotrópico: `BlurFilter` (`anicrop.filter.BlurFilter`)

O `BlurFilter` implementa desfoque Gaussiano ou Box de alta performance:
- **Raio 1D ou 2D**: `radius=5.0` (isotrópico) ou `radius=(8.0, 2.0)` (anisotrópico).
- **Ângulo Direcional**: `angle=45.0` difunde o desfoque em qualquer inclinação angular contínua.
- **Fusão de Covariância (`merge`)**: Ao fundir dois filtros Gaussianos através de matrizes afins, soma analiticamente seus tensores de covariância ($\Sigma_{\text{total}} = \Sigma_1 + \Sigma_2$) e extrai os raios e ângulo equivalentes via autovalores/autovetores sem perda de precisão.

---

### 3.5. Gerenciamento de Efeitos em `BaseLayer`

- **`add_effect(effect: Effect) -> Effect`**: Adiciona um efeito livremente à fila de pós-processamento.
- **`bind_effect(effect: Effect, mask: Mask | None = None, visible: bool = True) -> BoundEffect`**: Cria um `BoundEffect` ancorado à matriz inversa da camada e o anexa à fila.
- **`remove_effect(effect: Effect) -> None`**: Remove o efeito da fila.
- **`clear_effects() -> None`**: Remove todos os efeitos.
- **`@property effects -> tuple[Effect, ...]`**: Retorna uma tupla imutável com a fila de efeitos ativos.

---

## 4. Visão Geral de `EditLayer` (`anicrop.layer.EditLayer`)

O `EditLayer` representa uma alteração pontual (patch) ou sub-camada aplicada sobre o `Layer` pai. 

### Principais Características:
- **`image` (`Image`)**: Pixel data do patch local.
- **`region` (`Region`)**: Região local relativa ao Layer pai.
- **`local_matrix` (`np.ndarray`)**: Matriz de transformação combinando a matriz do edit com a posição da região (`matrix @ mat_position(region)`).
- **`get_lod(scale_factor: float) -> tuple[Image, np.ndarray]`**: Gera/retorna a pirâmide de nível de detalhe (LOD - Level of Detail) apropriada para otimizar a renderização de imagens de altíssima resolução (ex: Zarr).
