# Guia Espacial — Classe `Region` (`spatial.py`)

O módulo `anicrop.spatial` fornece as primitivas de geometria bidimensional e cálculos espaciais da biblioteca. A classe principal consumida publicamente é a `Region`, que representa um retângulo delimitador 2D (formado internamente por dois intervalos `Span` para os eixos X e Y).

---

## `Region` (`anicrop.spatial.Region`)

`Region` é uma classe imutável (`@dataclass(frozen=True)`) que define uma área retangular por meio dos seus intervalos horizontais (`x: Span`) e verticais (`y: Span`). Suporta coordenadas negativas e oferece operadores matemáticos expressivos para deslocamento, união e interseção.

---

### Mapeamento de Métodos e Propriedades de `Region`

#### `from_size(width: int, height: int) -> Region` *(Class Method)*
- **Descrição**: Cria uma nova `Region` posicionada na origem `(0, 0)` com as dimensões especificadas.
- **Parâmetros**:
  - `width` (`int`): Largura da região (deve ser > 0).
  - `height` (`int`): Altura da região (deve ser > 0).
- **Retorno**: `Region` — Região `(0, 0, width, height)`.

#### `from_rect(x: int, y: int, width: int, height: int) -> Region` *(Class Method)*
- **Descrição**: Cria uma `Region` especificando as coordenadas iniciais `(x, y)` do canto superior esquerdo e o tamanho `(width, height)`.
- **Parâmetros**:
  - `x` (`int`): Posição X inicial.
  - `y` (`int`): Posição Y inicial.
  - `width` (`int`): Largura em pixels.
  - `height` (`int`): Altura em pixels.
- **Retorno**: `Region` — A instância delimitadora correspondente.

#### `@property size -> tuple[int, int]`
- **Descrição**: Retorna uma tupla `(width, height)` contendo a largura e altura da região.
- **Retorno**: `tuple[int, int]`.

#### `@property top_left -> tuple[int, int]`
- **Descrição**: Retorna uma tupla `(x, y)` correspondente às coordenadas do canto superior esquerdo (coordenadas de início dos spans `x.start` e `y.start`).
- **Retorno**: `tuple[int, int]`.

#### `@property width -> int`
- **Descrição**: Retorna a largura em pixels (`x.length`).
- **Retorno**: `int`.

#### `@property height -> int`
- **Descrição**: Retorna a altura em pixels (`y.length`).
- **Retorno**: `int`.

#### `@property area -> int`
- **Descrição**: Retorna a área total em pixels da região (`width * height`).
- **Retorno**: `int`.

---

### Operadores Matemáticos de `Region`

#### Deslocamento / Translação Pura (`+`, `-`, `+=`, `-=`)
- **`region + (dx, dy)` / `region += (dx, dy)`**: Desloca a posição `(x, y)` da região preservando rigorosamente as suas dimensões originais (`width` e `height`).
- **Uso Idiomático**: Ideal para transladar camadas sem acúmulo de erro de arredondamento (*anti-drift*):
  ```python
  layer.region += (150, 200)  # Desloca a regiao local em X e Y
  ```
- **Tipos de `offset` aceitos**:
  - `int`: Desloca X e Y pelo mesmo valor.
  - `tuple[int, int]`: Desloca `(dx, dy)`.
  - `Region`: Utiliza as coordenadas `(top_left)` da outra região como offset.
- **Retorno**: Nova `Region` deslocada preservando as dimensões originais (`size`).

#### União de Regiões (`|`)
- **`region_a | region_b`**: Calcula o menor retângulo delimitador (AABB) que engloba totalmente ambas as regiões.
- **Retorno**: Nova `Region` representando a união global.

#### Interseção Global (`&`)
- **`region_a & region_b`**: Calcula a área comum de sobreposição entre duas regiões nas **coordenadas globais do Canvas**.
- **Retorno**: Nova `Region` com a interseção em coordenadas globais (utilizada internamente no renderizador para calcular patches de destino).

---

### Métodos Avançados de Interseção e Geometria

#### `overlaps(other: Region) -> bool`
- **Descrição**: Verifica se a região atual se sobrepõe/interseca com outra região `other`.
- **Parâmetros**:
  - `other` (`Region`): Região de teste.
- **Retorno**: `bool` — `True` se houver sobreposição em ambos os eixos X e Y.

#### `overlap_with(other: Region) -> Region` *(Interseção Local / Source Slicing)*
- **Descrição**: Calcula a área de interseção **relativa às coordenadas locais da região atual** (`self`). Transforma o topo-esquerdo da região atual em `(0, 0)`.
- **Uso Principal**: Essencial para fatiamento de matrizes NumPy (`ndarray`), permitindo extrair a sub-região de amostragem original sem desalinhamento de índices: `slice = img_region.overlap_with(canvas_region)`.
- **Retorno**: `Region` — Região relativa ao espaço local de `self`.

#### `align(ref: Region, x_factor: float = 0.5, y_factor: float = 0.5) -> Region`
- **Descrição**: Alinha a região atual em relação a uma região de referência (`ref`), distribuindo o espaço livre (*slack*) com base em âncoras normalizadas de `0.0` a `1.0`.
- **Parâmetros**:
  - `ref` (`Region`): Região de referência para alinhamento.
  - `x_factor` (`float`): `0.0` (esquerda), `0.5` (centro horizontal), `1.0` (direita).
  - `y_factor` (`float`): `0.0` (topo), `0.5` (centro vertical), `1.0` (base).
- **Fórmula Matemática**:
  $$\text{pos}_x = \text{ref.x.start} + \text{x\_factor} \times (\text{ref.width} - \text{self.width})$$
  $$\text{pos}_y = \text{ref.y.start} + \text{y\_factor} \times (\text{ref.height} - \text{self.height})$$
- **Retorno**: Nova `Region` alinhada preservando o tamanho original.

#### `expand(all: int | tuple[int, int] | None = None, *, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> Region`
- **Descrição**: Expande as margens da região para fora por um número específico de pixels.
- **Retorno**: Nova `Region` expandida.

#### `shrink(all: int | tuple[int, int] | None = None, *, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> Region`
- **Descrição**: Contrai as margens da região para dentro (garantindo dimensão mínima de 1 pixel e impedindo *drift* de coordenadas).
- **Retorno**: Nova `Region` contraída.

---

### Funções Utilitárias do Módulo

#### `rect_to_region(rect: tuple[int, int, int, int]) -> Region`
- **Descrição**: Converte uma tupla no formato OpenCV/PIL `(x, y, width, height)` em um objeto `Region`.
- **Retorno**: `Region`.
