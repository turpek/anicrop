# Guia Espacial — Primitivas `Point`, `Span` e `Region` (`spatial.py`)

O módulo `anicrop.spatial` fornece as primitivas de geometria bidimensional contínua e cálculos espaciais da biblioteca. Todas as operações espaciais operam internamente em números de ponto flutuante contínuos (`float`), permitindo transformações analíticas exatas e eliminando o erro de quantização (*size drift*). A conversão para o espaço discreto de pixels (`int`) ocorre apenas na fronteira com os buffers de renderização.

---

## 1. `Point` (`anicrop.spatial.Point`)

`Point` é uma `NamedTuple` contínua `(x: float, y: float)` que herda diretamente de `tuple[float, float]`. É utilizada para representar posições, dimensões e vetores espaciais 2D.

### Atributos e Métodos de `Point`
- **Atributos**:
  - `point.x` (`float`): Coordenada X / largura.
  - `point.y` (`float`): Coordenada Y / altura.
- **Acesso por Índice / Desempacotamento**:
  - Totalmente compatível com tuplas: `x, y = point` ou `w, h = region.size`.
- **Igualdade com Tolerância Analítica (`__eq__`)**:
  - Comparações com tuplas/outros pontos utilizam `math.isclose` com tolerância de `1e-4`, evitando discrepâncias por resíduos de ponto flutuante.
- **`to_int(mode: str = 'round') -> tuple[int, int]`**:
  - Converte as coordenadas contínuas para inteiros discretos.
  - Modos suportados: `'round'` (arredondamento padrão), `'floor'` (piso) e `'ceil'` (teto).

```python
from anicrop.spatial import Point

p = Point(10.4, 20.8)
print(p.x, p.y)  # 10.4 20.8
print(p.to_int("round"))  # (10, 21)
print(p.to_int("floor"))  # (10, 20)
```

---

## 2. `Span` (`anicrop.spatial.Span`)

`Span` representa um intervalo contínuo unidimensional imutável, definido por uma coordenada inicial `start: float` e um comprimento positivo `length: float`.

### Sobrecargas do Construtor (`__init__`)
- `Span(length: float, /)`: Cria um intervalo com `start=0.0` e o comprimento especificado.
- `Span(start: float, length: float, /)`: Cria um intervalo especificando início e comprimento.

### Propriedades e Métodos de `Span`
- **`@property start -> float`**: Posição inicial do intervalo (suporta coordenadas negativas).
- **`@property length -> float`**: Comprimento do intervalo (deve ser $> 0$).
- **`@property end -> float`**: Posição final (`start + length`).
- **`to_int(mode: str = 'round') -> tuple[int, int]`**: Retorna `(start_int, length_int)` discretizados.
- **Operadores**:
  - `span + offset` / `span - offset`: Deslocamento puro preservando rigorosamente o `length` original.
  - `span & other`: Interseção global entre intervalos.
  - `span | other`: União mínima envolvente entre intervalos.

---

## 3. `Region` (`anicrop.spatial.Region`)

`Region` é uma classe imutável (`@dataclass(frozen=True)`) que define uma área retangular 2D por meio dos seus intervalos horizontais (`x: Span`) e verticais (`y: Span`).

---

### Mapeamento de Métodos e Propriedades de `Region`

#### `from_size(width: float, height: float) -> Region` *(Class Method)*
- **Descrição**: Cria uma nova `Region` posicionada na origem `(0.0, 0.0)` com as dimensões especificadas.
- **Parâmetros**:
  - `width` (`float`): Largura da região (deve ser $> 0$).
  - `height` (`float`): Altura da região (deve ser $> 0$).
- **Retorno**: `Region` — Região `(0.0, 0.0, width, height)`.

#### `from_rect(x: float, y: float, width: float, height: float) -> Region` *(Class Method)*
- **Descrição**: Cria uma `Region` especificando as coordenadas iniciais `(x, y)` do canto superior esquerdo e o tamanho `(width, height)`.
- **Parâmetros**:
  - `x` (`float`): Posição X inicial.
  - `y` (`float`): Posição Y inicial.
  - `width` (`float`): Largura.
  - `height` (`float`): Altura.
- **Retorno**: `Region` — A instância delimitadora correspondente.

#### `@property size -> Point`
- **Descrição**: Retorna uma instância de `Point(width, height)` contendo a largura e altura da região.
- **Retorno**: `Point` (subtipo de `tuple[float, float]`).

#### `@property top_left -> Point`
- **Descrição**: Retorna uma instância de `Point(x, y)` correspondente às coordenadas do canto superior esquerdo (`x.start`, `y.start`).
- **Retorno**: `Point` (subtipo de `tuple[float, float]`).

#### `@property bottom_right -> Point`
- **Descrição**: Retorna uma instância de `Point(x, y)` correspondente ao canto inferior direito (`x.end`, `y.end`).
- **Retorno**: `Point` (subtipo de `tuple[float, float]`).

#### `@property width -> float` / `@property height -> float`
- **Descrição**: Retorna a largura (`x.length`) ou altura (`y.length`) contínua.
- **Retorno**: `float`.

#### `@property area -> float`
- **Descrição**: Retorna a área total da região (`width * height`).
- **Retorno**: `float`.

---

### Operadores Matemáticos de `Region`

#### Deslocamento / Translação Pura (`+`, `-`, `+=`, `-=`)
- **`region + (dx, dy)` / `region += (dx, dy)`**: Desloca a posição `(x, y)` da região preservando rigorosamente as suas dimensões originais (`width` e `height`).
- **Uso Idiomático**: Ideal para transladar camadas sem acúmulo de erro de arredondamento (*anti-drift*):
  ```python
  layer.region += (150.0, 200.0)  # Desloca a região local em X e Y
```
- **Tipos de `offset` aceitos**:
  - `float` / `int`: Desloca X e Y pelo mesmo valor.
  - `tuple[float, float]` / `Point`: Desloca `(dx, dy)`.
  - `Region`: Utiliza as coordenadas `top_left` da outra região como offset.
- **Retorno**: Nova `Region` deslocada preservando as dimensões originais (`size`).

#### União de Regiões (`|`)
- **`region_a | region_b`**: Calcula o menor retângulo delimitador (AABB) que engloba totalmente ambas as regiões.
- **Retorno**: Nova `Region` representando a união global.

#### Interseção Global (`&`)
- **`region_a & region_b`**: Calcula a área comum de sobreposição entre duas regiões nas **coordenadas globais do Canvas**.
- **Retorno**: Nova `Region` com a interseção em coordenadas globais.

---

### Métodos Avançados de Interseção e Geometria

#### `overlaps(other: Region) -> bool`
- **Descrição**: Verifica se a região atual se sobrepõe/interseca com outra região `other`.
- **Retorno**: `bool` — `True` se houver sobreposição em ambos os eixos X e Y.

#### `overlap_with(other: Region) -> Region` *(Interseção Local / Source Slicing)*
- **Descrição**: Calcula a área de interseção **relativa às coordenadas locais da região atual** (`self`). Transforma o topo-esquerdo da região atual em `(0.0, 0.0)`.
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

#### `expand(all: float | tuple[float, float] | None = None, *, left: float = 0, right: float = 0, top: float = 0, bottom: float = 0) -> Region`
- **Descrição**: Expande as margens da região para fora pelo deslocamento especificado.
- **Retorno**: Nova `Region` expandida.

#### `shrink(all: float | tuple[float, float] | None = None, *, left: float = 0, right: float = 0, top: float = 0, bottom: float = 0) -> Region`
- **Descrição**: Contrai as margens da região para dentro (garantindo dimensão mínima de 1e-4 e impedindo *drift* de coordenadas).
- **Retorno**: Nova `Region` contraída.

---

### Funções Utilitárias e de Quantização

#### `rect_to_region(rect: tuple[float, float, float, float]) -> Region`
- **Descrição**: Converte uma tupla no formato OpenCV/PIL `(x, y, width, height)` em um objeto `Region`.
- **Retorno**: `Region`.

#### `to_int_span(span: Span, mode: str = "round") -> tuple[int, int]`
- **Descrição**: Converte um `Span` contínuo para inteiros discretos `(start, length)`.

#### `to_int_region(region: Region, mode: str = "round") -> tuple[int, int, int, int]`
- **Descrição**: Converte uma `Region` contínua para uma tupla de inteiros discretos `(x, y, width, height)`.
