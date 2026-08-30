# Guia de Transformações Espaciais (`transform.py`)

O módulo `anicrop.transform` implementa a fundação matemática de matrizes homogêneas 3x3 de transformação 2D (translação, rotação, escala e pivôs). Ele separa com clareza o **estado mutável acumulado** (`Composer`) das **intenções imutáveis** (`Transform`).

---

## 1. Estado Mutável Acumulado: `Composer`

O `Composer` (e suas especializações `ComposerRel` e `ComposerAbs`) armazena o estado dinâmico mutável de transformação associado a um `Layer` ou `GroupLayer`.

### Como Acessar e Usar
Em vez de instanciar objetos `Transform` manualmente para operações comuns, **todo `Layer` e `GroupLayer` já instancia um `Composer` por padrão no seu atributo `.transform`**.

Isso permite aplicar transformações encadeadas diretamente no objeto:

```python
# Uso Direto e Idiomático:
layer = Layer(img)
layer.transform.rotate(45).scale(2.0, 2.0).translate(100, 50)
```

Separa internamente a matriz em duas partes para permitir otimizações de cache (invalidação cirúrgica de rotação vs translação):
- `_distortion`: Matriz acumulada de rotação e escala ($2 \times 2$ linear).
- `_translation`: Matriz acumulada de translação ($3 \times 3$).

---

### Principais Métodos de `Composer`

#### `@property matrix -> np.ndarray`
- **Descrição**: Retorna a matriz final combinada (`_translation @ _distortion`).
- **Retorno**: `np.ndarray` (3x3 `float32`).

#### `rotate(angle: float, px: float = 0.5, py: float = 0.5) -> Self`
- **Descrição**: Modifica o composer *in-place* acumulando uma rotação.
- **Parâmetros**:
  - `angle` (`float`): Ângulo em graus.
  - `px`, `py` (`float`): Coordenadas de pivô (relativo `0.5, 0.5` por padrão em `ComposerRel`).
- **Retorno**: `self`.

#### `scale(sx: float = 1, sy: float = 1, px: float = 0.5, py: float = 0.5) -> Self`
- **Descrição**: Modifica o composer *in-place* acumulando uma escala.
- **Retorno**: `self`.

#### `translate(x: int = 0, y: int = 0) -> Self`
- **Descrição**: Modifica o composer *in-place* acumulando uma translação $(x, y)$.
- **Retorno**: `self`.

#### `add_transform(transf: Transform, reference_size: tuple[int, int] = None) -> Self`
- **Descrição**: Aplica e acumula um objeto de intenção `Transform` neste composer.
- **Retorno**: `self`.

#### `copy() -> Self`
- **Descrição**: Cria uma cópia profunda e independente deste composer.
- **Retorno**: Novo `Composer`.

#### `copy_from(other: Self) -> None`
- **Descrição**: Copia o estado interno de outro composer para o objeto atual in-place.
- **Retorno**: `None`.

---

## 2. Cadeia de Intenções Imutáveis: `Transform`

A classe abstrata `Transform` (e suas derivadas `TransformRel` e `TransformAbs`) representa uma descrição imutável de operações de transformação desejadas (intenções). Ela é usada principalmente para definir transformações que serão aplicadas posteriormente via `layer.set_transform(transform)`.

### Construtores de Fábrica

#### `Transform.relative() -> TransformRel` *(Class Method)*
- **Descrição**: Cria uma cadeia de transformações utilizando coordenadas de pivô relativas (onde `0.5, 0.5` representa o centro da camada).
- **Retorno**: `TransformRel`.

#### `Transform.absolute() -> TransformAbs` *(Class Method)*
- **Descrição**: Cria uma cadeia de transformações utilizando coordenadas de pivô absolutas em pixels.
- **Retorno**: `TransformAbs`.

---

### Principais Métodos de `Transform` (`TransformRel` / `TransformAbs`)

#### `rotate(angle: float = 0, pivot_x: float = 0.5, pivot_y: float = 0.5) -> Self`
- **Descrição**: Adiciona uma intenção de rotação à cadeia e retorna um **novo** objeto `Transform` (imutabilidade).
- **Retorno**: Nova instância de `Transform`.

#### `scale(sx: float = 1, sy: float = 1, pivot_x: float = 0.5, pivot_y: float = 0.5) -> Self`
- **Descrição**: Adiciona uma intenção de escala à cadeia e retorna uma nova instância.
- **Retorno**: Nova instância de `Transform`.

#### `translate(x: int = 0, y: int = 0) -> Self`
- **Descrição**: Adiciona um deslocamento de translação $(x, y)$ à cadeia de transformações.
- **Retorno**: Nova instância de `Transform`.

#### `get_matrix(size: tuple[int, int] = (0, 0)) -> np.ndarray`
- **Descrição**: Converte toda a cadeia de intenções acumuladas em uma única matriz 3x3 resultante multiplicando as matrizes de translação e distorção.
- **Retorno**: `np.ndarray` — Matriz 3x3.

#### `create_composer(size: tuple[int, int]) -> Composer`
- **Descrição**: Instancia a versão mutável (`ComposerRel` ou `ComposerAbs`) compatível com esta transformação.
- **Retorno**: `Composer`.

---

## 3. Funções Principais de Matrizes

#### `mat_global(layer: Layer) -> np.ndarray`
- **Descrição**: Calcula a matriz de transformação 3x3 global/absoluta de uma camada ou grupo no espaço do Canvas:
  $$\mathbf{M}_{\text{global}} = \mathbf{M}_{\text{parent}} \cdot \mathbf{M}_{\text{position}} \cdot \mathbf{M}_{\text{transform}}$$
- **Retorno**: `np.ndarray` — Matriz 3x3 de tipo `float32`.

#### `mat_inverse(matrix: np.ndarray) -> np.ndarray`
- **Descrição**: Calcula a matriz inversa ($\mathbf{M}^{-1}$). Essencial para projetar pixels do espaço global de volta para o espaço local da imagem.
- **Retorno**: `np.ndarray`.

#### `mat_translation(x: float, y: float) -> np.ndarray` / `mat_rotation(angle: float)` / `mat_scale(sx: float, sy: float)`
- **Descrição**: Funções auxiliares que constroem as matrizes homogêneas elementares 3x3.

#### `create_pivot_transform_rel(...)` / `create_pivot_transform_abs(...)`
- **Descrição**: Constrói o "Sanduíche de Pivô" matricial: transporte do pivô para a origem, aplicação da transformação pura e transporte de volta:
  $$\mathbf{M} = \mathbf{T}_{\text{pos}} \cdot \mathbf{P}_{\text{pos}} \cdot \mathbf{R} \cdot \mathbf{P}_{\text{neg}} \cdot \mathbf{T}_{\text{neg}}$$
- **Retorno**: `np.ndarray`.
