# Guia da Janela de Visualização — Classe `Viewport` (`viewport.py`)

O módulo `anicrop.viewport` define a abstração da câmera/janela de visualização (`Viewport`), responsável por gerenciar a escala de zoom, o deslocamento da câmera (*pan*) e o enquadramento do Canvas na área de exibição da tela.

---

## 1. `Viewport` (`anicrop.viewport.Viewport`)

A classe `Viewport` representa o retângulo da janela de exibição (ex: um painel de UI de $800 \times 600$ pixels) e calcula as matrizes de projeção para mapear as coordenadas do `Canvas` para a tela.

---

### Principais Métodos e Propriedades de `Viewport`

#### `__init__(size: tuple[float, float], fit_scale: float = 1.0, canvas: AbstractCanvas | None = None, bg_color: tuple[int, int, int, int] = (204, 204, 204, 255))`
- **Descrição**: Inicializa a Viewport com a dimensão em pixels da janela de visualização, o fator inicial de ajuste, o Canvas observado e a cor de fundo padrão. Se nenhum canvas for fornecido, cria um Canvas padrão com a dimensão da Viewport.
- **Parâmetros**:
  - `size` (`tuple[float, float]`): Dimensão da janela de exibição `(width, height)`.
  - `fit_scale` (`float`): Escala inicial de enquadramento (padrão `1.0`).
  - `canvas` (`AbstractCanvas | None`): Instância do Canvas/Superfície observado. Padrão: `None`.
  - `bg_color` (`tuple[int, int, int, int]`): Cor de fundo em formato RGBA/BGRA `(R, G, B, A)`.
- **Retorno**: Instância de `Viewport`.

#### `set_canvas(canvas: AbstractCanvas) -> None`
- **Descrição**: Define ou troca o Canvas observado pela Viewport. Exige obrigatoriamente uma instância de `AbstractCanvas` (ou `Canvas`).
- **Parâmetros**:
  - `canvas` (`AbstractCanvas`): O Canvas a ser vinculado.

#### `@property canvas_size -> Point`
- **Descrição**: Retorna a dimensão `(width, height)` do Canvas atualmente vinculado à Viewport em tempo real como um `Point`.
- **Retorno**: `Point`.

#### `@property size -> Point`
- **Descrição**: Retorna a dimensão `(width, height)` da janela de exibição da Viewport como um `Point`.
- **Retorno**: `Point` (subtipo de `tuple[float, float]`).

#### `@property top_left -> Point`
- **Descrição**: Retorna as coordenadas `(x, y)` do canto superior esquerdo da região da Viewport como um `Point`.
- **Retorno**: `Point` (subtipo de `tuple[float, float]`).

#### `@property scale_factor -> float`
- **Descrição**: Retorna o fator de escala combinado resultante do zoom do usuário multiplicado pela escala de ajuste (`scale.sx * _fit.sx`).
- **Retorno**: `float`.

#### `@property region -> Region` / `@region.setter`
- **Descrição**: Acessa ou altera a `Region` retangular da Viewport.
- **Retorno**: `Region`.

#### `@property scale -> Scale` / `@scale.setter`
- **Descrição**: Acessa ou altera o objeto `Scale` representando o zoom atual da câmera.
- **Retorno**: `Scale`.

#### `@property roi_matrix -> ndarray`
- **Descrição**: Calcula e retorna a matriz de transformação 3x3 para a Região de Interesse (ROI - Region of Interest), combinando a escala com a translação do topo-esquerdo: `mat_pivot(scale, size) @ mat_translation(-x, -y)`.
- **Retorno**: `np.ndarray` (matriz 3x3 `float32`).

#### `fit_matrix() -> ndarray`
- **Descrição**: Calcula a matriz de transformação necessária para encolher/expandir e centralizar perfeitamente o Canvas vinculado dentro do espaço da Viewport.
- **Retorno**: `np.ndarray` — Matriz 3x3 combinando a escala de ajuste (`_fit`) com a translação de centralização `(offset_x, offset_y)`.

#### `roi(region: Region) -> Region`
- **Descrição**: Mapeia e projeta uma `Region` do espaço do Canvas de volta para as coordenadas de exibição da Viewport aplicando a matriz inversa de ROI.
- **Parâmetros**:
  - `region` (`Region`): Região no espaço do Canvas.
- **Retorno**: `Region` — A região correspondente no espaço da Viewport.
