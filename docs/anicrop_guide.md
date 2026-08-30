# Guia Geral do `anicrop` — Document, Viewer e Layout

O `anicrop` é um motor/biblioteca gráfica 2D em Python voltado para manipulação, composição não-destrutiva e renderização de camadas de imagem com suporte a transformações bidimensionais complexas e histórico reativo de ações (Undo/Redo).

Este guia cobre os pontos de entrada principais do fluxo de trabalho: a classe Facade `Document`, a propriedade de enquadramento `Layout` e o visualizador interativo `Viewer`.

---

## 1. `Document` (`anicrop.document.Document`)

A classe `Document` atua como a **Fachada (Facade) principal** da biblioteca. Ela gerencia o `Canvas` (dimensões de trabalho), o `GlobalHistory` (histórico de ações desfazíveis via proxies), a pilha de camadas (`LayerStack`) e os renderizadores de cena (`CanvasRender` e `ViewportRender`).

### Políticas de Documento (`DocumentPolicy`)
O `Document` opera sob duas políticas de execução:
- **`ReactiveDocumentPolicy` (`history=True`, Padrão)**: Encapsula camadas em Proxies (`ProxyLayer`, `GroupProxy`) permitindo histórico reativo e controle de estado desfazível (`Undo`/`Redo`).
- **`DirectDocumentPolicy` (`history=False`)**: Opera diretamente sobre os objetos de camada puros sem overhead de proxies, ideal para processamento em lote de alta performance.

---

### Principais Métodos e Propriedades de `Document`

#### `__init__(name: str, width: int, height: int, history: bool = True)`
- **Descrição**: Cria uma nova instância de documento configurando o `Canvas` com as dimensões especificadas (`width` x `height`) e inicializando a pilha de camadas (`stack`) e o histórico (`history`) de acordo com o parâmetro `history`.
- **Parâmetros**:
  - `name` (`str`): Nome identificador do documento.
  - `width` (`int`): Largura em pixels do Canvas.
  - `height` (`int`): Altura em pixels do Canvas.
  - `history` (`bool`): Se `True` (padrão), ativa o encapsulamento reativo via Proxies com suporte a Undo/Redo.
- **Retorno**: Instância de `Document`.

#### `open(path: str | Path, name: str, opacity: float = 1.0, blend_mode: BlendMode = BlendMode.NORMAL, history: bool = True, format: ImageFormat = ImageFormat.RGBA, bg_color: tuple[int, ...] | None = None, backend: AbstractImageIO | str | None = None) -> Document` *(Class Method)*
- **Descrição**: Construtor de conveniência que abre uma imagem do disco, cria o `Document` ajustado exatamente ao tamanho da imagem e a insere como primeira camada da pilha com o nome fornecido.
- **Parâmetros**:
  - `path` (`str | Path`): Caminho do arquivo de imagem base.
  - `name` (`str`): Nome identificador obrigatório da camada/documento.
  - `opacity` (`float`): Opacidade inicial da camada (padrão 1.0).
  - `blend_mode` (`BlendMode`): Modo de mesclagem (padrão `NORMAL`).
  - `history` (`bool`): Habilita/desabilita a política de histórico e proxies reativos.
  - `format` (`ImageFormat`): Formato de cores desejado (padrão `RGBA`).
  - `bg_color` (`tuple[int, ...] | None`): Cor de fundo do Canvas.
  - `backend` (`AbstractImageIO | str | None`): Backend de I/O específico (`"vips"`, `"opencv"` ou instância).
- **Retorno**: `Document` — Nova instância do documento dimensionada pela imagem.

#### `add(layer: LayerT) -> LayerT`
- **Descrição**: Adiciona um `Layer` ou `GroupLayer` existente à pilha do documento respeitando a política reativa atual. Valida estritamente a unicidade do nome da camada no documento (lança `ValueError` se já existir uma camada com o mesmo nome).
- **Parâmetros**:
  - `layer` (`Layer | GroupLayer`): A camada ou grupo a ser adicionado.
- **Retorno**: `Layer | GroupLayer` — A camada (ou Proxy correspondente) adicionada.

#### `add_group(name: str) -> GroupLayer`
- **Descrição**: Cria e adiciona um novo grupo de camadas (`GroupLayer`) à pilha do documento com nome obrigatório.
- **Parâmetros**:
  - `name` (`str`): Nome identificador único do grupo.
- **Retorno**: `GroupLayer` (ou `GroupProxy`) — O grupo criado e inserido na pilha.

#### `load_layer(path: str | Path, name: str, opacity: float = 1.0, blend_mode: BlendMode = BlendMode.NORMAL, format: ImageFormat = ImageFormat.RGBA, backend: AbstractImageIO | str | None = None) -> Layer`
- **Descrição**: Carrega uma imagem do disco no formato especificado e a adiciona diretamente à pilha de camadas do documento com nome obrigatório.
- **Parâmetros**:
  - `path` (`str | Path`): Caminho do arquivo de imagem no disco.
  - `name` (`str`): Nome identificador único da camada.
  - `opacity` (`float`): Opacidade inicial (0.0 a 1.0).
  - `blend_mode` (`BlendMode`): Modo de mesclagem (padrão `NORMAL`).
  - `format` (`ImageFormat`): Formato de cor desejado (padrão `RGBA`).
  - `backend` (`AbstractImageIO | str | None`): Backend de I/O específico.
- **Retorno**: `Layer` (ou `ProxyLayer`) — A camada carregada e adicionada.

#### `find(name: str, recursive: bool = True) -> BaseLayer | None`
- **Descrição**: Realiza busca por nome na hierarquia de camadas do documento.
- **Parâmetros**:
  - `name` (`str`): Nome da camada procurada.
  - `recursive` (`bool`): Se `True`, busca também recursivamente dentro de grupos aninhados.
- **Retorno**: `BaseLayer | None` — A camada encontrada ou `None`.

#### `remove(layer_or_name: BaseLayer | str) -> None`
- **Descrição**: Remove uma camada da hierarquia do documento, aceitando tanto o objeto da camada quanto o seu nome (`str`).
- **Parâmetros**:
  - `layer_or_name` (`BaseLayer | str`): Instância ou nome da camada a ser removida.
- **Retorno**: `None`.

#### Protocolo de Coleção / Mapeamento
O `Document` implementa o protocolo completo de contêiner Python para manipulação intuitiva das camadas:
- **`len(doc)`**: Retorna o total de camadas na raiz da pilha.
- **`doc[0]` / `doc[-1]`**: Acesso por índice inteiro à pilha (índice `0` é a camada mais no topo/frente).
- **`doc['fundo']`**: Acesso direto e polimórfico pelo **nome da camada** (lança `KeyError` se não existir).
- **`'fundo' in doc` / `layer in doc`**: Verifica pertencimento de camada por nome ou objeto.
- **`del doc[0]` / `del doc['fundo']`**: Remove a camada da pilha.
- **`for layer in doc:`**: Itera pelas camadas raiz da pilha.

#### `@property layout -> Layout`
- **Descrição**: Instância do motor de Layout para aplicar operações espaciais não-destrutivas (`fit`, `align`, `resize_bounds`, `fit_content`) diretamente nas camadas do documento.

#### `@property canvas -> Canvas`
- **Descrição**: Acessa o Canvas do documento, expondo `doc.canvas.width`, `doc.canvas.height`, `doc.canvas.size` e `doc.canvas.region`.

#### `@property canvas_render -> CanvasRender` / `@property viewport_render -> ViewportRender`
- **Descrição**: Acesso direto às instâncias de renderizador `CanvasRender` e `ViewportRender` internamente configuradas no documento.

#### `render(format: ImageFormat = ImageFormat.RGBA, interp: InterpMode = InterpMode.LANCZOS) -> Image`
- **Descrição**: Renderiza a composição final em alta resolução com base nas dimensões exatas do Canvas e retorna o objeto [`Image`](file:///home/gui/python/anicrop/docs/image.md) resultante.
- **Retorno**: `Image`.

#### `preview(viewport: Viewport, format: ImageFormat = ImageFormat.RGBA, interp: InterpMode = InterpMode.LANCZOS) -> Image`
- **Descrição**: Renderiza a cena atual visível dentro da janela de observação (`Viewport`) informada e retorna o objeto [`Image`](file:///home/gui/python/anicrop/docs/image.md).
- **Parâmetros**:
  - `viewport` (`Viewport`): Janela contendo dimensões, zoom e posição da câmera.
- **Retorno**: `Image`.

#### `export(path: str | Path, format: ImageFormat = ImageFormat.RGBA, interp: InterpMode = InterpMode.LANCZOS, options: SaveOptions | None = None, backend: AbstractImageIO | str | None = None) -> None`
- **Descrição**: Renderiza a composição final em alta resolução e salva diretamente no arquivo de imagem especificado via `doc.render().save(path, options=options, backend=backend)`.
- **Parâmetros**:
  - `path` (`str | Path`): Caminho do arquivo de saída (ex: `"output.png"`).
  - `format` (`ImageFormat`): Formato da composição renderizada.
  - `interp` (`InterpMode`): Algoritmo de interpolação.
  - `options` (`SaveOptions | None`): Opções de compressão e qualidade de codificação.
  - `backend` (`AbstractImageIO | str | None`): Backend de I/O específico.
- **Retorno**: `None`.

---

## 2. `Viewer` (`anicrop.viewer.Viewer`)

A classe `Viewer` oferece um visualizador gráfico interativo embutido baseado em OpenCV (`cv2`), permitindo inspecionar o resultado da renderização de um `Document` através de uma `Viewport` em tempo real.

---

### Principais Métodos de `Viewer`

#### `__init__(doc: Document, viewport: Viewport)`
- **Descrição**: Inicializa a janela OpenCV com tamanho fixo (`cv2.WINDOW_AUTOSIZE`) utilizando o nome do documento como título.
- **Parâmetros**:
  - `doc` (`Document`): O documento que será renderizado.
  - `viewport` (`Viewport`): A janela de observação que define o tamanho da exibição.
- **Retorno**: Instância de `Viewer`.

#### `fit_canvas() -> None`
- **Descrição**: Calcula e aplica automaticamente o fator de escala (`fit_scale`) necessário para que todo o Canvas do documento caiba centralizado e inteiramente dentro da dimensão da `Viewport`.
- **Retorno**: `None`.

#### `show() -> None`
- **Descrição**: Invoca `doc.preview(self.viewport)`, extrai o buffer de pixels e atualiza a janela OpenCV convertendo para o espaço de cores do OpenCV (BGRA/BGR).
- **Retorno**: `None`.

#### `wait(delay: int = 0) -> int`
- **Descrição**: Bloqueia a execução aguardando o pressionamento de uma tecla por um número especificado de milissegundos (ou indefinidamente se `delay=0`).
- **Parâmetros**:
  - `delay` (`int`): Tempo em milissegundos para aguardar.
- **Retorno**: `int` — Código ASCII da tecla pressionada.

#### `close() -> None`
- **Descrição**: Fecha e destrói explicitamente a janela OpenCV criada para o visualizador.
- **Retorno**: `None`.
