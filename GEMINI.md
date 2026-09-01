# GEMINI — anicrop

> **Projeto:** `anicrop` — Motor central (Core / Engine) para edição, manipulação e composição de imagens 2D não-destrutiva.

---

## 1. Visão Geral

- **Descrição curta:** Biblioteca/engine em Python para composição não-destrutiva de imagens 2D baseada em camadas (`Layer`, `GroupLayer`), suporte a transformações espaciais (matrizes homogêneas 3x3), mesclagem (*blend modes*), backend híbrido de memória (NumPy / Zarr para imagens gigantes) e renderização por patch.
- **Motivação:** Fornecer um *backend* de edição gráfica robusto, matematicamente preciso e de alta performance que possa alimentar scripts de automação complexos ou servir de motor gráfico para interfaces de usuário (GUIs).
- **Status Atual:** Desenvolvimento ativo. Estrutura de transformações, contêineres compostos, proxies reativos para histórico (Undo/Redo), suporte a Zarr/LOD em `EditLayer` e renderização via `CanvasRender` e `ViewportRender` estabelecidas. A abordagem atual do módulo `Layout` será reformulada.

---

## 2. Público e Uso Pretendido

- **Público-alvo:** Desenvolvedores, criadores de ferramentas gráficas, automação de processamento de imagens e o próprio autor.
- **Tipo de projeto:** Estritamente um Pacote / Biblioteca (*Core Library*).
- **Uso:** Importação direta em scripts Python ou integração como motor de processamento em aplicações GUI / Web. Não há escopo para CLI ou GUI nativa neste repositório.

---

## 3. Escopo e Recursos Principais

### O que o projeto FAZ (Core Features):
- **Gerenciamento Hierárquico de Camadas:** Árvore espacial utilizando o padrão *Composite* (`LayerStack`, `GroupLayer`, `Layer`, `EditLayer`).
- **Navegação e Matrizes Relativas:** Protocolo de nós (`NodeContainerProtocol`) permitindo recomposição de coordenadas pai-filho (`parent`, `_parent_inverse`).
- **Transformações Espaciais de Alta Precisão:** Matrizes 3x3 homogêneas para Rotação, Escala, Translação e "Sanduíches de Pivô" sem acúmulo de erro de arredondamento (*Size Drift*).
- **Edição Não-Destrutiva e Patches:** Fila de edições locais (`EditLayer`) preservando os pixels originais da imagem.
- **Backend Híbrido de Imagem & LOD:** Chaveamento transparente de dados de imagem (`Image`) entre `numpy.ndarray` (memória) e `zarr.core.Array` (disco) para imagens gigantes ($\ge 8192\text{px}$), com pirâmide de nível de detalhe (*Level of Detail* - LOD) em `EditLayer`.
- **Fachada & Histórico Reativo:** Classe Facade `Document` oferecendo políticas reativas com histórico (`GlobalHistory` via `ProxyLayer` e `GroupProxy`) ou modo direto de alta performance (`DirectDocumentPolicy`).
- **Motor de Renderização:** Renderização por patch (`ViewportRender` para previews interativos e `CanvasRender` para exportações finais em alta resolução) e visualizador OpenCV (`Viewer`).

### Escopo Futuro (Roadmap):
- **Filtros e Ajustes de Cor:** Suporte planejado para aplicação de filtros e camadas de ajuste de cor sobre o pipeline de composição.
- **`Layout.fit_content` com Crop / Máscaras / Edits:** Ajustar e padronizar o comportamento de `Layout.fit_content` quando a camada alvo possuir recortes (`Content.crop` via `BlendMode.CLIP`), patches de `EditLayer` ou `Mask` ativa, garantindo que o cálculo de *bounding box* considere os limites efetivos de transparência/visibilidade resultantes.

### O que o projeto NÃO faz:
- Não provê Interface Gráfica (GUI) nativa ou Linha de Comando (CLI).
- Não faz preenchimento generativo (*inpainting*) de áreas vazias.

---

## 4. Stack Tecnológica e Decisões Arquiteturais

- **Linguagem:** Python 3.12+ (gerenciado via `uv`).
- **Dependências Principais:** `numpy`, `opencv-python`, `pyvips`, `zarr`, `pillow`, `loguru`, `pytest`.
- **Padrões de Projeto Aplicados:** Facade (`Document`), Composite (`GroupLayer` / `Container`), Proxy (`ProxyLayer`, `GroupProxy`), Command/Memento (`GlobalHistory`).

### Estrutura das Abstrações Principais:
1. **`BaseLayer`**: Classe abstrata herdada por `Layer` e `GroupLayer`. Provê opacidade, blend mode, visibilidade, `_parent_inverse` e a propriedade `.transform` (instância de `ComposerRel` pronta no `__init__`).
2. **`Composer` (`ComposerRel` / `ComposerAbs`)**: Estado mutável acumulado de transformações na camada. Permite encadeamento direto: `layer.transform.rotate(45).scale(2, 2).translate(10, 0)`.
3. **`Transform` (`TransformRel` / `TransformAbs`)**: Descreve uma cadeia imutável de intenções puras de transformação.
4. **`Region` & `Span`**: Primitivas espaciais 2D imutáveis. Oferecem interseção global (`&`) e interseção local (`overlap_with` para fatiamento de matrizes NumPy).
5. **`Document` & `Viewer`**: Fachada principal e visualizador gráfico OpenCV para inspecionar `Viewport`s interativamente.

---

## 5. Índice da Documentação em `docs/`

Para detalhes de métodos, tipos de retorno e exemplos de uso de cada classe, consulte os guias em `docs/`:

- **[docs/anicrop_guide.md](file:///home/gui/python/anicrop/docs/anicrop_guide.md)** — Guia de `Document` e `Viewer`.
- **[docs/layout.md](file:///home/gui/python/anicrop/docs/layout.md)** — Operações espaciais do motor de `Layout` (`fit`, `align`, `resize_bounds`, `fit_content`).
- [docs/content.md](file:///home/gui/python/anicrop/docs/content.md) — Operações de manipulação e transformação de pixels/conteúdo (`crop`, `resize`, `fit`, `flip_x`, `flip_y`, `FitContext`).
- [docs/blend.md](file:///home/gui/python/anicrop/docs/blend.md) — Modos de mesclagem (`BlendMode`), `SOLID_FILL` vs `HARD_MASKING` e aceleração Cython/OpenMP.
- [docs/composition.md](file:///home/gui/python/anicrop/docs/composition.md) — Composição de camadas, agrupamento (`merge`), rasterização (`flatten`) e clonagem profunda (`clone_node`, `LayerComposition`).

- [docs/layer.md](file:///home/gui/python/anicrop/docs/layer.md) — Detalhes de `BaseLayer`, `Layer` e `EditLayer`.
- **[docs/spatial.md](file:///home/gui/python/anicrop/docs/spatial.md)** — Operações de geometria 2D e uso da classe `Region`.
- **[docs/container.md](file:///home/gui/python/anicrop/docs/container.md)** — Estrutura de `LayerStack`, `GroupLayer` e `NodeContainerProtocol`.
- **[docs/transform.md](file:///home/gui/python/anicrop/docs/transform.md)** — Matrizes 3x3, `Composer` mutável e intenções `Transform`.
- **[docs/image.md](file:///home/gui/python/anicrop/docs/image.md)** — Manipulação de pixels com `Image`, NumPy, subsistema `anicrop.io` e backend Zarr/LOD.
- **[docs/viewport.md](file:///home/gui/python/anicrop/docs/viewport.md)** — Projeções de câmera e janela de exibição `Viewport`.
- **[docs/benchmark.md](file:///home/gui/python/anicrop/docs/benchmark.md)** — Métricas oficiais de estresse de renderização, I/O e freeze de matrizes.

---

## 6. Testes, Qualidade e Regras de Interação com a IA (GEMINI)

- **Desenvolvimento Orientado a Testes (TDD):** A suíte de testes (`pytest`) é a fonte de verdade absoluta para validação de geometria, renderização e estado. A IA deve propor e executar cenários de teste antes/durante refatorações.
- **Formatação Automática (ruff format):** Sempre que a IA for autorizada a alterar, criar ou refatorar qualquer arquivo Python (`.py`), DEVE obrigatoriamente executar:
  ```bash
  uv run ruff format <arquivo.py>
  ```
- **Arquivos Temporários e Scratch:** Scripts de teste temporários ou de debug DEVEM ser gerados em `scratch/` ou `scripts/`, nunca na raiz do projeto.

### 6.1. Diretrizes Estritas para Criação de Testes (Pytest):
1. **Docstring Concisa:** Exatamente 1 linha limpa na primeira linha de cada função de teste.
2. **Zero Lógica Condicional (`if/else`):** Proibido `if/else` ou ternários no corpo do teste. O fluxo deve ser estritamente linear: *Arrange -> Act -> Assert*.
3. **Parametrização Declarativa (`@pytest.mark.parametrize`):** Variações de entrada e expectativa devem ser expressas como dados na tabela de parâmetros com IDs descritivos (`id="..."`).
4. **Helpers de Dados Dedicados:** Usar helpers reutilizáveis (`make_img`, `make_layer`, `make_solid_image`, `make_checkerboard_image`) para evitar código verboso de matrizes NumPy nos testes.
5. **Asserts Coesos:** Múltiplos asserts são permitidos somente quando pertencerem ao mesmo objeto sob teste e validarem facetas complementares do mesmo resultado.
6. **Casos de Borda Isolados:** Casos específicos (ex: pixel com transparência mínima ou fallbacks) devem ser testes dedicados, nunca misturados com `if` dentro de tabelas genéricas.

### 6.2. Padrão de Commits (Conventional Commits em Português):
- **Estrutura básica:**
  ```text
  tipo: descrição curta no imperativo

  [corpo opcional explicando o porquê]
  ```
- **Tipos mais comuns:**
  - `feat`: nova funcionalidade
  - `fix`: correção de bug
  - `docs`: documentação
  - `style`: formatação (sem mudar comportamento)
  - `refactor`: refatoração sem alterar funcionalidade
  - `test`: testes
  - `chore`: tarefas de manutenção/configuração
- **Corpo do commit (quando usar):**
  - Explique o porquê, não apenas o que foi feito.
  - Responder, se possível:
    - Qual era o problema?
    - Por que essa solução foi escolhida?
    - Existe impacto ou efeito colateral?



### 6.3. Fluxo de Git e Sincronização Multi-PC (`dev` <-> `main`):
- **Branch `dev` (Ambiente de Trabalho Ativo):** Contém todo o repositório rastreado (`GEMINI.md`, `docs/`, `planos/`, código e testes) para sincronização perfeita entre múltiplos computadores.
  - Enviar alterações de dev: `make push-dev` (ou `git push origin dev`).
  - Puxar no outro computador: `make pull-dev` (ou `git pull origin dev`).
- **Branch `main` (Produção e Distribuição Limpa):** Mantém estritamente os arquivos essenciais de código, testes, `README.md`, assets e build, sem poluição de rascunhos ou instruções de contexto.
  - Sincronizar código limpo para a main: `make sync-main` (puxa da dev estritamente os arquivos de produção e commita na main).
  - Publicar a main no GitHub: `make push-main`.

---

## 7. Status Atual e Decisões Consolidadas (Handoff)

- **Arquitetura do Módulo `Layout`:**
  - **Escopo Puramente Espacial ("Retrato" / Moldura):** O módulo `Layout` opera exclusivamente sobre a geometria espacial e o enquadramento lógico das camadas. **Não lida com manipulação de pixels, imagens ou máscaras.**
  - **Operações via `GeometryController` e `GeometryStrategy`:** Estratégias como `fit` e `resize_bounds` atuam gerenciando a `GeometryStrategy` no `GeometryController` (ex: `FitGeometry`). A `base.region` original permanece **intacta**, preservando a geometria estrutural e o pivô de rotação (imunidade contra o "Efeito Pêndulo").
  - **Projeção Global Unificada (`global_region`):** As referências e alinhamentos (`_resolve_region`, `align`, `fit_content`) utilizam a projeção em **Espaço Global** (`global_region` / `mat_global`), imunizando o sistema contra distorções por rotação, escala e *skew*.
  - **Protocolo Estrutural `LayoutStrategy(Protocol)`:** Tipagem estática elegante via *duck typing* estrutural no `_resolve_strategy`.

- **Arquitetura de Máscaras (`Mask` & `ProxyMask`):**
  - **1 Máscara Única por Camada:** `BaseLayer.set_mask(...)`, `BaseLayer.remove_mask()` e `@property mask -> Mask | None`.
  - **Indexação Direta e Micro-Snapshots:** `Mask` suporta mutação atômica via slices e `Region` (`mask[key] = data`), roteadas através do `ProxyMask` para `MaskCommand` com `MaskImageSnapshot` e `MaskStateSnapshot` gerenciando Undo/Redo com pegada mínima de memória.

- **Arquitetura de Efeitos e Filtros (`Effect`, `BoundEffect`, `BlurFilter`):**
  - **Protocolo `Effect` Puro (3 métodos):** `get_padding()`, `apply(image, matrix)` e `merge(other, matrix)`. Sem `prepare` e sem estado espacial interno (`self.matrix`).
  - **Envelope `BoundEffect`:** Ancla o efeito puro à matriz inversa da camada (`matrix`), calcula a matriz delta combinada no render ($\Delta M = M_{\text{render}} \cdot M_{\text{base\_inv}}$), modula por máscara opcional (`mask`) e controla visibilidade (`visible`).
  - **`BlurFilter` Anisotrópico:** Implementa desfoque Gaussiano/Box com fusão matemática exata de tensores de covariância 2D ($\Sigma_{\text{total}} = \Sigma_1 + \Sigma_2$).
  - **API em `BaseLayer`:** `add_effect` (livre), `bind_effect` (ancorado com matriz inversa), `remove_effect` e `@property effects -> tuple[Effect, ...]`.

- **Arquitetura de Manipulação de Conteúdo (`Content`, `ProxyContent`, `BlendMode.CLIP`):**
  - **Módulo Puro `Content`:** Fornece operações de corte e transformação de pixels/conteúdo (`crop`, `resize`, `fit`). `crop` atua via `LayerLayoutStrategy.fit` + máscara `EditLayer` com `BlendMode.CLIP` (preservando o formato original e cor branca sólida/transparente). `resize` e `fit` operam diretamente sobre matrizes afins (`target.transform`).
  - **Transações Atômicas (`MacroCommand` e `AtomicPolicy`):** `ProxyContent` intercepta chamadas públicas sob `with history.atomic(name):`, que injeta um `MacroCommand` no topo do histórico. Subcomandos disparados internamente durante a operação são absorvidos no mesmo macro-comando e selados no encerramento do bloco, garantindo **1 único Undo** para reverter a operação completa.

- **Arquitetura Modular de I/O de Imagens (`anicrop.io`):**
  - **Contratos Puros e Tipados:** [`AbstractImageIO`](file:///home/gui/python/anicrop/src/anicrop/interfaces/io.py) (`read`, `write`, `get_size`) e [`SaveOptions`](file:///home/gui/python/anicrop/src/anicrop/interfaces/io.py) para controle estrito de qualidade, compressão e transparência.
  - **Multi-Backend de Alta Performance:**
    - [`PyvipsBackend`](file:///home/gui/python/anicrop/src/anicrop/io/vips.py): Backend padrão nativo em C/SIMD com streaming e subamostragem `shrink` direta no decoder (speedup de **$58\times$ em WebP 1080p** e **$2.6\times$ em PNG 4K**).
    - [`OpenCVBackend`](file:///home/gui/python/anicrop/src/anicrop/io/opencv.py): Backend modular com funções puras especializadas e fallback automático caso a `libvips` não esteja instalada no sistema.
  - **Gerenciamento e Roteamento:** [`set_default_backend`](file:///home/gui/python/anicrop/src/anicrop/io/registry.py), [`get_default_backend`](file:///home/gui/python/anicrop/src/anicrop/io/registry.py) e seleção pontual por chamada (`Image.open(..., backend="opencv")`).

- **Arquitetura de Composição de Camadas (`anicrop.composition`):**
  - **Clonagem Profunda Isolada com Zero-Copy:** `clone_layer`, `clone_group` e `clone_node` duplicam inteiramente matrizes afins 3x3, compositores (`Composer`), buffers de máscara e instâncias de `BoundEffect`, compartilhando com segurança os buffers pesados de pixels (`EditLayer.image`) por referência.
  - **Agrupamento Não-Destrutivo (`merge`):** Encapsula cópias desacopladas de camadas em um `GroupLayer`, preservando filtros individuais, matrizes locais e cálculo dinâmico de limites via `GroupGeometry`.
  - **Rasterização Plana (`flatten`):** Utiliza `CanvasRender.render_container` para assar a cena completa em uma única imagem rasterizada dentro de um único `Layer` folha.
  - **Fachada Estática `LayerComposition`:** Expõe `.merge()`, `.flatten()` e `.clone()` como métodos puros e desacoplados.
  - **Serviço Acoplado `Combine` (`doc.combine`):**
    - `doc.combine.merge(target, name, count=1, remove_source=True)`: Mescla descendente com até `count` camadas visíveis abaixo.
    - `doc.combine.flatten(target, name, count=1, format=None, ...)`: Achata descendente herdando o `ImageFormat` da camada topo.
    - `doc.combine.bake(target: GroupLayer | str, name=None, ...)`: Assa os filhos internos de um `GroupLayer` e substitui o grupo por um `Layer` plano em seu container pai.
    - `doc.combine.bake_stack(name="Layer", ...)`: Método especialista para assar toda a pilha (`doc.stack`) em uma única camada plana.

- **Renderização Pura de Sequências e Properties do Document:**
  - **`CanvasRender.render_container`:** Renderiza sequências avulsas (`Sequence[BaseLayer]` ou `Container`) calculando o Canvas automaticamente pela união das `global_region` dos elementos renderizáveis (`layer.is_renderable`), ideal para pipelines de streaming/batch como o Stitcher.
  - **Properties de Render:** `doc.canvas_render` e `doc.viewport_render` expostas na fachada do `Document`.

- **Arquitetura Espacial e Geometria Contínua em `float` (`Point`, `Span`, `Region`):**
  - **Primitiva `Point(NamedTuple)`:** Subclasse direta e eficiente de `tuple[float, float]` com suporte a desempacotamento, igualdade com tolerância analítica (`math.isclose`) e conversão discreta `.to_int(mode="round"|"floor"|"ceil")`.
  - **Pipeline Contínuo e Imunidade a Size Drift:** Toda a álgebra de `Span`, `Region`, `Composer`, `Layout` e `Content` opera estritamente em ponto flutuante analítico (`float`). A quantização para inteiros ocorre exclusivamente na fronteira física de buffers de imagem (`Image`, `ScratchBuffer`).
  - **Padronização Estrita de Tipagem e Sobrecargas:** Sobrecargas polimórficas padronizadas com `@overload` sob `if TYPE_CHECKING:` (com `pass` em linha dedicada) e `@ovld` em `else:` para despacho dinâmico em tempo de execução (`Span.__init__`, `Content.fit`, `Layer.__init__`).

- **Fachada `Document` e Tipagem Estrita:**
  - **Parâmetro Semântico `history: bool = False`:** Configura `DirectDocumentPolicy` (padrão de alta performance) vs `ReactiveDocumentPolicy` (experimental).
  - **Tipagem Pura de Domínio:** Referências diretas a `LayerStack`, `BaseLayer`, `Layer`, `GroupLayer` e remoção limpa via protocolo de contêineres e `NullContainer`. Sobrecargas `@overload` em `Document.__getitem__` para inferência precisa.
  - **Qualidade de Código:** 100% de conformidade estrita no `mypy` (0 erros com `--check-untyped-defs`) e **958 testes passando** (0 falhas, 0 pulados) no `pytest`.

