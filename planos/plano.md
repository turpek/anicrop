# Master Plan: Anicrop

Este documento centraliza todos os objetivos arquiteturais, otimizações e o progresso estrutural do desenvolvimento do motor de renderização.

---

## 📋 Lista de Tarefas (Status Atual)

- [x] ~~1. Usar o `Zarr` para dar suporte a imagens grandes.~~
- [x] ~~2. Implementar uma classe que renderiza o `Layer` no espaço local do layer (`LayerRender`).~~
- [x] ~~3. Criar um novo sistema de LOD na classe `EditLayer` (`LODManager`).~~
- [ ] 4. Migrar para o sistema de tiles com a classe `Tile` compondo a classe `Layer`.
- [x] ~~5. Criar uma classe para renderizar o `Layer` no espaço da viewport (`ViewportRender`).~~
- [ ] 6. Alocar um único buffer para renderizar os edits no processo de população das tiles.
- [x] ~~7. Refatorar as conversões de `bbox` para `rect` (ex: `rect_to_region`).~~
- [x] ~~8. Otimização por Oclusão Conservadora (Early-Exit via _opacity_mask).~~
- [ ] 9. Recorte Opcional de Edição nos Limites da Camada (`_clip_to_parent`).
- [ ] 10. Implementar Abordagem Híbrida: Tiled Pass para Zoom In (Alta Resolução).
- [x] ~~11. Tratar Artefatos de Borda (Ringing do Lanczos).~~
- [x] ~~12. Refatorar o `Layout` para operar via `GeometryController` / `GeometryStrategy` (puramente espacial, sem `crop` de pixels).~~
- [x] ~~13. Criar teste para validar o dessincronismo entre o cache da máscara (`FitGeometry`) e a geometria estrutural (`base`) após mutação de coordenadas.~~
- [x] ~~14. Implementar Pipeline de Processamento de Pixels e Efeitos (Filtros, Ajustes de Cor e Tom).~~
- [x] ~~15. Decisão Arquitetural: Análise da aplicação de translação em `calculate_new_rect` e `calculate_region_rect` ao consumir `mat_global`.~~
- [x] ~~16. Consolidação Unificada de Frames (`BaseFrame`, `CanvasFrame`, `ViewportFrame`) e Separação entre `surface` e `view_region`.~~
- [x] ~~17. Validação e Correção da Máscara de Oclusão (`_opacity_mask` / Early-Exit) em Relação ao `surface_size`.~~
- [x] ~~18. Decisão Arquitetural: Natureza e Gerenciamento da Transformação em `BaseLayer` (Sincronização de Região no `Composer` via `sync_region`).~~
- [x] ~~19. (Otimizações Analíticas) Álgebra Afim 2D no Pipeline de Renderização (`warp_patch`, `calculate_new_corners`, `mat_inverse`).~~
- [x] ~~20. Padronizar o comportamento de `Layout.fit_content` quando a camada possui crop (`BlendMode.CLIP`), máscara ativa (`Mask`) ou patches de `EditLayer`.~~
- [x] ~~22. Implementar `ViewportLayoutStrategy` para gerenciar enquadramento, navegação e foco de câmera (`fit`, `align`, `fit_content`, `resize_bounds`).~~
- [x] ~~23. Padronizar herança de propriedades e comportamentos na rasterização plana de camadas (`flatten`, `Combine.flatten`, `Combine.bake`).~~
- [x] ~~24. Correção do Erro de Dimensão por Arredondamento Subpixel na Discretização de AABB (`warp_patch` vs `Image.__region_to_slice` / `hard_masking`).~~
- [ ] 25. Modos Avançados de Fusão e Composição para Fotografia e Transições Suaves (Multi-Band Blending e Feather Blending).
- [x] ~~26. Eliminação de Contaminação de Cor e Franja Escura nas Bordas em `warp_affine` e `warp_patch` (Padding Alpha-Aware e `ImageFormat.is_straight_alpha`).~~
- [ ] 27. (Resolução Dinâmica de Borda por Formato) Suporte a `border_mode` e `border_value` em `warp_affine`, `warp_perspective` e `warp_patch` (`BORDER_REPLICATE` para opacos vs `BORDER_CONSTANT` para alfa).
- [x] ~~28. Sistema de Cache de Camadas com Decorators e Renderização Incremental (`LayerCache`).~~
- [x] ~~29. Suporte Nativo a Formatos BGR e BGRA para Pipelines de Vídeo e Visão Computacional (Zero-Copy com OpenCV / Aniseek).~~
- [ ] 30. Consolidação e Integração Abrangente do Sistema de Histórico (Undo/Redo para Combine, Contêineres, Remoções Aninhadas e Filhos).
- [x] ~~31. Modificar a Referência das Camadas no Cache para Referência Fraca (`weakref` em `LayerCache._states`).~~
- [x] ~~32. Sistema de Invalidação mais Robusto para Efeitos via Inspeção de Bytecode de `apply` (`dis` no escopo exclusivo de `apply`).~~
- [x] ~~33. Sistema de Invalidação mais Robusto para Edits Usando Somente `visible` e `blend_mode` (`snapshot_edit`).~~
- [x] ~~34. Remoção do Método Obsoleto `offset` do `EditLayer`.~~
- [x] ~~35. Otimizações de Baixa Latência e Zero-Alloc na Invalidação do `LayerCache` (Comparação de Matriz por Bytes e `ListView`).~~

---








## 🏗 Detalhamento Arquitetural

### Zarr no sistema (Concluído)
O projeto provê uma fábrica que lida com `ndarray`, permitindo gerir arrays do ecossistema *Zarr* nativamente e lidar com grandes blocos de imagens pela RAM de forma particionada sem saturar a memória local.

### Renderização em Espaço Local e Viewport (Concluído)
- **O Desafio da Câmera:** Transformações e pivôs ocorrem matematicamente no espaço Absoluto (o canto esquerdo do retângulo original, mesmo após girado). O `ViewportRender` funciona como uma *Câmera*, traduzindo o que o usuário clica na UI (Tela 0,0) para o espaço da matriz inversa do motor (Matemática Absoluta).
- O `LayerRender` concentra a Composição Plana local. O cache é invalidado inteligentemente por matriz $2\times2$ (conteúdo re-amostrado) ou $3\times3$ (apenas posição transladada, reutilizando o buffer).
- Uso estrito de `view_region` (ROI Local) nos warpings para poupar CPU com o que não está visível.

### Implementação do LOD (Concluído)
Para não ler 50.000 pixels quando a tela mede apenas 1000, o `LODManager`:
- Calcula degraus baseados no zoom (escala $f$ da Viewport): $N = \lfloor -\log_2(f) \rfloor$.
- Acima de $1.0$, puxa a ROI original; abaixo aplica `INTER_AREA`.
- Uma matriz de escala (`m_adjust`) é enviada ao renderizador para equalizar os espaços projetivos.

### Padronização Geométrica (Concluído)
Ambiguidade resolvida. O ecossistema nomeia rigorosamente `rect` como `(X, Y, Largura, Altura)` e o termo `bbox` fica reservado às projeções absolutas extremas matemáticas de Bounding Boxes em tupla limpa `(x1, y1, x2, y2)` presentes no `calculate_new_corners`.

### Otimização por Oclusão Conservadora (Early-Exit) (Concluído)
- Varredura de desenho feita no formato `Front-to-Back` (De cima para baixo).
- Cria-se uma miniatura crua (`_opacity_mask` $32 \times 32$) usando *Min-pooling* do Alpha.
- Se o `LayerRender` detecta um layer `BlendMode.NORMAL`, opacidade `1.0`, e cuja miniatura diz que aquela sub-região da tela é $100\%$ sólida (255), os cálculos dos layers de fundo são cancelados e abortados (Culling agressivo).

### Tratar Artefatos de Ringing (Lanczos Boundary) (Concluído)
O Overshoot causado pelo kernel Sinc gera linhas fantasmas ("Edge Ringing"). As opções de downscaling para a `Viewport` delegam a suavização a kernels neutros (`CUBIC` ou `AREA`), preservando o `LANCZOS` opcionalmente para exports nativos de alta fidelidade isolada.

---

## 🛠 Tarefas Futuras Pendentes

### A) Renderização Híbrida e Sistema de Tiles
Para exibir painéis imensos (gigapixels), o fluxo foi dividido em 2 braços, dos quais o segundo ainda precisa de implementação:
1. **Direct Pass (Zoom Out - Já resolvido pelo cache LOD):** Carrega-se a malha reduzida no tamanho da tela, fazendo warp direto.
2. **Tiled Pass (Zoom In - PENDENTE):** Quando a câmera aproxima perto do limite $1:1$, o motor deve invocar uma malha (Grid).
   - **O que fazer:** Criar uma classe `TileGrid`. Através dela, toda interseção visual ativa fatias virtuais de $512 \times 512$ na RAM.
   - **Alocação Inteligente:** Deve-se utilizar um **buffer reutilizável único**. O sistema mastiga a visibilidade pintando os ladrilhos necessários um de cada vez pelo pipeline, colando na Viewport sem criar explosões de RAM simultâneas.

### B) Recorte Opcional de Edição (`_clip_to_parent`)
Edições de patches menores que escapam dos limites reais do layer base (fundo original) podem re-aparecer erroneamente durante uma rotação de enquadramento.
- **O que fazer:** Ao acoplar a edição (`add_edit(..., clip_to_bounds=True)`), extrair a interseção global `self._region & self._clip_to_parent`. Esse retrato vira o limite fixo. A leitura será baseada nesse slicing para destruir fisicamente o resíduo vazado que flutuava fora do canvas.

### C) Redesign do `Layout`: Arquitetura Puramente Espacial baseada em `GeometryStrategy`

O módulo `Layout` opera estritamente sobre a geometria espacial e o enquadramento lógico das camadas ("retrato"/moldura), **sem manipular pixels, imagens ou máscaras**.

**Decisões Consolidadas de Arquitetura:**
1. **Remoção de `crop`:** O método `crop` foi totalmente **removido** do `Layout`. Operações de corte de pixels pertencem ao ecossistema de edições e máscaras (`EditLayer`), e não ao módulo de layout espacial.
2. **Estratégias via `GeometryController`:** Operações de adaptação (como `fit` e `resize_bounds`) passam a manipular a `GeometryStrategy` do `layout` no `GeometryController` (ex: via `FitGeometry`). A `base.region` original permanece **intacta**, preservando a geometria estrutural e o pivô de rotação (imunidade contra o "Efeito Pêndulo").
3. **Projeção em Espaço Global (`global_region`):** As referências e alinhamentos (`_resolve_region`, `align`, `fit_content`) utilizam a projeção em **Espaço Global** (`global_region` / `mat_global`), imunizando o sistema contra distorções por rotação, escala e *skew*.
4. **Polimorfismo em `GroupLayer`:** O `GroupLayer` integra-se via `GroupGeometry` / `GeometryStrategy`, calculando a Bounding Box projetada consolidada do conjunto.

#### 🚨 Tabela de Soluções Arquiteturais no `Layout`

| Método | Novo Comportamento Arquitetural | Garantia de Estabilidade |
| :--- | :--- | :--- |
| **`_resolve_region(ref)`** | Obtém a Bounding Box no Espaço Global (`ref.global_region`). | Imune a rotação/escala da referência. |
| **`Layer.crop`** | **REMOVIDO** do `Layout`. | Corte de pixels delegado para `EditLayer` e máscaras. |
| **`Layer.fit`** | Atualiza a `GeometryStrategy` no `GeometryController` (`FitGeometry`). | Preserva a `base.region` intacta (sem Efeito Pêndulo). |
| **`Layer.align`** | Alinha o retângulo projetado no Espaço Global. | Bordas visuais alinhadas perfeitamente em 0.0/0.5/1.0. |
| **`Layer.fit_content`** | Calcula a ROI global dos `_edits` e ajusta via `GeometryStrategy`. | Preserva o eixo do pivô de rotação. |
| **`Canvas.fit_content`** | Engloba o retângulo projetado ("diamante") no Espaço Global. | Sem amputação visual ou fundo transparente. |
| **`Group.*`** | Opera via `GroupGeometry` / `GeometryStrategy`. | Trata o grupo como um retângulo espacial único. |

### D) Teste de Dessincronismo de Cache (Concluído)
- **Status:** Validação da sincronia entre `FitGeometry`/`layout` e `base` garantida via `GeometryController.sync` e `GeometryControllerSnapshot`.

### E) Comportamento de `Layout.fit_content` com Crop, Máscaras e Edições
- **Objetivo:** Definir e padronizar o comportamento do algoritmo de enquadramento de conteúdo (`Layout.fit_content`) quando a camada alvo possuir:
  1. Recortes não-destrutivos (`Content.crop` via `BlendMode.CLIP`).
  2. Patches em fila de `EditLayer`.
  3. Máscara de camada ativa (`Mask`).
- **Garantias Técnicas Necessárias:**
  - **ROI Efetiva:** O cálculo de *bounding box* através de `calculate_content_rect` deve considerar os limites efetivos de transparência/visibilidade resultantes da composição dos patches e máscaras.
  - **Não-Expansão por Pixels Oclusos:** Pixels transparentes gerados por `BlendMode.CLIP` ou por `Mask` não devem expandir a Bounding Box calculada.
  - **Preservação de Geometria:** O enquadramento deve ser aplicado preservando a integridade da geometria estrutural (`base.region`) e do pivô de rotação natural.

### F) Subclasse de `EditLayer` e Controle de Visibilidade de Edições (`_flatten_edits` / Modelo GIMP)
- **Objetivo:** Implementar controle comutável de visibilidade em edições (`EditLayer.visible: bool = True` ou subclasse dedicada como `CropEditLayer`), viabilizando o modelo não-destrutivo com alternância de exibição e restauração transparente de camadas (estilo GIMP).
- **Comportamento e Diretrizes de Implementação:**
  1. **Pipeline de Renderização (`_flatten_edits` / `render_edit`):** Ignora edições com `visible = False`, evitando qualquer custo computacional de warp ou composição de pixels.
  2. **Toggling e Restauração de Crop:** Permite alternar a visibilidade do recorte (`crop_edit.visible = False`) para revelar a imagem base original completa sem remover a edição da fila ou poluir o histórico.
  3. **Integração com `fit_content`:** O cálculo de ROI filtra estritamente as edições ativas (`edit.visible`), permitindo alternar de forma previsível entre o enquadramento do recorte e o enquadramento da imagem base total.

### G) Estratégia de Layout para Viewport (`ViewportLayoutStrategy`)
- **Objetivo:** Integrar a `Viewport` como uma cidadã de primeira classe no sistema polimórfico de `Layout` (`Layout(viewport)` ou `viewport.layout`), controlando enquadramento, zoom e navegação de câmera de forma expressiva e desacoplada.
- **Natureza da Câmera (Display Fixo):**
  - Diferente de uma camada de imagem (onde o *fit* altera a moldura do objeto), na `Viewport` o tamanho físico da janela de exibição ($W_{\text{view}} \times H_{\text{view}}$) permanece **fixo**.
  - As operações de layout manipulam os controles de Câmera: **Zoom** (`viewport.scale`) e **Pan** (`viewport.region.top_left`).
- **Comportamento dos Métodos:**
  1. **`viewport.layout.fit(target)`:**
     - Calcula a escala uniforme $s = \min(W_{\text{view}} / W_{\text{target}}, H_{\text{view}} / H_{\text{target}})$, preservando o *aspect ratio*.
     - Aplica o zoom: `viewport.scale = Scale(s, s)`.
     - Aplica o pan para centralizar o alvo no meio da tela ($\Delta x = X_{\text{target\_centro}} - \text{canvas\_w}/2, \Delta y = Y_{\text{target\_centro}} - \text{canvas\_h}/2$).
  2. **`viewport.layout.align(target, anchor_x=0.5, anchor_y=0.5)`:**
     - Mantém o zoom atual intacto (`viewport.scale` inalterado).
     - Move apenas o Pan da câmera para apontar para a âncora especificada no Canvas.
  3. **`viewport.layout.fit_content()`:**
     - Enquadra toda a área útil de pixels visíveis da cena (`global_content_region`).
  4. **`viewport.layout.resize_bounds(new_w, new_h)`:**
     - Redimensiona a janela física da Viewport para $(new\_w, new\_h)$ preservando o ponto focal e o zoom atuais.

---

### ⚠️ Diretriz de Snapshot para Estratégias Futuras (Undo/Redo)
- **Preservação/Recálculo da Matriz Inversa:** Ao expandir a classe `GeometryControllerSnapshot` ou adicionar estratégias como `FitGeometry` / `CropGeometry`, certificar-se de salvar ou recriar a **matriz inversa** (`_local_matrix`) da estratégia restaurada para garantir alinhamento espacial perfeito.

---

## 🎭 Arquitetura de Máscaras: Dois Sistemas Distintos

O ecossistema do `anicrop` divide a responsabilidade das máscaras em dois sistemas totalmente independentes para otimização de performance e clareza de responsabilidades:

### 1. Máscara de Edição (`EditLayer` Level)
- **Atuação:** Atua no momento da edição (ferramentas de seleção, pinceladas, balde de tinta, operações de edição).
- **Comportamento:** A máscara recorta ou restringe a criação/modificação do retalho (`EditLayer`). O resultado é gravado diretamente no canal Alfa da própria imagem do patch.
- **Impacto no Renderizador:** **Nenhum.** O motor de renderização apenas compõe o `EditLayer` normalmente como uma imagem RGBA com transparência nativa.

### 2. Máscara Dinâmica de Camada (`BaseLayer` Level / Layer Mask)
- **Atuação:** Atributo pertencente ao `BaseLayer` (ex: `layer.mask`).
- **Comportamento:** Utilizada para ocultar ou revelar partes da camada de forma dinâmica e não-destrutiva, mantendo os pixels da imagem fonte 100% intactos.
- **Impacto no Renderizador:** **Atua na fase de renderização (`render.py`).** Durante o desenho, o renderizador multiplica o resultado visível da camada pelo fator dessa máscara dinâmica.

---

## 🎨 Arquitetura de Processamento de Pixels e Efeitos

Este trecho especifica o design da pipeline de processamento e aplicação não-destrutiva de efeitos de pixels (filtros, ajustes de cor, distorções e estilo) no ciclo de renderização do `anicrop`.

### 1. Interface dos Efeitos (`Effect` Protocol)

Os efeitos são representados por objetos funcionais e imutáveis que atendem ao protocolo formal `Effect`:

```python
from typing import Protocol, runtime_checkable
from anicrop.image import Image


@runtime_checkable
class Effect(Protocol):
    def apply(self, image: Image) -> Image:
        """Recebe o buffer RGBA atual e retorna um novo buffer Image processado."""
        ...

    def get_padding(self) -> tuple[int, int, int, int]:
        """Retorna a margem extra (top, right, bottom, left) necessária para efeitos de expansão de borda."""
        ...
```

### 2. Pontos de Injeção no Ciclo de Renderização

Os efeitos atuarão dinamicamente em dois momentos estratégicos do motor de renderização:

#### A. Pós-Renderização de `Layer` (`BaseRenderer.render_area`)
- **Momento:** Executado no `BaseRenderer.render_area` logo após achatar as edições geométricas (`EditLayer`s) no buffer RGBA da camada.
- **Escopo:** Aplica a fila de efeitos `layer.effects` exclusivamente sobre a área visível rasterizada (`dst_region`).
- **Objetivo:** Otimização de performance, evitando processar pixels fora da área visível ou em regiões não renderizadas.

#### B. Pós-Renderização de `GroupLayer` (`SceneTraverser.traverse`)
- **Momento:** Executado no `SceneTraverser.traverse` logo após compor e mesclar todos os elementos filhos de um grupo (`blend_rendered_images`).
- **Escopo:** Aplica a fila de efeitos `group.effects` sobre o buffer final consolidado do grupo.
- **Objetivo:** Permitir efeitos unificados sobre múltiplos elementos (ex: desfoque de profundidade de campo em todo um grupo de objetos).

### 3. Categorização de Operações em Pixels

A interface `Effect` abstrai e suporta quatro grandes famílias de operações de imagem:

1. **Ajustes de Cor e Tom (Operações Pontuais / Point Operations)**
   - Operações em que a cor de cada pixel é processada individualmente.
   - Inclui ajustes de Brilho, Contraste, Exposição, Gama, Matiz/Saturação (HSV), Curvas de Tom, Níveis e tabelas LUT (Look-Up Tables).

2. **Filtros Espaciais e Convolução (Neighborhood Operations / Kernels)**
   - Operações onde a cor de um pixel depende do bloco de pixels ao seu redor.
   - Inclui Desfoques (Gausseano, Motion Blur, Box Blur), Nitidez (Sharpen, High Pass) e Detecção de Bordas (Sobel, Laplacian).

3. **Distorção e Deslocamento de Pixels (Pixel Displacement / Warping)**
   - Operações que alteram o remapeamento geométrico dos pixels no espaço do buffer.
   - Inclui Displacement Maps (ondas, refração), Distorções de Lente (Fisheye, Aberração Cromática dividindo RGB) e Ondulações.

4. **Texturização, Estilo e Iluminação**
   - Efeitos compostos com ruído ou funções de distância.
   - Inclui Vinheta, Granulação de Filme, Vinheta Gradiente e Sombras.

> [!NOTE]
> **Nota Arquitetural sobre Margem e Expansão de Borda (Padding):**
> Efeitos espaciais como *Gaussian Blur*, *Drop Shadow* ou *Glow* espalham os pixels para fora do limite original da camada (`dst_region`).
> Para evitar que os pixels sejam cortados de forma abrupta nas bordas (*clipping*), o `BaseFrame` consultará a soma dos paddings retornados por `effect.get_padding()` de cada efeito ativo. Essa margem estendida será adicionada ao cálculo do `dst_region` antes da alocação da matriz `Image.new()`.

---

## 📐 Centralização e Semântica das Funções de Cálculo de Rect (`transform.py`)

No módulo `transform.py`, existem diversas funções utilitárias responsáveis por calcular a Bounding Box projetada (`rect`) de objetos e regiões através de matrizes homogêneas. Para evitar redundâncias e evitar confusões conceituais durante o desenvolvimento de novas estratégias, o plano prevê a simplificação do módulo e a declaração explícita de intenções.

### 1. Diagnóstico da Ambiguidade Atual
Existe uma diferença fundamental entre as duas formas de projeção retangular no motor:
- **Projeção partindo da Origem Zero (`calculate_new_rect`):** Assume que a transformação parte de `(0, 0)`. Ela é correta quando a matriz utilizada (como a matriz global da camada) já carrega internamente a translação de posição, ou quando se deseja apenas projetar dimensões puras.
- **Projeção preservando a coordenada inicial (`calculate_region_rect`):** Preserva o `top_left` original da região. Ela é indispensável em estratégias como `FitGeometry` ou `CropGeometry`, onde o retalho no Espaço Local possui uma origem diferente de `(0, 0)`.

O uso trocado dessas abordagens (por exemplo, aplicar a projeção com a coordenada inicial sobre uma matriz que já contém a posição embutida) resulta em uma translação duplicada, deslocando a camada para coordenadas incorretas no Canvas.

### 2. Proposta de Simplificação e Separação Conceitual
A proposta de arquitetura visa:
- **Centralização do Núcleo de Transformação:** Manter o cálculo geométrico centralizado em uma única função núcleo de transformação espacial, eliminando invólucros redundantes.
- **Declaração de Intenção Explícita:** Documentar de forma clara a intenção de cada função de conveniência, deixando explícito quando o chamador deve optar pela projeção partindo do zero (matriz com translação embutida) versus a projeção que preserva o deslocamento inicial da região.
- **Eliminação de Código Duplicado:** Reduzir a quantidade de funções utilitárias duplicadas no módulo `transform.py`, tornando a API interna do motor mais enxuta, legível e segura.

### 3. Ponto de Decisão: `mat_global` e Translação de `top_left` em `calculate_new_rect` vs `calculate_region_rect`

**Relato do Problema:**
- `calculate_new_rect(matrix, size)` projeta um retângulo partindo da origem `(0, 0)`.
- `calculate_region_rect(matrix, region)` projeta uma `Region` considerando suas coordenadas `top_left` (`region.x.start`, `region.y.start`).
- A função `mat_global(layer)` gera uma matriz homogênea $3 \times 3$ que **já contém embutida a translação da posição da camada** (`mat_position(layer.base.region)`).

Quando uma matriz com translação embutida (`mat_global`) é passada para `calculate_region_rect` em conjunto com uma `Region` que também contém coordenadas `top_left`, o deslocamento de posição é aplicado duplamente.

**Objetivo da Tarefa:**
Analisar e definir uma decisão arquitetural clara sobre como `calculate_new_rect` e `calculate_region_rect` devem tratar matrizes que já contêm a translação de `top_left` embutida, sem assumir previamente nenhuma correção ou alteração de código.

---

## 🖼️ Unificação e Consolidação da Hierarquia de Frames (`BaseFrame`, `CanvasFrame`, `ViewportFrame`)

Este trecho consolida a padronização e simetria de comportamento entre as classes de enquadramento (`CanvasFrame` e `ViewportFrame`) e o pipeline de renderização por patch (`render_image`, `CanvasRender` e `ViewportRender`), preparando a infraestrutura para suporte eficiente a **máscaras dinâmicas** e **grupos enquadrados**.

### 1. Diagnóstico da Assimetria Atual

Atualmente, existe uma divergência de assinaturas, tipos e expectativas entre as duas classes concretas de frame:
- **`CanvasFrame`:** Aceita na sua instanciação `view_region: Optional[Region | Canvas]`. O `CanvasRender` retorna uma `Image` com dimensões variáveis correspondentes à interseção entre a camada e a `view_region` (recorte estrito do patch).
- **`ViewportFrame`:** Aceita exclusivamente `viewport: Viewport`. O `ViewportRender` opera como uma câmera interativa, retornando uma `Image` com a dimensão física da tela/viewport (`viewport.size`).

Essa assimetria impede que `CanvasRender` e `ViewportRender` compartilhem o mesmo fluxo polimórfico de travessia e dificulta a aplicação de máscaras de grupo e de camada de forma otimizada.

### 2. Nova Separação Conceitual: `surface` vs `view_region`

A nova arquitetura divide a responsabilidade do enquadramento espacial em dois conceitos ortogonais e bem definidos:

1. **`surface: SurfaceProtocol` (Superfície Física de Destino):**
   - Representada por instâncias de `Canvas` ou `Viewport`.
   - **`surface.size`:** Determina a dimensão final da imagem/buffer gerado pelo renderizador (`dest_size`).
   - **`surface.region`:** Define a janela espacial física da superfície no espaço de coordenadas do Canvas.
   - > [!IMPORTANT]
   - > **Nota Arquitetural sobre o Buffer do `GroupLayer`:**
   - > O Buffer interno do `GroupLayer` **não implementa `SurfaceProtocol`**. Portanto, para instanciar um `BaseFrame` (ou derivado), é **sempre obrigatório fornecer um `Canvas` ou `Viewport`**, mesmo que seja uma superfície temporária.

2. **`view_region: Region | None` (Região de Interesse / Recorte Lógico):**
   - Representa uma restrição lógica adicional de visibilidade (ex: ROI de máscara de camada, máscara de grupo ou recorte de câmera).
   - Se omitida (`None`), assume por padrão a região física da própria superfície (`surface.region`).

### 3. Matemática e Comportamento Unificado de Renderização

Com essa separação, o comportamento de renderização passa a ser rigorosamente padronizado:
- **Conteúdo Visível Efetivo:** O conteúdo rasterizado é matematicamente a interseção trilateral:
  $$\text{conteúdo visível} = \text{layer.global\_region} \ \& \ \text{view\_region} \ \& \ \text{surface.region}$$
- **Dimensão da Imagem Gerada:** O renderizador gera sempre uma `Image` alinhada às dimensões do buffer da `surface` / frame.
- **Otimização por ROI (Patch Rendering):** O cálculo da matriz inversa (`mat_inverse`) determina o retângulo mínimo de origem (`src_region`) sobre os pixels da imagem fonte antes de executar o warp (`cv2.warpAffine`), processando estritamente os pixels que caem dentro da interseção visível e evitando renderizar áreas ocluídas ou mascaradas.

### 4. Roadmap de Execução da Tarefa

```
[FASE 1: Contrato dos Frames]
└── Atualizar assinatura unificada em BaseFrame, CanvasFrame e ViewportFrame.
    Aceitar (layer, surface: SurfaceProtocol, view_region: Optional[Region] = None, local: bool = False).

[FASE 2: Sincronização dos Renderers]
└── Adequar BaseRenderer, render_image, CanvasRender e ViewportRender para o novo fluxo.
    Retornar imagens com dimensões de superfície e conteúdo recortado pela interseção tripla.

[FASE 3: Travessia e Composição (SceneTraverser)]
└── Propagar view_region no SceneTraverser para compor filhos e grupos com ROI otimizado.

[FASE 4: Validação TDD]
└── Criar testes unitários e de integração cobrindo CanvasFrame, ViewportFrame, CanvasRender e ViewportRender.
```

---

## 🛡️ Validação e Correção da Máscara de Oclusão (`_opacity_mask` / Early-Exit) em Relação ao `surface_size`

Esta seção documenta a análise do potencial bug de falso positivo de oclusão no mecanismo de *Early-Exit* e define o requisito para testes específicos.

### 1. Diagnóstico do Potencial Bug

A otimização de oclusão conservadora opera gerando uma miniatura de $32 \times 32$ pixels (`_opacity_mask`) para cada camada rasterizada. O `SceneTraverser` acumula essas miniaturas na matriz global `miniview` ($32 \times 32$). Se toda a matriz atingir o valor $255$ (opaco), as camadas inferiores da pilha são descartadas (Early-Exit).

O cálculo de escala proporcional em `generate_opacity_mask` depende de `surface_size`:
```python
scale_x = target_size[0] / surface_size[0]
scale_y = target_size[1] / surface_size[1]
```

**O Risco Arquitetural:**
Se `surface_size` for erroneamente informado como o tamanho da própria camada (`bounds.size`) em vez do tamanho real da superfície (`canvas.size` ou `viewport.size`), a miniatura da camada será mapeada como se cobrisse $100\%$ do Canvas.

* **Exemplo do Bug:** Uma pequena camada opaca de $50 \times 50\text{px}$ posicionada em um Canvas de $1920 \times 1080\text{px}$ geraria uma `_opacity_mask` cobrindo toda a grade $32 \times 32$ com $255$. O `SceneTraverser` interpretaria incorretamente que o Canvas inteiro está coberto, descartando todas as camadas de fundo e gerando uma imagem final corrompida (fundo apagado).

### 2. Diretriz de Implementação e Teste Específico

1. **Garantia de Superfície:** O `BaseFrame` e seus derivados devem sempre alimentar `generate_opacity_mask` com o `surface_size` da superfície física real (`surface.size`), garantindo que camadas menores ocupem apenas a fração geométrica proporcional na grade $32 \times 32$.
2. **Criação de Teste de Regressão (TDD):**
   - Criar teste em `test_render.py` com um Canvas grande (ex: $1000 \times 1000$) e duas camadas:
     - Camada Topo: Pequena (ex: $100 \times 100$), opaca ($1.0$), posicionada no canto.
     - Camada Fundo: Tela cheia ($1000 \times 1000$), visível no restante da área.
   - Validar que o `SceneTraverser` **não** interrompe a renderização prematuramente e que a camada de fundo é devidamente renderizada nas áreas não cobertas pelo topo.

---

## 📐 Decisão Arquitetural: Natureza e Gerenciamento da Transformação em `BaseLayer`


Esta seção registra o diagnóstico sobre o ciclo de vida das transformações em `BaseLayer` e o acoplamento remanescente com o estado interno do `Composer`.

### 1. O Problema que foi Resolvido

Ao aplicar transformações geométricas com pivô relativo `(0.5, 0.5)` (como rotação e escala) em instâncias de `Layer` ou `GroupLayer` após uma redefinição de moldura/enquadramento (ex: via `Layout.fit` ou `resize_bounds`), o cálculo do centro de rotação utilizava as dimensões e a origem de `base.region` (pixels originais da imagem ou bounding box física dos filhos) em vez da região ativa de layout (`layout.region` / `self.region`).

* **Consequência do bug original:**
  - Em um `GroupLayer` com filhos ocupando $40 \times 40\text{px}$ em $(50, 50)$ que recebia um `Layout.fit` definindo moldura em $(0, 0, 100, 100)$, o pivô $(0.5, 0.5)$ calculava o centro em $(70, 70)$ dos filhos em vez de $(50, 50)$ da moldura.
  - A rotação ocorria de forma excêntrica e deslocava a `global_region` no Canvas, gerando desalinhamentos em relação à geometria pretendida.

### 2. O Problema Atual (Acoplamento e Acesso Indevido a `transform._region`)

Para que o cálculo do pivô relativo considere a moldura ativa de layout, a property `transform` da classe `BaseLayer` está atribuindo diretamente o atributo privado do `Composer`:
```python
@property
def transform(self) -> Composer:
    self._transform._region = self.region  # Atribuição direta em atributo privado
    return self._transform
```

* **Aspectos do problema técnico:**
  1. **Invasão de Encapsulamento:** `BaseLayer` depende de um atributo privado (`_region`) da classe `Composer`.
  2. **Ciclo de Vida do Composer vs. Geometria de Camada:** O objeto `Composer` armazena um estado geométrico imutável no momento da sua instanciação (`self._region = Region.from_size(*size)`), enquanto a geometria de `BaseLayer` é dinâmica e mutável via `GeometryController` (onde `self.region` pode mudar com adição/remoção de filhos em `GroupLayer` ou troca de estratégia para `FitGeometry` / `FitGroupGeometry`).
  3. **Necessidade de Decisão:** É necessário definir formalmente o contrato de sincronização entre a geometria ativa de uma camada (`GeometryController` / `GeometryStrategy`) e o referencial de dimensão/pivô utilizado pelo `Composer` para aplicar transformações relativas.

---

## 🥞 23. Padronização da Herança de Propriedades na Rasterização Plana (`flatten` / `merge down`)

Esta seção documenta a especificação para que a camada plana resultante (`flat_layer`) herde fielmente o comportamento das camadas compostas em relação ao restante do documento.

### 1. Herança de Modo de Mesclagem (`blend_mode`)
- **Merge Down de Camadas (`Combine.flatten`):**
  - Ao mesclar a camada alvo (`resolved_target`) com as camadas visíveis abaixo dela (`sequence`), a interação interna entre as camadas é assada nos pixels rasterizados.
  - A camada resultante `flat_layer` deve herdar o `blend_mode` da **camada base inferior** (`sequence[0].blend_mode`) para preservar como aquele bloco se mescla com o restante da pilha abaixo dele (comportamento padrão de editores como Photoshop/Krita).
- **Bake de Grupo (`Combine.bake` / `GroupLayer`):**
  - Ao assar os filhos internos de um `GroupLayer`, o `flat_layer` substituto deve herdar obrigatoriamente o `blend_mode` do próprio grupo (`group.blend_mode`).

### 2. Preservação e Inferência do Formato de Cor (`ImageFormat`)
- A função pura `flatten()` atualmente assume `ImageFormat.RGBA` como padrão fixo.
- **Novo Comportamento:** Quando `format=None`, deve inferir automaticamente o formato a partir da camada superior do conjunto (`sequence[-1].format` ou `layers[-1].format`), preservando formatos de cor especializados (`PRGBA`, `RGBX`, `RGB`) sem forçar conversão desnecessária para `RGBA`.

### 3. Preservação de Visibilidade e Opacidade
- **`visible`:** A camada resultante `flat_layer.visible` deve herdar o estado de visibilidade da camada alvo ou do grupo (`sequence[-1].visible` ou `group.visible`).
- **`opacity` em Grupos (`Combine.bake`):**
  - Se o `GroupLayer` possuir `opacity < 1.0`, essa opacidade global do grupo deve ser transferida para `flat_layer.opacity` (mantendo a rasterização interna dos filhos isolada), ou assada diretamente nos pixels caso especificado.

---

## 🐛 24. Correção do Erro de Dimensão por Arredondamento Subpixel na Discretização de AABB (`warp_patch` vs `Image.__region_to_slice` / `hard_masking`)

Esta seção documenta o diagnóstico da causa raiz matemática e a diretriz de solução para o erro de divergência de 1 pixel em transformações afins contínuas.

### 1. Diagnóstico e Causa Raiz Matemática

Ao rasterizar camadas com transformações afins fracionárias (rotação contínua como $15.5^\circ$ ou escalas subpixel como $1.033$), a Bounding Box (`Region`) possui coordenadas `start` e `end` fracionárias em ponto flutuante.

O erro de colisão de dimensões ocorre devido a uma inconsistência de cálculo entre o alocador de destino no warp (`warp_patch`) e o fatiamento de visualização do buffer (`Image.__region_to_slice`):

1. **Alocação no `warp_patch` (`render.py`):**
   ```python
   dest_size = (int(round(dest_region.width)), int(round(dest_region.height)))
   ```
   Calcula a dimensão a partir do comprimento contínuo: `round(length) = round(end - start)`.

2. **Fatiamento no `Image.__region_to_slice` (`image.py`):**
   ```python
   slice(int(round(region.x.start)), int(round(region.x.end)))
   ```
   Calcula a dimensão fatiada como: `round(end) - round(start)`.

Matematicamente, `round(end - start)` **não é estritamente igual** a `round(end) - round(start)`.
* **Exemplo real:**
  - `start = 100.4`, `end = 5425.7`, `length = 5325.3`.
  - `round(length) = 5325` $\rightarrow$ `warp_patch` cria imagem com largura **$5325$**.
  - `round(end) - round(start) = 5426 - 100 = 5326` $\rightarrow$ `buffer.view(region)` fatia com largura **$5326$**.
  - No `hard_masking` (ou qualquer rotina de blend estrita), `base.size (5326, 4773) != overlay.size (5325, 4773)` $\rightarrow$ **`ValueError: Size mismatch`**.

### 2. Diretriz Arquitetural de Solução

1. **Padronização da Discretização no `Image.__region_to_slice`:**
   O `__region_to_slice` utiliza diretamente `region.top_left.to_int()` e `region.size.to_int()`:
   ```python
   def __region_to_slice(self, region: Region) -> tuple[slice, slice]:
       x, y = region.top_left.to_int()
       w, h = region.size.to_int()
       return (
           slice(y, y + max(1, h)),
           slice(x, x + max(1, w)),
       )
   ```
2. **Garantia de Equivalência Dimensional:**
   O tamanho do slice do NumPy coincide com o tamanho da imagem rasterizada gerada por `warp_patch` e `Image.new(region.size)`.
3. **Testes de Regressão TDD:**
   Validados em `tests/test_render.py` cobrindo transformações fracionárias contínuas com `HARD_MASKING` e `SOLID_FILL`.

---

## 🎨 25. Modos Avançados de Fusão e Composição para Fotografia e Transições Suaves

Para além dos modos binários focados em animação (`HARD_MASKING` e `SOLID_FILL`), o pipeline de mesclagem pode ser expandido para acomodar cenários de fotografia e fusão de gradientes contínuos:

### 1. Multi-Band Blending (Pirâmide Laplaciana de Burt & Adelson)
* **Objetivo:** Fusão contínua de fotografias do mundo real onde existem diferenças de exposição, vinheta de lente ou iluminação natural.
* **Mecanismo:**
  * Decompõe cada imagem em uma pirâmide Gaussiana/Laplaciana de frequências espaciais.
  * **Altas frequências (detalhes, bordas, texturas):** Mescladas com uma transição estreita e nítida para evitar *ghosting* ou borrões.
  * **Baixas frequências (iluminação geral, cor do céu, gradientes):** Mescladas com uma transição suave e ampla para equalizar a iluminação sem costuras visíveis.
* **Referência:** Padrão ouro em softwares como Hugin, Photoshop Photomerge e OpenCV `MultiBandBlender`.

### 2. Feather Blending (Gradiente de Distância / Distance Transform)
* **Objetivo:** Transição linear suave entre frames sobrepostos para superfícies com gradientes contínuos (ex: céu limpo, grama ou neblina).
* **Mecanismo:**
  * Aplica *Distance Transform* euclidiano a partir das bordas do frame para gerar uma máscara de rampa suave de $3$ a $8\text{ pixels}$.
  * Pondera a transição de cores suavemente na zona de sobreposição.
  * Requer alinhamento afim/subpixel de alta precisão para evitar duplicação de traços finos.

---

## ⚡ 19. Otimizações Analíticas de Álgebra Afim 2D no Pipeline de Renderização (Concluído)

O profiling linha por linha do pipeline de renderização por patch (`warp_patch`) identificou que cerca de **5% a 10% do tempo de frame** em transformações afins é consumido por overhead Python/NumPy antes do despacho para o kernel nativo C++ do OpenCV (`cv2.warpAffine`).

### Oportunidades Mapeadas e Validadas

#### 19.1. Composição Direta da Matriz Afim Final ($\mathbf{3.1\times}$ mais rápido)
* **Estado Atual:**
  ```python
  M_src_offset = mat_translation(*target_region.top_left)
  M_dst_offset_inv = mat_translation(-dst_x, -dst_y)
  M_cv2 = (M_dst_offset_inv @ matrix_global @ M_src_offset).astype(np.float64)
  ```
  Aloca duas matrizes temporárias $3 \times 3$ no NumPy e executa **duas multiplicações matriciais `@` completas**.
* **Solução Analítica:**
  Em matrizes afins 2D, translações à esquerda e à direita alteram exclusivamente a coluna de deslocamento ($X, Y$), mantendo o bloco $2 \times 2$ intacto:
  ```python
  sx, sy = target_region.top_left
  dx, dy = dest_region.top_left
  M_cv2 = np.empty((2, 3), dtype=np.float64)
  M_cv2[0, 0] = matrix_global[0, 0]
  M_cv2[0, 1] = matrix_global[0, 1]
  M_cv2[0, 2] = matrix_global[0, 0] * sx + matrix_global[0, 1] * sy + matrix_global[0, 2] - dx
  M_cv2[1, 0] = matrix_global[1, 0]
  M_cv2[1, 1] = matrix_global[1, 1]
  M_cv2[1, 2] = matrix_global[1, 0] * sx + matrix_global[1, 1] * sy + matrix_global[1, 2] - dy
  ```
* **Métrica:** Redução de **`702 ms`** para **`224 ms`** em 100.000 iterações ($\mathbf{3.1\times}$ mais rápido, zero alocações intermediárias).

#### 19.2. Projeção Escalar Pura de Vértices de Bounding Box ($\mathbf{5.8\times}$ mais rápido)
* **Estado Atual (`calculate_new_corners`):**
  Aloca um array NumPy com os 4 cantos `np.array([...]).T`, executa multiplicação matricial `@`, normalização projetiva e 4 chamadas a `np.min` / `np.max`.
* **Solução Escalar:**
  Transformar diretamente os 4 vértices do retângulo via álgebra escalar direta com os coeficientes $m_{00}, m_{01}, \dots$ e funções nativas `min()` e `max()`.
* **Métrica:** Redução de **`774 ms`** para **`134 ms`** em 50.000 iterações ($\mathbf{5.8\times}$ mais rápido).

#### 19.3. Inversão Afim 2D Analítica ($\mathbf{1.7\times}$ mais rápido)
* **Estado Atual (`mat_inverse`):**
  Chama `np.linalg.inv`, delegando para rotinas genéricas LAPACK com decomposição LU (`dgetrf`/`dgetri`).
* **Solução Analítica:**
  Inversão exata $3 \times 3$ fechada via determinante $ad - bc$:
  $$\text{inv\_det} = \frac{1}{ad - bc}$$
  $$M^{-1}_{0,2} = (b \cdot t_y - d \cdot t_x) \cdot \text{inv\_det}, \quad M^{-1}_{1,2} = (c \cdot t_x - a \cdot t_y) \cdot \text{inv\_det}$$
* **Métrica:** Redução de **`198 ms`** para **`119 ms`** em 50.000 iterações ($\mathbf{1.7\times}$ mais rápido, zero alocações LAPACK).

---

## 🛡️ 26. Eliminação de Contaminação de Cor e Franja Escura nas Bordas em `warp_affine` e `warp_patch` (Concluído)

### 1. Diagnóstico da Causa Raiz
* **O Problema:** Durante a reamostragem com filtros contínuos (Lanczos, Linear, Cubic), pixels subpixel na fronteira de retalhos RGBA amostravam o valor de borda configurado `BORDER_CONSTANT = (0, 0, 0, 0)`.
* **Consequência no Straight Alpha:** O canal Vermelho de um retalho puro `[255, 0, 0, 255]` sofria interpolação com preto, caindo de **`255`** para até **`155`** (média degradada para **`235.69`**).
* **Impacto no Stitching (`HARD_MASKING`):** Ao binarizar $\alpha \ge 128$ para $\alpha = 255$, os pixels perimetrais tornavam-se opacos contendo cores escurecidas `[155, 0, 0, 255]`, gerando uma linha escura permanente na emenda da composição.

### 2. Solução Implementada
1. **Property Semântica em `ImageFormat` (`is_straight_alpha`):**
   Identifica formatos com alfa desacoplado (`has_alpha and not is_premultiplied`: `RGBA`, `GRAY_ALPHA`, `CMYK_ALPHA`), separando-os de formatos opacos (`RGB`, `GRAY`, `RGBX`) e premultiplicados (`PRGBA`).
2. **Padding Alpha-Aware de Alta Performance (`pad_straight_alpha_into`):**
   No `warp_patch`, a margem expandida do kernel (1 a 4 pixels) é preenchida replicando os pixels de cor da borda original e fixando rigorosamente o canal alfa em `0` (transparente).
3. **Passada Única SIMD no OpenCV:**
   O `cv2.warpAffine` roda em uma única chamada de 4 canais em velocidade máxima C++, eliminando a franja sem adicionar overhead perceptível ($< 0.15\text{ ms}$).
4. **Auto-Padding em Chamadas Avulsas:**
   `warp_affine` e `warp_perspective` receberam os parâmetros `format` e `auto_pad: bool = True`, garantindo preservação de cores mesmo em chamadas diretas com matriz afim ajustada analiticamente ($\Delta T = -pad$).

### 3. Validação
* **Valor mínimo de Vermelho na borda:** Subiu de **`155`** para **`255`** exatos.
* **Valor médio de Vermelho na borda:** Subiu de **`235.69`** para **`255.00`**.
* **Suíte de Testes:** Validado com 4 novos testes unitários dedicados em `tests/test_render.py` cobrindo `uint8`, `uint16` e `float32`.

---

## 🔲 27. Resolução Dinâmica de Borda por Formato de Imagem (`border_mode` e `border_value`)

### 1. Diagnóstico e Problema Arquitetural
* **Estado Atual:** Em `warp_affine`, `warp_perspective` e `warp_patch`, o modo de borda está fixado estaticamente em `borderMode=cv2.BORDER_CONSTANT` com `border_val = (0,) * channels`.
* **Comportamento Incorreto em Formatos Opacos:**
  * **Formatos com Alfa (`RGBA`, `GRAY_ALPHA`, `CMYK_ALPHA`):** O valor `0` no canal alfa representa **transparência total**, que é o comportamento correto para o "vazio" ao redor de retalhos no Canvas.
  * **Formatos Opacos sem Alfa (`RGB`, `GRAY`, `RGBX`, `CMYK`):** O valor `0` não é transparente; representa **preto sólido** (`alpha = 255` implícito pelo motor de blend). Ao rotacionar ou deformar imagens opacas, os cantos triangulares do *bounding box* são preenchidos com preto opaco `(0, 0, 0)`, gerando uma moldura preta indesejada que sobrepõe e oculta as camadas que estão embaixo no Canvas.

### 2. Diretrizes Técnicas e Solução Mapeada
1. **Resolução de Borda por Formato de Imagem:**
   * **Formatos com Canal Alfa (`format.has_alpha == True`):**
     * `border_mode = cv2.BORDER_CONSTANT`
     * `border_value = (0,) * channels` *(garante transparência total no vazio)*
   * **Formatos Opacos sem Alfa (`format.has_alpha == False`):**
     * `border_mode = cv2.BORDER_REPLICATE` *(ou configurável)*
     * `border_value = (0,) * channels` *(evita criação de moldura e cantos pretos sólidos)*
2. **Encapsulamento e Separação de Responsabilidades:**
   * Criar a função especialista `get_opencv_border(format: ImageFormat | None, channels: int = 4) -> tuple[int, tuple[float, ...]]` em `render.py` (e/ou properties declarativas no `ImageFormat`).
   * Permitir que chamadas avulsas em `warp_affine`, `warp_perspective` e `transform_image` recebam `border_mode: int | None = None` e `border_value: tuple[float, ...] | None = None` opcionais para customização pelo usuário (ex: borda branca para exportação fotográfica).
3. **Validação:**
   * Adicionar cenários de teste em `tests/test_render.py` verificando a rotação de imagens `RGB` com `BORDER_REPLICATE` e imagens `RGBA` com `BORDER_CONSTANT`.

---

## ⚡ 28. Sistema de Cache de Camadas com Decorators e Renderização Incremental (`LayerCache`)

* **Plano Detalhado:** Consulte a especificação técnica completa, diagramas e fases de implementação em [planos/sistema_cache_decorators.md](file:///home/gui/python/anicrop/planos/sistema_cache_decorators.md).
* **Resumo:**
  1. Criação do método `Layer.background(size, format, dtype)` para municiar o buffer inicial da camada no renderizador (`layer_image`).
  2. Gerenciador `LayerCache` com context manager cirúrgico `with cache(container):` envolvendo o `traverse` em `render_scene`.
  3. Decorator `CachedLayerDecorator` que intercepta `background`, `edits` e `effects` apenas dentro do context manager, preservando o contrato original do `Layer` fora do escopo.
  4. Efeito `CacheEffect` para entrega acelerada da imagem pós-processada na fila `base.effects`.
  5. Suporte à injeção externa de imagens pré-assadas (`cache.set_baked(layer, image)`), eliminando *workarounds* de matriz inversa no `anifuse`.

---

## 🎥 29. Suporte Nativo a Formatos BGR e BGRA para Pipelines de Vídeo e Visão Computacional (Zero-Copy com OpenCV / Aniseek)

### 1. Diagnóstico e Motivação
Em pipelines de alta taxa de quadros e costura contínua de vídeo (como no fluxo `aniseek` $\rightarrow$ `anifuse` $\rightarrow$ `anicrop`):
* O leitor de vídeo (`aniseek` / OpenCV `cv2.VideoCapture`) entrega frames decodificados nativamente em arrays NumPy no formato `BGR` (3 canais) ou `BGRA` (4 canais).
* **O Gargalo Atual:** A ausência de `ImageFormat.BGR` e `ImageFormat.BGRA` no `anicrop` força uma conversão manual de canais (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` ou `COLOR_BGR2RGBA`) a cada frame lido.
* **Custo em Vídeos Longos:** Para uma sequência de 500 a 1000 frames em 1080p, são alocados e reescritos mais de **$4\text{ GB}$ a $16\text{ GB}$ de dados na memória**, gastando tempo de CPU e saturando o barramento de memória (RAM $\leftrightarrow$ Cache L3) apenas para inverter dois canais ($0 \leftrightarrow 2$).
* E se a saída final for visualizada com `cv2.imshow` ou gravada com `cv2.VideoWriter`, o resultado precisa ser convertido de volta para BGR.

### 2. Fundamentação Teórica e Simetria de Blending
A viabilidade de suportar BGR/BGRA nativamente decorre de três pilares da arquitetura do `anicrop`:
1. **Mesclagem Canal a Canal (`blend.pyx`):**
   * As fórmulas de mesclagem (`blend_normal`, `hard_masking`, `solid_fill`) são puramente lineares e operam de forma isolada em cada canal de cor ($0, 1, 2$), utilizando o índice $3$ exclusivamente como canal Alfa.
   * Como tanto em `RGBA` quanto em `BGRA` o canal Alfa reside **rigorosamente no índice 3**, a composição de uma camada `BGRA` sobre um Canvas `BGRA` é analiticamente **idêntica** à de `RGBA` sobre `RGBA`. Os mesmos kernels compilados em Cython/OpenMP funcionam sem necessidade de duplicação de código.
2. **Transformações e Warping Agnósticos (`render.py`):**
   * O OpenCV `cv2.warpAffine` e o fast-path `without_distortion` operam sobre blocos contíguos de memória independentemente de a ordem de bytes ser BGR ou RGB.
3. **Economia Exponencial de Conversões:**
   * **Fluxo sem suporte (Atual):** $N$ conversões de alta resolução na entrada ($N$ frames $\times$ BGR $\rightarrow$ RGB).
   * **Fluxo com suporte nativo (Proposto):** $0$ conversões durante todo o loop de costura. Se o usuário exportar via `PyvipsBackend` para PNG/WebP, ocorre **uma única conversão final** no panorama consolidado.

### 3. Diretrizes Técnicas de Implementação

#### 3.1. Contrato de Enum (`ImageFormat` em `anicrop.enums`)
* Adicionar novos membros:
  ```python
  class ImageFormat(StrEnum):
      ...
      BGR = "bgr"
      BGRA = "bgra"
  ```
* Atualizar properties:
  * `has_alpha`: inclui `ImageFormat.BGRA`.
  * `is_straight_alpha`: inclui `ImageFormat.BGRA`.
  * `channels`: `BGR: 3`, `BGRA: 4`.
  * `same_spaces`: define compatibilidade entre espaços de cores (ex: `bgr` e `bgra` pertencem ao espaço `"bgr"`).

#### 3.2. Mapeamento de Conversões de Cor (`anicrop.color`)
* Estender `FORMAT_CONVERTERS` para cobrir todos os pares necessários:
  * `BGR <-> RGB`: `cv2.cvtColor(data, cv2.COLOR_BGR2RGB)` / `COLOR_RGB2BGR`.
  * `BGRA <-> RGBA`: `cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)` / `COLOR_RGBA2BGRA`.
  * `BGR <-> BGRA`: `cv2.cvtColor(data, cv2.COLOR_BGR2BGRA)` / `COLOR_BGRA2BGR`.
  * `BGR <-> GRAY`: `cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)` / `COLOR_GRAY2BGR`.
  * `BGRA <-> GRAY`: `cv2.cvtColor(data, cv2.COLOR_BGRA2GRAY)` / `COLOR_GRAY2BGRA`.
  * Conversões cruzadas (ex: `BGR -> RGBA`, `BGRA -> RGB`, etc.).

#### 3.3. Injeção e Extração Zero-Copy em `Image` (`anicrop.image`)
* **Construtor `Image(data, ImageFormat.BGR)` / `Image(data, ImageFormat.BGRA)`:**
  * Aceita e encapsula a matriz NumPy do OpenCV sem nenhuma cópia intermediária.
* **Fábrica `Image.from_bgr(data, target_format=None)`:**
  * Quando `target_format is None`, auto-detecta e retorna `Image(data, ImageFormat.BGR)` ou `Image(data, ImageFormat.BGRA)` **diretamente com zero-copy** (sem chamar `cv2.cvtColor`).
  * Quando `target_format` for fornecido explicitamente, converte conforme solicitado.
* **Extração `img.bgr(region)`:**
  * Se `self.format in (ImageFormat.BGR, ImageFormat.BGRA)`, retorna o slice fatiado diretamente (`return frame`), com **custo zero de cópia**.

#### 3.4. Harmonização Automática no Renderizador (`anicrop.render`)
* Em `blend_rendered_images`:
  * Adicionar checagem de compatibilidade de espaço de cores:
    ```python
    if not image.format.same_spaces(buffer.format):
        image = image.to_format(buffer.format)
    ```
  * Se o Canvas for `BGRA` e as camadas forem `BGRA` (cenário padrão do Anifuse), a condição não dispara e o blend ocorre em velocidade máxima nativa sem conversões.
  * Se houver mistura heterogênea (ex: camada `RGB` sobre Canvas `BGRA`), a camada discrepante é harmonizada antes do blend.
* Em `SceneTraverser` e `render_scene`:
  * Garantir que `format=ImageFormat.BGRA` seja plenamente aceito e propagado pelo Canvas e buffers de grupo.

#### 3.5. Subsistema de I/O (`anicrop.io`)
* **`OpenCVBackend`:**
  * Na gravação (`write`), se a imagem for `BGR` ou `BGRA`, grava diretamente com `cv2.imwrite` sem conversão.
* **`PyvipsBackend`:**
  * Na gravação (`write`), se a imagem for `BGR` ou `BGRA`, converte internamente para `RGB` ou `RGBA` antes de entregar para o pipeline da `libvips`.

### 4. Roadmap de Execução da Tarefa

```
[FASE 1: Contrato do Enum ImageFormat]
└── Adicionar BGR e BGRA em ImageFormat com properties channels, has_alpha, is_straight_alpha e same_spaces.

[FASE 2: Tabela de Conversão em color.py]
└── Mapear todos os pares bidirecionais envolvendo BGR e BGRA com cv2.cvtColor.

[FASE 3: Otimização Zero-Copy em Image]
└── Image.from_bgr retorna BGR/BGRA nativo zero-copy quando target_format=None.
└── img.bgr() retorna slice direto sem cópia para BGR e BGRA.

[FASE 4: Harmonização no Renderizador e Blending]
└── Suporte a Canvas e camadas BGRA no render_scene e blend_rendered_images.
└── Harmonização automática no blend se format.same_spaces for diferente.

[FASE 5: Compatibilidade com Backends de I/O]
└── OpenCVBackend: gravação direta zero-copy de BGR/BGRA.
└── PyvipsBackend: conversão final segura BGR/BGRA -> RGB/RGBA na gravação.

[FASE 6: Suíte de Testes (TDD)]
└── Testes de contrato de ImageFormat (channels, has_alpha).
└── Testes de conversão de cores em test_color_formats.py.
└── Testes de zero-copy em test_image.py.
└── Testes de renderização, blending e culling com Canvas BGRA em test_render.py.
└── Testes de exportação com OpenCVBackend e PyvipsBackend.
```

---

## ⏳ 30. Consolidação e Integração Abrangente do Sistema de Histórico (Undo/Redo para Combine, Contêineres, Remoções Aninhadas e Filhos)

### 📋 Sub-Tarefas de Execução

- [x] ~~**30.1. Mutadores de Contêiner e Métodos Fluentes em `BaseContainerProxy` e `ProxyComposer`** (Concluído)~~
  - Mapeamento declarativo de mutadores no `_ACTION_ROUTER` de `BaseContainerProxy`: `move_relative`, `move_to_front`, `move_to_back`, `swap`, `reverse`, `__delitem__`.
  - Implementação nativa de `_CONTEXT_ROUTER` no `BaseHistoryProxy` com `build_context_wrapper`, permitindo que operações em lote como `clear()` rodem nativamente sob `history.atomic("clear")` sem duplicar métodos concretos no proxy.
  - Inclusão do método mutante in-place `"copy_from"` em `ProxyComposer._MUTATING_METHODS`.
  - Cobertura de 9 testes dedicados em `tests/test_reactive_container.py` validando Undo/Redo e modo direto.
- [x] ~~**30.2. Roteamento de Proxy na Remoção de Camadas Aninhadas em `Document.remove`** (Concluído)~~
  - Correção da checagem de contêiner sentinela: substituído `if not isinstance(layer.parent, NullContainer)` por `if layer.parent is not _NULL_CONTAINER`.
  - Remoção por nome (`doc.remove("nome")`) ou por proxy (`doc.remove(proxy)`) propaga naturalmente para `layer.parent` (que já é `GroupProxy`), gravando `ReparentCommand` no histórico.
  - Remoção por instância de domínio crua (`doc.remove(raw_layer)`) atua diretamente no domínio físico sem poluir o histórico com comandos inválidos.
- [ ] **30.3. Reatividade e Transações Atômicas no Serviço `Combine`**
  - Integrar `Combine` (`merge`, `flatten`, `bake`, `bake_stack`) com o histórico através de blocos `with doc.history.atomic(action_name):`.
  - Operar sobre proxies reativos de contêiner (`doc.stack` ou `GroupProxy`), registrando a remoção das camadas de origem e inserção do novo nó consolidado em **1 único MacroCommand** (1 único Undo/Redo).
  - Garantir restauração de camadas originais com propriedades, matrizes, efeitos e ordem intactas ao desfazer (`undo`).
- [ ] **30.4. Atualização de Documentação e Remoção de Ressalvas**
  - Remover do docstring de `Document.__init__` a ressalva experimental de que operações de `Combine` não gravam passos no histórico.
  - Atualizar os guias técnicos `docs/history.md`, `docs/composition.md`, `docs/anicrop_guide.md` e `GEMINI.md` documentando a integração completa de Undo/Redo em fusões e manipulações de hierarquia.

---

### 1. Diagnóstico e Lacunas Identificadas no Histórico

Embora o motor `anicrop.history` disponha de arquitetura avançada de políticas (`NormalPolicy`, `AtomicPolicy`, `MergeContinuousPolicy`), comandos com snapshots cirúrgicos e proxies para `Layer`, `GroupLayer`, `Canvas` e `Mask`, persistem lacunas estruturais que impedem a cobertura de Undo/Redo em operações de composição e contêineres:

1. **Serviço de Composição e Fusão (`doc.combine` / `Combine`):**
   * **Situação:** Métodos como `merge`, `flatten`, `bake` e `bake_stack` modificam os contêineres pai (`parent.remove()`, `parent.insert()`, `doc.stack.clear()`, `doc.stack.append()`) acessando diretamente as instâncias de domínio.
   * **Problema:** Nenhuma ação é registrada no `GlobalHistory`. Como alertado no docstring de `Document.__init__`, chamadas a `doc.combine` criam ou removem camadas fora da pilha de histórico, quebrando a linha do tempo de Undo.
2. **Remoção de Camadas Aninhadas em Grupos via `doc.remove()`:**
   * **Situação:** `doc.remove(layer_or_name)` localiza camadas em qualquer profundidade da árvore. Se a camada estiver na raiz da pilha (`layer in self.stack`), invoca `self.stack.remove(layer)` (roteado via `LayerStackProxy` para `ReparentCommand`).
   * **Problema:** Se a camada estiver dentro de um `GroupLayer`, o método invoca `layer.parent.remove(layer)`. Como `layer.parent` no objeto de domínio aponta para a instância pura de `GroupLayer` (e não para `GroupProxy`), o método de domínio é invocado diretamente sem registro de histórico.
3. **Métodos Omissos no `BaseContainerProxy._ACTION_ROUTER`:**
   * **Situação:** `BaseContainerProxy._ACTION_ROUTER` mapeia apenas `append`, `insert`, `remove`, `move` e `pop`.
   * **Problema:** Faltam no roteador mutadores essenciais de `Container`:
     * `move_relative(item, steps)`
     * `move_to_front(item)`
     * `move_to_back(item)`
     * `clear()`
     * `extend(items)`
     * `__delitem__(index)`
     * `__setitem__(index, item)`
4. **Método Fluente `copy_from` em `ProxyComposer`:**
   * `ProxyComposer._MUTATING_METHODS` contém apenas `{"rotate", "scale", "translate", "add_transform"}`.
   * `copy_from(other)` muta o `Composer` in-place, mas é ignorado pelo histórico.
5. **Mutações Diretas em Filhos sem Proxy (`Effect` e `EditLayer`):**
   * Mutações em instâncias filhas pós-adição (ex: `layer.effects[0].visible = False` ou `layer.edits[0].visible = False`) não são interceptadas porque `Effect` e `EditLayer` não possuem proxies registrados no `ProxyRegistry`.

### 2. Detalhamento Técnico das Sub-Tarefas

#### 30.1. Mutadores Avançados de Contêiner e ProxyComposer
1. **Completude do `_ACTION_ROUTER` em `BaseContainerProxy`:**
   - **`move_relative(item, steps)`:** Converte o deslocamento relativo de índice em transição de reordenação via `ReparentCommand`.
   - **`move_to_front(item)` / `move_to_back(item)`:** Mapeia a movimentação para o topo/base do contêiner registrando `ReparentCommand`.
   - **`swap(a, b)`:** Registra a troca de posições entre dois nós dentro de um bloco atômico ou snapshot de reordenação.
   - **`clear()`:** Quando executado em proxy reativo sob histórico, opera iterativamente sob `with history.atomic("clear"):` removendo filho a filho (ou via snapshot de contêiner), permitindo restaurar a árvore de filhos integralmente em 1 único Undo.
   - **`extend(items)`:** Adiciona múltiplos filhos iterativamente sob `with history.atomic("extend"):`, registrando cada inserção em 1 único MacroCommand.
   - **`__delitem__(index)`:** Suporte à deleção por índice mapeando o filho correspondente para `ReparentCommand`.
2. **Atualização de `ProxyComposer`:**
   - Adicionar `"copy_from"` ao conjunto `_MUTATING_METHODS` em `src/anicrop/reactive/fluent.py`, garantindo que cópias diretas de transformações emitam o `ComposerCommand` correspondente.

#### 30.2. Roteamento de Proxy na Remoção de Camadas Aninhadas (`Document.remove`) (Concluído)
1. **Correção de Checagem Sentinela de Contêiner:**
   - Em `Document.remove(layer_or_name)`, a checagem anterior `if not isinstance(layer.parent, NullContainer)` falhava para qualquer nó real de domínio porque `Container` herda de `NullContainer`.
   - Substituído pelo padrão idiomático do motor: `if layer.parent is not _NULL_CONTAINER:`.
2. **Separação Limpa de Domínio vs. Reatividade:**
   - Remoção por nome (`doc.remove("nome")`): `doc.find()` obtém o `ProxyLayer`, cujo `layer.parent` já devolve o `GroupProxy`, disparando o `ReparentCommand` e permitindo Undo/Redo cirúrgico.
   - Remoção por proxy (`doc.remove(proxy)`): `proxy.parent` já devolve o `GroupProxy`, registrando histórico legitimamente.
   - Remoção por instância crua (`doc.remove(raw_layer)`): atua estritamente no contêiner físico sem criar comandos fantasmas nem forçar proxies retroativos no histórico.
3. **Conclusão e Testes:**
   - Implementado com sucesso em `src/anicrop/document.py` e validado por 6 novos testes em `tests/test_document.py` no commit `ddc80c1`.

#### 30.3. Reatividade e Transações Atômicas no Serviço `Combine`
1. **Orquestração Atômica de `Combine`:**
   - Em operações que alteram a árvore de camadas (`merge`, `flatten`, `bake`, `bake_stack`):
     - Quando `doc.history_enabled=True`, executar a etapa de manipulação do contêiner sob `with self._doc.history.atomic(f"Combine: {op_name}"):`.
     - Obter o contêiner alvo (`parent`) encapsulado em proxy (`self._doc.stack` se for raiz, ou `self._doc._policy.process_layer(parent, self._doc.history)` se for grupo).
     - Invocar as remoções e inserções através do contêiner reativo.
2. **Garantia de 1 Único Passo de Undo/Redo:**
   - O `MacroCommand` agrupa atomicamente:
     1. A remoção das camadas de origem `sequence` (se `remove_source=True`);
     2. A inserção do novo nó consolidado (`GroupLayer` ou `Layer`) no índice correto (`lowest_index`).
   - Ao executar `doc.history.undo()`, a camada consolidada é removida do contêiner e todas as camadas originais são reinseridas em suas posições e estados exatos.
   - Ao executar `doc.history.redo()`, a fusão é reaplicada.
3. **Casos Especialistas (`bake` e `bake_stack`):**
   - Em `bake(group)`: substitui atomicamente o grupo por uma camada rasterizada. O Undo recria o grupo intacto com seus filhos originais.
   - Em `bake_stack()`: limpa a pilha via `self._doc.stack.clear()` e adiciona a camada achatada via `self._doc.stack.append(flat_layer)`. O Undo restaura todas as camadas anteriores da pilha na ordem original.

#### 30.4. Atualização de Documentação e Remoção de Ressalvas
1. **Limpeza da API Pública do Documento:**
   - Remover o aviso experimental sobre `Combine` no docstring de `Document.__init__`.
2. **Atualização da Documentação Técnica:**
   - Atualizar `docs/history.md` com as garantias de atomicidade para contêineres e `Combine`.
   - Atualizar `docs/composition.md` com exemplos práticos demonstrando `doc.combine.merge(...)` e `doc.combine.bake(...)` seguidos de `doc.history.undo()`.
   - Atualizar `GEMINI.md` no roadmap refletindo a conclusão da consolidação do histórico.

---

### 3. Matriz de Testes Planejados (TDD)

1. **Testes de Contêiner Reativo (`tests/test_reactive_container.py`):**
   - Validação de Undo/Redo para `move_relative`, `move_to_front`, `move_to_back`, `swap`.
   - Validação de Undo/Redo para `clear()` com múltiplos filhos e restauração integral da lista.
   - Validação de Undo/Redo para `extend([a, b, c])` em 1 único passo atômico.
   - Validação de Undo/Redo para `del container[idx]`.
2. **Testes de ProxyComposer (`tests/test_reactive_fluent.py`):**
   - Validação de Undo/Redo para `layer.transform.copy_from(other_composer)`.
3. **Testes de Remoção Aninhada (`tests/test_document_history.py`):**
   - `doc.remove(nested_layer)` dentro de `GroupLayer`: Undo deve recolocar o nó exatamente dentro do grupo na posição correta.
4. **Testes de Composição Reativa (`tests/test_combine_history.py`):**
   - Undo/Redo de `doc.combine.merge` (1 passo restaura camadas originais e descarta grupo).
   - Undo/Redo de `doc.combine.flatten` (1 passo restaura camadas originais e descarta camada achatada).
   - Undo/Redo de `doc.combine.bake` (1 passo restaura `GroupLayer` e seus filhos).
   - Undo/Redo de `doc.combine.bake_stack` (1 passo restaura toda a pilha do documento).

---

## ✅ 31. Modificar a Referência das Camadas no Cache para Referência Fraca (`weakref` em `LayerCache._states`) (Concluído)

### 1. Diagnóstico e Motivação
Atualmente, a classe `LayerCache` mantinha o dicionário `self._states: dict[Layer, LayerFrameState] = {}`. Como as chaves eram referências fortes (*strong references*) para as instâncias de `Layer`:
* Se uma camada fosse removida da cena (`doc.remove`, `container.remove`, `Combine.flatten`, `Combine.bake`), ela continuaria retida em memória enquanto o `LayerCache` existisse.
* Pior ainda: o `LayerFrameState` retido segurava instâncias pesadas de imagem em `status.baked_warp` e `status.baked_effects`, gerando vazamento crônico de memória RAM e de buffers mapeados em disco (`MMapBuffer`).

### 2. Diretrizes Técnicas e Solução Arquitetural
1. **Adoção de `weakref.WeakKeyDictionary`:**
   * Substituição do dicionário padrão `dict[Layer, LayerFrameState]` por `weakref.WeakKeyDictionary[Layer, LayerFrameState]`.
   * Quando uma camada é descartada pelo Garbage Collector (ou removida de todos os contêineres e variáveis do usuário), sua entrada no `_states` e seus buffers pré-assados (`baked_warp`, `baked_effects`) são expurgados automaticamente sem necessidade de `cache.unregister` manual.
2. **Eliminação de Ciclos de Referência Forte em Métodos Monkey-Patched:**
   * Removidos todos os atributos `orig_*` (`orig_add_edit`, `orig_add_effect`, `orig_bind_effect`, `orig_background`) de `LayerFrameState` que criavam referências cíclicas fortes para `Layer` através de *bound methods*.
   * Restauração limpa de métodos na camada delegada diretamente ao dicionário da instância (`layer.__dict__.pop(...)`), permitindo que a hierarquia de classes retome o despacho original sem retenção de memória.

### 3. Conclusão e Resolução
* Implementado com sucesso em `src/anicrop/cache.py` no commit `1ab3ab5`.
* Suíte completa com 1.303 testes aprovada sem regressões, incluindo teste dedicado de coleta automática (`test_layer_cache_clears_discarded_layers_via_weakref`).

---

## ✅ 32. Sistema de Invalidação mais Robusto para Efeitos via Inspeção de Bytecode de `apply` (Concluído)

### 1. Diagnóstico e Motivação
O sistema atual de invalidação de efeitos em `LayerCache` apresentava duas limitações graves:
1. **Invalidação Ingênua por Contagem e Booleano:** Apenas checava se `len(layer.effects) < baked_effects_count` ou se a tupla de visibilidade booleana mudou. Se um efeito fosse substituído por outro diferente, ou se o usuário alterasse os parâmetros de um filtro (ex: `blur.radius_x = 10.0`), o cache não detectava e continuava servindo o `baked_effects` desatualizado.
2. **Mutações Diretas na Referência Original:** Mesmo com proxies, se o usuário mantivesse uma referência da variável original (`blur = BlurFilter(5.0); layer.add_effect(blur); blur.radius_x = 10.0`), a mutação ocorria diretamente no objeto sem passar por nenhum proxy.

### 2. Diretrizes Técnicas e Solução Arquitetural
1. **Inspeção de Bytecode Focada Estritamente no Método `cls.apply`:**
   * Utilizar `dis.get_instructions(cls.apply)` para inspecionar os acessos a atributos da instância (`LOAD_FAST 'self'` seguido de `LOAD_ATTR <nome>`).
   * **Escopo estrito:** Inspeciona **exclusivamente o método `apply`** (sem recursão em métodos auxiliares).
   * **Filtros e Preservação:**
     - Ignora dunders (`__...__`).
     - Ignora métodos/callables definidos na classe (`callable(getattr(cls, name, None))`).
     - **Preserva atributos privados com `_`** (ex: `_radius`), assegurando que se o usuário alterar um atributo através de um método/setter e `apply` consumir `self._radius`, a mutação seja capturada.
     - Inclui `"visible"` como atributo monitorado.
2. **Compilação e Cache por Tipo (`WeakKeyDictionary`):**
   * Compila a lista de nomes de atributos monitorados uma única vez por tipo de efeito em `WeakKeyDictionary[type, tuple[str, ...]]`.
3. **Snapshot de Estado e Validação a Cada Frame:**
   * No cache, extrai os valores dos atributos monitorados de cada efeito ativo (`snapshot_effect`). Converte `ndarray` para `.tobytes()` e executa recursão para `BoundEffect`.
   * Compara o snapshot atual com o snapshot salvo. Qualquer divergência invalida `status.baked_effects`.

### 3. Conclusão e Resolução
* Implementado com sucesso em `src/anicrop/cache.py` no commit `ac35cb9`.
* Suíte completa com 1.302 testes aprovada sem regressões.

---

## ✅ 33. Sistema de Invalidação mais Robusto para Edits Usando Somente `visible` e `blend_mode` (Concluído)

### 1. Diagnóstico e Motivação
Na arquitetura do `anicrop`, um `EditLayer` é um registro imutável do corte/patch: sua imagem (`image`), região (`region`) e matriz espacial (`matrix`) são fixados na criação. Os **únicos** atributos mutáveis de um `EditLayer` são:
1. `visible: bool`
2. `blend_mode: BlendMode`

Anteriormente, o `LayerCache` tentava monitorar edits acumulando instâncias em uma lista paralela `status.edits` via monkey-patching em `layer.add_edit`. Isso falhava gravemente quando:
* O histórico executava `undo()` ou `redo()`: a lista `layer._edits` era restaurada via snapshot, sem chamar `add_edit`.
* Edits permaneciam vivos em múltiplos frames ou eram desativados via `edit.visible = False` ou tinham seu modo de mesclagem alterado.

### 2. Diretrizes Técnicas e Solução Arquitetural
1. **Snapshot Leve por Tupla de Primitivos (`snapshot_edit`):**
   * Em vez de instanciar classes que retenham referências fortes a instâncias de `EditLayer` (e suas respectivas `Image`s pesadas), a função `snapshot_edit(edit)` extrai a tupla de primitivos `(id(edit), edit.visible, edit.blend_mode)`.
   * **Zero retenção de memória:** O cache armazena apenas inteiros e booleanos em `status.baked_edits_snapshot`. Se um edit for descartado, a memória é liberada imediatamente pelo GC.
2. **Eliminação do Acumulador Paralelo de Deltas:**
   * Removidos `wrap_add_edit` e a lista `status.edits`.
   * Quando `baked_warp` é válido, os edits pendentes a renderizar sobre ele são os excedentes da coleção atual (`layer._edits[len(status.baked_edits_snapshot):]`). O cache expõe temporariamente apenas esse trecho dentro do escopo (`with cache(...)`) e restaura a coleção original no `__exit__`.
   * Validação do prefixo por iteração (`zip(status.baked_edits_snapshot, layer._edits)`), sem indexar em loop.
   * Se `len(layer._edits) < len(status.baked_edits_snapshot)` (houve `undo`) ou se qualquer edit assado divergir em identidade, visibilidade ou blend mode, invalida o `baked_warp` e re-assa do zero.

### 3. Conclusão e Resolução
* Implementado com sucesso em `src/anicrop/cache.py` no commit `67ee62b`.
* Suíte completa com 1.307 testes aprovada sem regressões, com validações cobrindo múltiplos frames, alternância de `blend_mode`, redução de contagem de edits e substituição de instâncias.

---

## ✅ 34. Remoção do Método Obsoleto `offset` do `EditLayer` (Concluído)

### 1. Diagnóstico e Motivação
A classe `EditLayer` possuía o método `offset(offset_x: int, offset_y: int) -> None` que mutava `self._region += (offset_x, offset_y)`.
* **Código Morto:** O método não era invocado em nenhum lugar do repositório (`anicrop`, testes ou benchmarks).
* **Violação de Imutabilidade Espacial:** A região de um `EditLayer` representa a moldura geométrica fixa onde o patch foi aplicado em coordenadas locais da camada. Mutações arbitrárias de offset violavam a imutabilidade do patch e criavam brechas para dessincronismo no cache de LOD (`_lod_cache`).

### 2. Conclusão e Resolução
* Método removido com sucesso de `src/anicrop/edit_layer.py` no commit `91ba2b2`.
* Suíte completa com 1.294 testes aprovada sem regressões.

---

## ✅ 35. Otimizações de Baixa Latência e Zero-Alloc na Invalidação do `LayerCache` (Concluído)

### 1. Diagnóstico e Motivação
A validação de integridade do cache a cada frame é executada em loops interativos e renderização contínua (ex: visualizador `Viewer`, pipelines de vídeo no `Anifuse` e animações com 60 a 120 FPS). Embora o custo inicial de checagem fosse baixo (~5 µs), existiam gargalos desnecessários de alocação de memória no heap e sobrecarga em funções dinâmicas do Python e NumPy:
1. **Sobrecarga de `np.allclose` / `np.array_equal`:** Comparar matrizes 2x2/3x3 custava ~**6.500 ns**, pois o NumPy realiza verificações de broadcasting, aloca arrays booleanos temporários e reduz com `.all()`.
2. **Alocação Contínua de Tuplas em Properties:** Invocar `layer.edits` a cada frame disparava `return tuple(self._edits)`, alocando tuplas efêmeras a cada frame e pressionando o Garbage Collector.

### 2. Diretrizes Técnicas e Solução Arquitetural
1. **Comparação de Matrizes Afins por Bytes (`matrix[:2, :2].tobytes()` / `memcmp`):**
   * Armazenamento de `status.matrix_2x2_bytes: bytes | None = None` em `LayerFrameState`.
   * A comparação `curr_2x2_bytes != status.matrix_2x2_bytes` em `_activate_layer` e `is_dirty` é traduzida diretamente para a função C `memcmp()`.
   * **Speedup comprovado:** Redução de **6.500 ns para ~15 a 105 ns** ($60\times$ a $400\times$ mais rápido), com zero alocações temporárias.
2. **`ListView[T]` Genérica Somente Leitura e Migração de `Layer._edits` para `list`:**
   * Criada `ListView[T](Sequence[T])` genérica em `src/anicrop/type.py`, com `__slots__ = ("_data",)`, provendo `__len__`, `__iter__`, `__getitem__` (`int` e `slice`), `__contains__`, `__repr__` e `__eq__`. Sem métodos mutantes.
   * `Layer._edits` migrado de `deque` para `list` pura em `src/anicrop/layer.py`, `src/anicrop/command.py` e `src/anicrop/cache.py`.
   * `Layer.edits` devolve `ListView(self._edits)` (~30 ns, sem cópia de dados).
   * O cache lê `layer._edits` diretamente em `_activate_layer` e `set_baked`.
3. **Snapshot de Primitivos de Efeitos e Edits:**
   * Assinatura imutável nativa em `snapshot_effect` e `snapshot_edit` eliminando alocações dinâmicas.

### 3. Conclusão e Resolução
* Implementado com sucesso nos commits `1fe4a9e` e anteriores.
* Suíte completa com 1.311 testes aprovada sem regressões, cobrindo operações e invariâncias de `ListView`, migração de snapshots de histórico e invalidação instantânea por bytes.

---












