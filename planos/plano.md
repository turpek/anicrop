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
- [ ] 14. Implementar Pipeline de Processamento de Pixels e Efeitos (Filtros, Ajustes de Cor e Tom).
- [x] ~~15. Decisão Arquitetural: Análise da aplicação de translação em `calculate_new_rect` e `calculate_region_rect` ao consumir `mat_global`.~~
- [x] ~~16. Consolidação Unificada de Frames (`BaseFrame`, `CanvasFrame`, `ViewportFrame`) e Separação entre `surface` e `view_region`.~~
- [x] ~~17. Validação e Correção da Máscara de Oclusão (`_opacity_mask` / Early-Exit) em Relação ao `surface_size`.~~
- [x] ~~18. Decisão Arquitetural: Natureza e Gerenciamento da Transformação em `BaseLayer` (Sincronização de Região no `Composer` via `sync_region`).~~
- [ ] 19. (Micro-otimização) Multiplicação Especializada de Matrizes Afins 2D ($2 \times 3$).
- [ ] 20. Padronizar o comportamento de `Layout.fit_content` quando a camada possui crop (`BlendMode.CLIP`), máscara ativa (`Mask`) ou patches de `EditLayer`.
- [x] ~~21. Implementar subclasse de `EditLayer` (ou atributo `visible`) para controle de visibilidade de edições/crop e integração no `_flatten_edits` (Modelo GIMP).~~
- [ ] 22. Implementar `ViewportLayoutStrategy` para gerenciar enquadramento, navegação e foco de câmera (`fit`, `align`, `fit_content`, `resize_bounds`).






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





