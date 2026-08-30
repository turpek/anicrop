# Receita de Implementação: Evolução do EditRenderer

Este documento descreve as etapas estruturadas em forma de lista para implementar o suporte a ROI (Region of Interest) local e LOD (Level of Detail) na renderização de edições.

---

## 1. Novo Parâmetro `view_region` (ROI Local)
* **Objetivo:** Evitar o warp completo de edições gigantes, processando apenas a sub-região visível na Viewport.
* **Passos na Classe Chamadora (`ViewportRender` / `LayerRender`):**
  1. Calcular a matriz combinada que mapeia do espaço local do Layer para a tela da Viewport ($M_{view\_layer} = M_{view} \cdot M_{layer\_global}$).
  2. Calcular a inversa dessa matriz combinada ($M_{view\_layer\_inv} = \text{inv}(M_{view\_layer})$).
  3. Projetar o retângulo visível da tela (`render_region`) de volta para o espaço local do Layer para gerar a `view_region_local` (`calculate_new_bbox`).
  4. Enviar a `view_region_local` como argumento para o método `EditRenderer.render`.
* **Passos no `EditRenderer.render`:**
  1. Receber o parâmetro `view_region: Optional[Region] = None` (representando a região no espaço local do Layer).
  2. Passar `view_region` para a classe `CanvasPlan` no momento do planejamento geométrico.
  3. Recuperar a `clipped_region` (o retângulo final exato no espaço do Layer onde o warp deve desenhar).
  4. Recuperar a `local_region` (a sub-região exata da imagem do edit original que deve ser lida/cortada).
  5. Fatiar a imagem de origem usando a `local_region` antes de enviar para o warp.

---

## 2. Novo Parâmetro `scale_factor` (Seleção de LOD)
* **Objetivo:** Ler a imagem de origem da edição em baixa resolução para otimizar processamento e memória RAM/I/O.
* **Passos no `EditRenderer.render`:**
  1. Receber o parâmetro `scale_factor: float = 1.0` (indicando o fator de zoom atual da tela).
  2. Chamar o método de LOD interno da classe `EditLayer` passando o `scale_factor`.
  3. Obter os pixels do nível de LOD correspondente e a matriz de ajuste `m_adjust`.
  4. Multiplicar a matriz de renderização local por `m_adjust` para compensar a redução de escala das coordenadas locais da imagem.
  5. Executar o warp usando os pixels do LOD e a matriz ajustada.

---

## 3. Comportamento de `Layout.fit_content` com Crops, Máscaras e Edições
* **Objetivo:** Definir e padronizar o cálculo de enquadramento de conteúdo (`Layout.fit_content`) quando o alvo possuir recortes (`Content.crop` via `BlendMode.CLIP`), patches de `EditLayer` ou máscara de camada ativa (`Mask`).
* **Passos e Aspectos Técnicos:**
  1. **Bounding Box Efetiva:** A ROI calculada por `calculate_content_rect` deve considerar os limites efetivos de transparência/visibilidade resultantes da composição dos patches e máscaras.
  2. **Interseção com Máscaras e Clip:** Pixels tornados transparentes por `BlendMode.CLIP` ou por `Mask` não devem expandir a Bounding Box calculada.
  3. **Preservação de Geometria:** O enquadramento deve ser aplicado preservando a integridade da geometria estrutural (`base.region`) e do pivô de transformação.
