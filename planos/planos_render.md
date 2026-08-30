# Plano de Renderização: Casos de Borda e Otimizações Sem Cache (`planos_render.md`)

Este documento consolida o planejamento de testes para casos de borda e o roadmap de otimizações de performance estruturais (sem uso de cache) para o motor de renderização do `anicrop` (`render.py`, `blend.py`, `frame.py`, `transform.py`).

---

## 📋 Lista Enumerada de Tarefas

- [x] 1. **Testes de Culling Total e Recorte de Coordenadas Negativas nas Bordas da Superfície**

- [x] 2. **Testes de Comportamento para `GroupLayer` Vazio, Monofilho e Filhos Fora da Tela**

- [x] 3. **Testes de Early-Exit por Oclusão Total (`miniview`) e Camadas Invisíveis (`visible=False`)**

- [x] 4. **Testes de Transformações Geométricas Extremas (Rotações Não-Ortogonais e Escalas)**

- [x] 5. **Testes de `render_patch` com `view_region` Disjunto e Fatiamento de Grupos**

- [x] 6. **Otimização: Fast-Path para Translação Pura (Bypass de `cv2.warpAffine` via Slicing NumPy)**
- [x] 7. **Otimização: Fast-Path de Blending (Bypass de Porter-Duff para Imagens Opacas via `np.copyto`)**
- [x] 8. **Otimização: Pre-Culling de AABB no `SceneTraverser` com `freeze_geometry`**
- [x] 9. **Otimização: Alocação Direta de Buffer de Destino via `_scratch_image` com `Image` e `Region`**
- [x] 10. **[VETADO / CANCELADO] Otimização: Bypass de Composição Intermediária para `GroupLayer` com Filho Único**
- [x] 11. **[TRANSFERIDO] Otimização: Multiplicação Especializada de Matrizes Afins 2D ($2 \times 3$) -> `planos/plano.md`**


---

## 🛠️ Detalhamento das Sessões

---

### Tarefa 1: Testes de Culling Total e Recorte de Coordenadas Negativas nas Bordas da Superfície

- **Objetivo:** Garantir que elementos fora ou cortados nas bordas da superfície não gerem erros de indexação no NumPy nem alocações desnecessárias.
- **Arquivos Impactados:** `tests/test_render.py`, `src/anicrop/render.py`, `src/anicrop/frame.py`.
- **Cenários a Cobrir:**
  1. **Culling Total (100% fora do Canvas):** Camada em `Region(1000, 1000, 100, 100)` em `Canvas(500, 500)`. Validar que `frame.dst_region` é `None`, `render_area` retorna `None` e nenhum item é adicionado ao `rendered_items`.
  2. **Recorte Superior/Esquerdo (Coordenadas Negativas):** Camada em `Region(-40, -30, 100, 100)` em `Canvas(500, 500)`. Validar que `dst_region` resultante é `Region(0, 0, 60, 70)` e que o slicing de arrays é livre de exceções.
  3. **Recorte Inferior/Direito:** Camada em `Region(450, 470, 100, 100)` em `Canvas(500, 500)`. Validar `dst_region == Region(450, 470, 50, 30)` e mesclagem limpa no Canvas.
- **Critério de Aceite:** Todos os casos passam com asserções geométricas estritas e sem vazamento de limites de array.

---

### Tarefa 2: Testes de Comportamento para `GroupLayer` Vazio, Monofilho e Filhos Fora da Tela

- **Objetivo:** Blindar o pipeline de travessia e blend contra estruturas de árvore vazias, degeneradas ou sem itens renderizáveis.
- **Arquivos Impactados:** `tests/test_render.py`, `src/anicrop/render.py`.
- **Cenários a Cobrir:**
  1. **Grupo Vazio (`len(group) == 0`):** Validar que `SceneTraverser.traverse(group)` retorna lista vazia sem tentar alocar imagem de tamanho zero.
  2. **Grupo com Filhos Fora da Tela:** Todas as camadas filhas sofrem culling. Validar que o grupo é ignorado na lista final de renderização.
  3. **Grupos Aninhados Profundos ($3+$ níveis de profundidade):** Composição de grupo dentro de grupo com matrizes encadeadas. Validar que as posições locais relativas aos buffers de grupo intermediários são compensadas corretamente.
- **Critério de Aceite:** O renderizador executa sem exceções em qualquer topologia de grupos vazios ou aninhados.

---

### Tarefa 3: Testes de Early-Exit por Oclusão Total (`miniview`) e Camadas Invisíveis (`visible=False`)

- **Objetivo:** Validar o mecanismo de early-exit por oclusão conservadora (*Front-to-Back*) e o descarte imediato de camadas com visibilidade desativada.
- **Arquivos Impactados:** `tests/test_render.py`, `src/anicrop/render.py`.
- **Cenários a Cobrir:**
  1. **Oclusão Total (Early-Exit):** Camada superior sólida 100% opaca cobrindo todo o Canvas com `BlendMode.NORMAL`. Validar que as camadas inferiores não têm `render_area` invocado (`break` ativado).
  2. **Camadas com `visible = False`:** Camadas isoladas ou grupos inteiros com `visible = False` são sumariamente ignorados na travessia.
- **Critério de Aceite:** Spy em `render_area` confirma que zero chamadas são feitas para camadas ocluídas ou invisíveis.

---

### Tarefa 4: Testes de Transformações Geométricas Extremas (Rotações Não-Ortogonais e Escalas)

- **Objetivo:** Assegurar a precisão matemática do cálculo de AABB projetado e do warp com transformações não-triviais.
- **Arquivos Impactados:** `tests/test_render.py`, `tests/test_render_transform.py`.
- **Cenários a Cobrir:**
  1. **Rotações em Ângulos Arbitrários (45°, 135°, 210°):** Validar cálculo do AABB circunscrito no frame e interpolação sem artefatos de borda.
  2. **Downscaling e Upscaling Extremos:** Camada gigante $4000 \times 4000$ reduzida para $50 \times 50$ (ativação de LOD) e camada $10 \times 10$ ampliada para $500 \times 500$.
  3. **Camada de Dimensão Mínima ($1 \times 1\text{px}$):** Validar que matrizes, interpolações e operações de `Span` não sofrem divisão por zero.
- **Critério de Aceite:** Composição final idêntica aos padrões de renderização matemática esperados.

---

### Tarefa 5: Testes de `render_patch` com `view_region` Disjunto e Fatiamento de Grupos

- **Objetivo:** Validar a renderização por patch com regiões de visão arbitrárias e restritivas.
- **Arquivos Impactados:** `tests/test_render.py`, `src/anicrop/render.py`.
- **Cenários a Cobrir:**
  1. **`view_region` Disjunto:** Renderizar patch em região sem camadas. Deve retornar imagem preenchida com `surface.bg_color`.
  2. **`view_region` Fatiando um Grupo:** O `view_region` cobre apenas parte da área de um `GroupLayer`. Validar que apenas os pixels interceptados são gerados.
- **Critério de Aceite:** Retalhos gerados batem exatamente com as dimensões de `surface.size` e o conteúdo delimitado por `view_region`.

---

### Tarefa 6: [x] Otimização: Fast-Path para Translação Pura (Bypass de `cv2.warpAffine` via Slicing NumPy)

- **Objetivo:** Acelerar em até $200\times$ a renderização de camadas que sofrem apenas translação inteira sem rotação, escala ou deformação.
- **Arquivos Impactados:** `src/anicrop/transform.py`, `src/anicrop/render.py`.
- **Implementação Técnica:**
  - Implementada função geométrica `has_distortion(matrix)` em `transform.py`.
  - Em `render_image`, verificar `not has_distortion(m_render)`.
  - Se verdadeiro, calcular `src_region = dest_region - (tx, ty)` e extrair diretamente `image[src_region]` sem invocar `render_patch` nem `cv2.warpAffine`.
- **Ganho Esperado:** Eliminação completa do overhead de interpolação e reamostragem em camadas puramente transladadas.


---

### Tarefa 7: [x] Otimização: Fast-Path de Blending (Bypass de Porter-Duff para Imagens Opacas via `np.copyto`)

- **Objetivo:** Otimizar o algoritmo de mesclagem `blend_normal` para evitar conversões para ponto flutuante quando o canal alfa for desnecessário.
- **Arquivos Impactados:** `src/anicrop/blend.py`.
- **Implementação Técnica:**
  - Quando `opacity >= 1.0` e a imagem for sólida (`RGB` ou `alpha == 255` em todos os pixels do patch), executar `np.copyto(b_view, e_view)` diretamente em nível de C.
  - Se a base for RGBA e o edit for RGB sólido, copiar RGB e definir `b_view[..., 3] = 255`.
- **Ganho Esperado:** Redução significativa de alocações temporárias no Garbage Collector e ganho expressivo de FPS no blending.


---

### Tarefa 8: [x] Otimização: Pre-Culling de AABB no `SceneTraverser` (Zero Alocação de Frame)

- **Objetivo:** Descartar camadas fora da superfície antes de instanciar classes de frame ou multiplicar matrizes inversas.
- **Arquivos Impactados:** `src/anicrop/render.py`, `src/anicrop/geometry.py`, `src/anicrop/container.py`.
- **Implementação Técnica:**
  - No loop de `SceneTraverser.traverse`, verificar antecipadamente se `item.global_region.overlaps(effective_region)`.
  - Se não houver colisão, descartar o item imediatamente com zero alocações de objetos `BaseFrame`.
  - Integrar `with freeze_geometry(container):` em `render_scene` e `render_patch` para avaliação preguiçosa com cache de matrizes na travessia.
- **Resultados de Benchmark:** Consulte a documentação oficial consolidada em [`docs/benchmark.md`](file:///home/gui/python/anicrop/docs/benchmark.md).

---

### Tarefa 9: [x] Otimização: Alocação Direta de Buffer de Destino no `cv2.warpAffine` (`dst=...`)

- **Objetivo:** Evitar a criação e cópia intermediária de arrays pelo OpenCV durante o warping.
- **Arquivos Impactados:** `src/anicrop/render.py`.
- **Implementação Técnica:**
  - `BaseRenderer` gerencia um `_scratch_image` com expansão geométrica ($1.5\times$) e fatiamento via `Image.view(Region)`.
  - `_flatten_edits` obtém o scratch buffer uma única vez por camada `self._get_scratch_buffer(*layer_image.size, layer_image.format)`.
  - `render_image` fatia o target local `target_dst = dst[dst_patch_local]` e repassa para `cv2.warpAffine(..., dst=target_dst)` ou `np.copyto(target_dst, image[src_view])`.
  - Zero alocações intermediárias de arrays NumPy por edição durante a renderização.
- **Baseline de Estresse:** Consulte os resultados de estresse Full HD vs. Patch em [`docs/benchmark.md`](file:///home/gui/python/anicrop/docs/benchmark.md).
- **Ganho Obtido:** Eliminação de alocações transitórias de memória RAM e total reaproveitamento de buffer via API `Image`/`Region`.



---

### Tarefa 10: [VETADO / CANCELADO] Otimização: Bypass de Composição Intermediária para `GroupLayer` com Filho Único

- **Motivo do Cancelamento:** Violação do contrato arquitetural de composição isolada de `GroupLayer` (onde o grupo gerencia isoladamente sua própria opacidade, blend mode, bounding box e mapeamento de coordenadas).

---

### Tarefa 11: [TRANSFERIDO] Otimização: Multiplicação Especializada de Matrizes Afins 2D ($2 \times 3$)

- **Status:** Transferido para [`planos/plano.md`](file:///home/gui/python/anicrop/planos/plano.md) como micro-otimização opcional (Item 19).

