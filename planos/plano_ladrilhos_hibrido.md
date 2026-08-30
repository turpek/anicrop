# Arquitetura Híbrida de Renderização (Tiling Adaptativo por LOD)

Este documento especifica a estratégia de renderização híbrida do **anicrop**, desenvolvida para otimizar o desempenho de exibição em dois cenários distintos: **Zoom Out** (visão geral) e **Zoom In** (detalhamento de alta resolução).

---

## 1. Problema de Desempenho do Tiling Rígido

Se aplicarmos a divisão por ladrilhos (*tiles*) de $512 \times 512$ de forma indistinta em todos os níveis de visualização:
* Durante o **Zoom Out** em uma imagem de $50.000 \times 50.000$, a Viewport precisa exibir a camada por inteiro.
* O motor teria que iterar, compor e fazer o *warp* de milhares de tiles pequenos para compor uma imagem que caberia em uma tela Full HD ($1920 \times 1080$), gerando um enorme *overhead* de CPU e queda drástica de FPS.

---

## 2. A Solução: Renderização Híbrida em Dois Regimes

O motor altera dinamicamente seu pipeline com base na **Resolução Efetiva do LOD Ativo** necessária para atender o zoom atual da Viewport:

```text
               ┌─────────────────────────────────────────┐
               │    Qual a resolução do Layer no LOD    │
               │        necessário para o Zoom?          │
               └────────────────────┬────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [ Resolução Pequena ]               [ Resolução Gigante ]
      (LOD reduzido ou Layer normal)         (Zoom In ou LOD 0)
                  │                                   │
                  ▼                                   ▼
       ┌─────────────────────┐             ┌─────────────────────┐
       │     Direct Pass     │             │     Tiled Pass      │
       │    (Passo Único)    │             │   (Grade de Tiles)  │
       └─────────────────────┘             └─────────────────────┘
```

### 2.1. Direct Pass (Passo Único - Zoom Out / Layers Normais)
* **Critério de Ativação:** A resolução necessária do LOD atual cabe dentro do limiar de alta performance (ex: menor ou igual à resolução da Viewport ou $\le 2048 \times 2048$).
* **Funcionamento Geral:** O motor ignora a divisão em tiles, obtém o buffer reduzido do layer, realiza a composição dos edits em passo único e aplica um único *warp* para o espaço de exibição.

Para a execução do **Passo Único**, duas estratégias de gerenciamento de LOD e cache foram mapeadas:

#### Abordagem A: Com LOD Adaptativo (Por Degraus Discretos)
* **Como funciona:** O nível do LOD é selecionado dinamicamente com base no zoom atual da Viewport ($N = \lfloor -\log_2(f) \rfloor$).
* **Comportamento do Cache:** Os níveis de LOD são faixas discretas ($1.0, 0.5, 0.25, 0.125$). Pequenas variações de zoom dentro da mesma faixa não invalidam o cache. Apenas ao cruzar um degrau de escala é que um novo nível de LOD é solicitado.
* **Vantagens:** Mantém a nitidez excelente e ajustada precisamente à escala de exibição da tela.
* **Gerenciamento:** Armazena os buffers planos compostos por nível de LOD em dicionário (`cache[layer._id][lod_level]`), tornando a alternância de zoom instantânea.

#### Abordagem B: Com LOD Base (Cache de Visão Geral / Overview Cache)
* **Como funciona:** Define um LOD ou resolução base fixa (ex: $\approx 1.500 \times 1.500$ pixels ou o tamanho que caiba na Viewport) para ser a referência de todo o modo de Zoom Out.
* **Comportamento do Cache:** A composição local dos edits ocorre **uma única vez** nessa resolução base de visão geral. Qualquer variação contínua de zoom out ou movimentação de câmera (Pan) reusa estritamente este único buffer mantido em RAM, aplicando apenas o *warp* de redimensionamento no OpenCV.
* **Vantagens:** Zero invalidação de cache em qualquer mudança de zoom out, reduzindo drasticamente o reprocessamento e mantendo consumo de RAM irrisório (~9 MB por camada).
* **Gerenciamento:** O buffer base é mantido até que um edit seja alterado/adicionado ou o zoom ultrapasse o limiar de Zoom In (1:1).

### 2.2. Tiled Pass (Grade de Tiles - Zoom In / Alta Resolução)
* **Critério de Ativação:** O usuário aproxima o zoom (Zoom In), exigindo a leitura em alta resolução (LOD 0 ou LOD 1), onde o tamanho da área visível tornaria a alocação de um único buffer inviável.
* **Funcionamento:**
  1. Ativa a grade de tiles (`TileGrid`).
  2. Calcula a interseção entre o retângulo visível da Viewport (ROI) e a grade.
  3. Renderiza e compõe estritamente os 4 a 9 tiles de $512 \times 512$ visíveis na tela.
* **Benefícios:** Consumo fixo e mínimo de memória RAM (poucos megabytes), mesmo navegando em imagens de dezenas de gigabytes.

---

## 3. Integração com as Classes do Motor

* **`LODManager` / `EditLayer`:** Fornece a imagem na resolução adequada para a escala solicitada.
* **`ViewportRender` / `LayerRender`:** Avalia o tamanho da camada no LOD retornado. Se couber no limiar, executa o `Direct Pass`. Caso contrário, ativa a iteração de `TileGrid`.
