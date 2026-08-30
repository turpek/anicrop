# Plano de Otimização: Renderização, Caching e Oclusão Conservadora

Este documento contém o plano técnico detalhado e o contexto de arquitetura discutido para a implementação da otimização de renderização de camadas (Layers) no projeto **anicrop**. Use este plano para guiar o desenvolvimento.

---

## 1. Visão Geral da Arquitetura de Renderização

Atualmente, o projeto possui dois renderizadores principais em `src/anicrop/render.py`:

*   **`LayerRender`**: Renderiza a imagem final de uma camada em resolução cheia no espaço do Canvas. É usado para exportação e composição estática.
*   **`ViewportRender`**: Projeta as camadas diretamente na tela (Viewport) aplicando Zoom, Pan e Centralização, utilizando o `LODManager` para decimar imagens quando o nível de zoom está muito baixo.

### A Regra de Ouro: Separação de Caches

```text
  [ Assets de Edição: Imagem Base + Edits ]
                     │
                     ▼  (Fase 1: Composição Local - CARA, mas RARA)
       [ Local Cache: Imagem Plana do Layer ]
                     │
                     ▼  (Fase 2: Warp Espacial - EXTREMAMENTE RÁPIDA)
       [ Matriz Global: Giro, Escala, Pan ]
                     │
                     ▼
       [ Viewport Final / Imagem de Saída ]
```

1.  **Composição Local (Fase 1)**: Aplica edits, máscaras e filtros locais de cor sobre a imagem base. Deve ser recalculada apenas quando os edits ou filtros sofrerem alteração.
2.  **Warp Espacial (Fase 2)**: Aplica a matriz de translação/rotação/escala do Layer sobre o buffer plano pré-composto da Fase 1. É executada continuamente durante a navegação/edição (zoom, pan, rotação).

---

## 2. Otimização por Oclusão (Early-Exit)

Ao desenhar a pilha de layers (`LayerStack`), percorremos de cima para baixo (do topo para o fundo) para verificar se algum layer no topo obstrui completamente a visão da viewport. Se obstruir, podemos interromper a renderização e ignorar todos os layers abaixo dele.

### Algoritmo Front-to-Back de Varredura

```python
def resolver_layers_visiveis(
    layers: list[Layer], viewport_region: Region
) -> list[Layer]:
    layers_a_desenhar = []

    # Varremos do topo (mais visível) para o fundo
    for layer in reversed(layers):
        layers_a_desenhar.append(layer)

        # Critérios para oclusão total:
        # 1. Modo de mesclagem NORMAL (substitui o fundo)
        # 2. Opacidade 100% (1.0)
        # 3. O layer cobre toda a viewport
        # 4. A miniatura de opacidade indica 100% de cobertura opaca
        if (
            layer.blend_mode == BlendMode.NORMAL
            and layer.opacity == 1.0
            and layer.canvas_region.contains(viewport_region)
            and verificar_cobertura_opaca(layer, viewport_region)
        ):
            # Encontramos um layer totalmente opaco que tampa a visão.
            # Ignoramos todos os layers que estão abaixo dele na pilha.
            break

    return list(reversed(layers_a_desenhar))
```

---

## 3. Oclusão Conservadora: Evitando Falsos-Positivos

Se verificarmos apenas a Bounding Box (`canvas_region`), um layer com transparências internas (canal Alpha < 255) causará o desaparecimento incorreto dos layers de fundo (**falso-positivo de oclusão**).

Para resolver isso de forma performática na CPU, mantemos uma **miniatura de opacidade** de baixíssima resolução (ex: $32 \times 32$ pixels, consumindo apenas 1 KB de RAM) que armazena o canal Alpha do Layer reduzido de forma **conservadora**.

### O que é uma Redução Conservadora (Min-Pooling)?
Um pixel na miniatura só será considerado $255$ (totalmente opaco) se **todos** os pixels originais daquele bloco correspondente forem exatamente $255$. Se houver sequer um único pixel semi-transparente ou transparente no bloco original, a miniatura registrará um valor menor que $255$, impedindo o culling indevido.

### Implementações do Algoritmo de Min-Pooling

#### Opção A: Usando NumPy Puro (Reshape + Min)
```python
import numpy as np


def gerar_miniatura_alpha_conservadora(
    alpha_original: np.ndarray, target_size: tuple[int, int] = (32, 32)
) -> np.ndarray:
    h, w = alpha_original.shape[:2]
    th, tw = target_size

    block_h = h // th
    block_w = w // tw

    # Trunca para ser múltiplo exato
    truncated_h = block_h * th
    truncated_w = block_w * tw
    crop_alpha = alpha_original[:truncated_h, :truncated_w]

    # Reshape para agrupar em blocos e tira o mínimo nos eixos internos dos blocos
    reshaped = crop_alpha.reshape(th, block_h, tw, block_w)
    return np.min(reshaped, axis=(1, 3))
```

#### Opção B: Usando OpenCV (Erosão Morfológica + Nearest Resize)
```python
import cv2
import numpy as np


def gerar_miniatura_opencv(
    alpha_original: np.ndarray, target_size: tuple[int, int] = (32, 32)
) -> np.ndarray:
    h, w = alpha_original.shape[:2]
    th, tw = target_size

    kernel_h = max(1, h // th)
    kernel_w = max(1, w // tw)
    kernel = np.ones((kernel_h, kernel_w), dtype=np.uint8)

    # A erosão propaga os pixels de menor opacidade (mais escuros)
    eroded_alpha = cv2.erode(alpha_original, kernel)

    # Amostra via vizinho mais próximo
    return cv2.resize(eroded_alpha, target_size, interpolation=cv2.INTER_NEAREST)
```

---

## 4. Passo a Passo para Implementação

Para implementar este plano, siga estes passos no código:

1.  **Armazenar a miniatura no cache do Layer**:
    No `Layer` (em `src/anicrop/layer.py`), adicione uma propriedade `_opacity_mask` (inicialmente `None`).
2.  **Atualizar a miniatura na Fase 1**:
    No `LayerRender.render()` (em `src/anicrop/render.py`), quando o cache de pixels local for reconstruído (`RenderFlags.PIXELS`), extraia o canal Alpha da imagem resultante e gere a miniatura utilizando um dos algoritmos acima, armazenando-a no layer.
3.  **Implementar o `verificar_cobertura_opaca`**:
    Escreva a lógica que projeta a área útil da Viewport de volta para o espaço local do Layer, extrai a sub-região correspondente na miniatura de opacidade de $32 \times 32$ e valida com `np.all(sub_miniatura == 255)`.
4.  **Integrar na Composição da Pilha**:
    Atualize o método de renderização da pilha de camadas (em `src/anicrop/layer_stack.py` ou no loop de composição principal) para varrer os layers de cima para baixo antes de enviar as chamadas de renderização.
