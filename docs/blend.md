# Guia de Modos de Mesclagem (`blend.py` & `BlendMode`)

O subsistema de mesclagem (`anicrop.blend` e `anicrop.enums.BlendMode`) é o motor responsável pela composição matemática e fusão de pixels entre camadas (`Layer`, `GroupLayer`), patches de edição (`EditLayer`) e buffers de renderização (`CanvasRender`, `ViewportRender`).

---

## 1. Modos de Mesclagem Disponíveis (`BlendMode`)

O enum [`BlendMode`](file:///home/gui/python/anicrop/src/anicrop/enums.py) define as operações de mistura suportadas pelo motor:

| Modo de Mesclagem | Nome String | Prioridade | Comportamento Principal | Casos de Uso Recomendados |
| :--- | :--- | :--- | :--- | :--- |
| **`BlendMode.NORMAL`** | `"normal"` | Camada Superior | Composição padrão Porter-Duff ($A \text{ over } B$) com interpolação de canal alfa. | Camadas gerais, ilustrações, gráficos com transparência gradual. |
| **`BlendMode.NORMAL_LINEAR`** | `"normal_linear"` | Camada Superior | Composição Porter-Duff calculada no espaço linear de luminosidade (gama corrigido). | Composição de alta fidelidade física que evita escurecimento em bordas translúcidas. |
| **`BlendMode.MULTIPLY`** | `"multiply"` | Múltipla | Multiplica os valores normalizados de cor da base e do overlay ($C = C_{\text{base}} \times C_{\text{overlay}}$). | Sombras, texturização, sobreposição de rascunhos. |
| **`BlendMode.HARD_MASKING`** | `"hard_masking"` | **Top-First** *(Overlay)* | Substituição binária direta de pixels: onde o overlay possui $\alpha > 0$, ele **sobrescreve** a base. | Adesivos (*stickers*), marcas d'água, *chroma key*, recortes duros (*1-bit alpha*). |
| **`BlendMode.SOLID_FILL`** | `"solid_fill"` | **Base-First** *(Canvas)* | Preenchimento protegido: o canvas consolidado ($\alpha \ge 250$) é **intocável**; o overlay só preenche vazios ($\alpha \ge 200$) e fixa $\alpha = 255$. | **Costura de Panoramas (*Stitching* / Mosaicos)**, reconstrução contínua de animes sem franjas. |
| **`BlendMode.CLIP`** | `"clip"` | Modulação | Modula o canal alfa da base onde houver transparência e preenche com branco sólido as áreas cortadas de camadas RGB. | Recorte de pixels não-destrutivo (`Content.crop`). |

---

## 2. Comparativo: `HARD_MASKING` vs `SOLID_FILL`

Enquanto ambos os modos operam com binarização rápida de pixels (sem o custo de blend alpha fracionário), eles atendem a direções de prioridade **opostas**:

```text
HARD_MASKING (Top-First):
   [ Base Existente ]  <--- Sobrescrito por --->  [ Novo Overlay (alpha > 0) ]
   (O que está em cima substitui o que está embaixo)

SOLID_FILL (Base-First):
   [ Base Sólida (alpha >= 250) ]  <--- Intocável!
   [ Buracos Transparentes ]        <--- Preenchidos por ---> [ Novo Overlay (alpha >= 200) ]
   (O que já foi desenhado na base é protegido; o novo frame só preenche lacunas)
```

### Por que o `SOLID_FILL` elimina 100% das franjas em panoramas?
Ao rotacionar frames com algoritmos de interpolação contínua (`LANCZOS`, `CUBIC`), as bordas extremas ganham pixels semitransparentes de *antialiasing* ($\alpha \approx 20 \dots 80$).
* No `HARD_MASKING`: esses pixels fracos sobrescreviam o canvas sólido anterior, degradando a opacidade acumulada em dezenas de frames.
* No `SOLID_FILL`: o canvas opaco consolidado é bloqueado contra alterações, a penumbra fraca ($\alpha < 200$) é descartada e os novos pixels válidos recebem $\alpha = 255$ puro, gerando costuras perfeitamente contínuas e sólidas.

---

## 3. Arquitetura e Aceleração de Performance

O módulo `anicrop.blend` implementa arquitetura híbrida de execução:

```mermaid
flowchart TD
    Call["blend_mode(base, overlay, opacity)"] --> CheckCython{"Extensão Cython Compilada?"}
    CheckCython -- "Sim (_HAS_CY_BLEND)" --> Native["C / Cython SIMD (blend.pyx)<br>• nogil (Zero Python Overhead)<br>• OpenMP Multi-core (prange)<br>• 32-bit Integer Word Loads/Stores"]
    CheckCython -- "Não (Fallback)" --> PureNumPy["NumPy Vectorized (blend.py)<br>• Operações vetorizadas em C puro"]
```

### Características Técnicas do Kernel Cython:
1. **Zero GIL Overhead (`nogil`):** Toda a mesclagem é executada em código de máquina C nativo compilado com `-O3 -march=native -ffast-math`.
2. **Paralelismo Multi-Core (`OpenMP prange`):** O processamento de linhas da imagem é distribuído entre todos os núcleos da CPU.
3. **Carga e Descarga Vetorial de 32-bit:** Em modos de substituição como `SOLID_FILL` e `HARD_MASKING`, as operações em buffers RGBA lêem e gravam palavras de 32 bits (`uint32_t`) em uma única instrução Assembly (`MOV`), habilitando auto-vetorização AVX2/SSE4 pelo compilador.

---

## 4. Exemplos Práticos de Uso

### 4.1. Definindo o Modo de Mesclagem em uma Camada
```python
from anicrop import Document, Image, ImageFormat, Layer
from anicrop.enums import BlendMode

# Criar camada com modo SOLID_FILL para stitching
layer = Layer(
    Image.open("frame_02.png", format=ImageFormat.RGBA),
    blend_mode=BlendMode.SOLID_FILL,
    name="Frame02",
)

# Ou alterar diretamente na propriedade:
layer.blend_mode = BlendMode.MULTIPLY
```

### 4.2. Usando em Patches Não-Destrutivos (`add_edit`)
```python
# Adiciona um patch que substitui os pixels locais com corte duro (HARD_MASKING)
layer.add_edit(patch_img, patch_region, blend_mode=BlendMode.HARD_MASKING)
```

### 4.3. Usando na Composição e Flatten
```python
from anicrop.composition import flatten
from anicrop.enums import BlendMode, ImageFormat

# Flatten preserva automaticamente o blend_mode da base e o ImageFormat do topo
cena_plana = flatten([fundo, camada_detalhe], name="CenaConsolidada")
```
