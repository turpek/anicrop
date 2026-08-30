# anicrop

**anicrop** é um motor gráfico (*engine/core*) em Python para manipulação, composição e edição não-destrutiva de imagens 2D de alta performance.

Projetado com rigor matemático e precisão geométrica, o motor utiliza uma árvore de camadas baseada no padrão *Composite*, transformações afins 3x3 sem acúmulo de erro (*Size Drift*), sistema modular de I/O acelerado por SIMD via `libvips`, backend híbrido de memória (NumPy / Zarr para imagens gigantes) e pipeline de renderização flexível.

---

## ✨ Principais Recursos

- 🌳 **Gerenciamento Hierárquico de Camadas:** Árvore espacial completa (`LayerStack`, `GroupLayer`, `Layer`, `EditLayer`) com cálculo dinâmico de limites e matrizes relativas pai-filho.
- 📐 **Transformações Afins de Alta Precisão:** Matrizes 3x3 homogêneas para Rotação, Escala, Cisalhamento (*Skew*) e Translação com pivôs arbitrários sem degradação cumulativa.
- 🎨 **Motores Especializados no Documento:**
  - **`doc.layout`**: Enquadramento e alinhamento espacial puro (`fit`, `align`, `resize_bounds`, `fit_content`).
  - **`doc.content`**: Manipulação e corte não-destrutivo de pixels (`crop` via `BlendMode.CLIP`, `resize`, `fit`, `flip_x`, `flip_y`).
  - **`doc.combine`**: Orquestrador de composição e fusão na árvore de camadas (`merge`, `flatten`, `bake`, `bake_stack`).
- ⚡ **I/O Modular de Alta Performance:** Decodificação e subamostragem direta (*shrink-on-load*) nativa em C/SIMD via `PyvipsBackend` (até **58× mais rápido**) com fallback transparente para `OpenCVBackend`.
- 💾 **Backend Híbrido & LOD:** Chaveamento transparente de buffers para Zarr em imagens gigantes ($\ge 8192\text{px}$) com pirâmide de nível de detalhe (*Level of Detail*).
- 🎭 **Máscaras e Filtros Anisotrópicos:** Efeitos ancorados à matriz da camada (`BoundEffect`), filtros Gaussianos com fusão de tensores de covariância 2D (`BlurFilter`) e máscaras atômicas.
- 🔄 **Organização Fluida da Pilha:** Métodos declarativos no contêiner (`move_relative`, `move_to_front`, `move_to_back`, `swap`, `reverse`).
- 👁️ **Pipeline de Renderização & Visualizador:** `CanvasRender` para exportações em alta resolução, `ViewportRender` para previews interativos e visualizador OpenCV `Viewer`.

---

## 📦 Instalação

Recomenda-se o uso do [`uv`](https://github.com/astral-sh/uv) para gerenciamento do ambiente:

```bash
uv add numpy opencv-python pyvips zarr pillow loguru
```

---

## 🚀 Exemplo Completo de Composição (End-to-End)

O fluxo abaixo demonstra a composição não-destrutiva de uma cena: criando o canvas a partir do background, redimensionando e ancorando elementos via `.content` e `.layout`, achatando camadas (`flatten`), aplicando enquadramento proporcional (`fit_contain`), filtros de profundidade e exportando a imagem final:

```python
from anicrop import Document, ImageFormat, Viewer
from anicrop.content import FitContext
from anicrop.filter import BlurFilter

# 1. Abre o Documento herdando as dimensões exatas do background (1376x768)
doc = Document.open("assets/background.jpg", name="Fundo")

# 2. Carrega as camadas: Personagem e Chapéu (no topo)
personagem = doc.load_layer("assets/character.png", name="Personagem")
chapeu = doc.load_layer("assets/hat.png", name="Chapeu")

# 3. Redimensiona o chapéu diretamente via .content da camada
chapeu.content.resize(250, 250)

# 4. Alinha o chapéu no topo da cabeça da personagem diretamente via .layout
chapeu.layout.align(personagem, anchor_x=0.51, anchor_y=-0.07)

# 5. Achata o chapéu com a personagem em uma única camada (Merge Down: Chapéu + 1 camada abaixo)
heroina = doc.combine.flatten("Chapeu", name="Heroina", count=1)

# 6. Enquadra a heroína proporcionalmente para caber na altura do Canvas (fit_contain)
fit_payload = FitContext(heroina, doc.canvas).fit_contain
heroina.content.fit(fit_payload)

# 7. Posiciona a heroína no lado direito do cenário diretamente via .layout
heroina.layout.align(doc.canvas, anchor_x=0.85, anchor_y=1.0)

# 8. Profundidade de campo: aplica desfoque suave no fundo
fundo = doc["Fundo"]
fundo.bind_effect(BlurFilter(radius=3.0))

# 9. Renderiza a cena e salva no disco
resultado = doc.render(format=ImageFormat.RGBA)
resultado.save("assets/cena_final.png")

# 10. (Opcional) Visualiza interativamente na janela OpenCV
Viewer(doc).show()
```

---

## 🛠️ Motores e Operações Principais

### Disposição Espacial e Moldura (`layer.layout` / `doc.layout`)
Opera exclusivamente sobre a moldura lógica e o alinhamento de nós no Espaço Global sem alterar pixels:
- `layer.layout.fit(ref)`: Enquadra a moldura da camada dentro da região de referência mantendo o aspect ratio.
- `layer.layout.align(ref, anchor_x=0.5, anchor_y=0.5)`: Alinha a posição global da camada com base em âncoras normalizadas.
- `layer.layout.resize_bounds(width, height)`: Redimensiona a moldura lógica ancorada.

### Manipulação de Pixels e Conteúdo (`layer.content` / `doc.content`)
Operações sobre o conteúdo visual, transformações afins e recortes não-destrutivos:
- `layer.content.crop(regiao)`: Corte não-destrutivo aplicando máscara `EditLayer` com `BlendMode.CLIP`.
- `layer.content.resize(width, height)`: Redimensiona a escala global da imagem.
- `layer.content.fit(ref)`: Ajusta a escala e translada a camada para preencher a região de referência.
- `layer.content.flip_x()` / `layer.content.flip_y()`: Espelhamento horizontal e vertical imediato.

### Composição e Fusão (`doc.combine`)
Orquestra fusão, agrupamento e rasterização na árvore de camadas:
- `doc.combine.merge(alvo, name="Grupo", count=1)`: Mescla o alvo com $N$ camadas visíveis abaixo em um `GroupLayer`.
- `doc.combine.flatten(alvo, name="Plano", count=1)`: Achata o alvo e as camadas abaixo em uma única camada folha.
- `doc.combine.bake("Grupo")`: Assa o conteúdo interno de um grupo e o substitui por um `Layer` rasterizado.
- `doc.combine.bake_stack(name="Final")`: Achata toda a pilha do documento em uma camada plana única.

### Manipulação da Pilha de Camadas (`doc.stack`)
- `doc.stack.move_relative(camada, steps=1)`: Desloca a camada para cima ou para baixo.
- `doc.stack.move_to_front(camada)` / `doc.stack.move_to_back(camada)`: Move para o topo ou base da pilha.
- `doc.stack.swap(camada_a, camada_b)`: Troca a posição de duas camadas.
- `doc.stack.reverse(recursive=True)`: Inverte a ordem das camadas com suporte opcional a recursão profunda.

---

## 🏛️ Arquitetura do Sistema

```
                    ┌─────────────────────────┐
                    │        Document         │
                    │ (Fachada Principal/API) │
                    └────────────┬────────────┘
         ┌───────────────┬───────┴───────┬───────────────┐
         ▼               ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │  Layout   │   │  Content  │   │  Combine  │   │  Render   │
   │ (Espacial)│   │ (Pixels)  │   │ (Fusão)   │   │ (Câmera)  │
   └───────────┘   └───────────┘   └───────────┘   └───────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │      LayerStack       │
             │   (Container Raiz)    │
             └───────────┬───────────┘
         ┌───────────────┴───────────────┐
         ▼                               ▼
   ┌───────────┐                   ┌───────────┐
   │   Layer   │                   │GroupLayer │
   │  (Folha)  │                   │(Composto) │
   └───────────┘                   └───────────┘
```
