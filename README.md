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

## 🚀 Guia Rápido & Exemplos de Uso

### 1. Criando um Documento e Manipulando Camadas

```python
from anicrop import Document, Layer, Image, ImageFormat
from anicrop.enums import BlendMode

# Cria um novo documento (1920x1080) com fundo transparente
doc = Document("MeuProjeto", width=1920, height=1080)

# Carrega camadas diretamente do disco
fundo = doc.load_layer("assets/background.jpg", name="Fundo")
personagem = doc.load_layer("assets/character.png", name="Personagem")

# Aplica opacidade e modo de mesclagem
personagem.opacity = 0.95
personagem.blend_mode = BlendMode.NORMAL
```

---

### 2. Transformações Espaciais e Layout

```python
from anicrop.enums import FitMode, HAlign, VAlign

# Encadeamento direto de transformações afins 3x3 na camada
personagem.transform.scale(1.2, 1.2).rotate(15).translate(100, 50)

# Ajuste espacial inteligente (Fit) e alinhamento via motor de Layout
doc.layout.fit(personagem, doc.canvas, mode=FitMode.CONTAIN)
doc.layout.align(personagem, doc.canvas, halign=HAlign.CENTER, valign=VAlign.MIDDLE)
```

---

### 3. Corte Não-Destrutivo (`Content.crop`)

```python
from anicrop.spatial import Region

# Define uma região de interesse e aplica o corte não-destrutivo
# (preserva a imagem original e ativa uma máscara EditLayer com BlendMode.CLIP)
corte_roi = Region(x=100, y=100, width=500, height=400)
doc.content.crop(personagem, corte_roi)
```

---

### 4. Filtros e Efeitos Ancorados (`BlurFilter`)

```python
from anicrop.filter import BlurFilter

# Adiciona um filtro de desfoque Gaussiano ancorado à geometria da camada
desfoque = BlurFilter(sigma_x=10.0, sigma_y=10.0)
personagem.bind_effect(desfoque)
```

---

### 5. Agrupamento, Mesclagem e Bake (`doc.combine`)

```python
# 1. Agrupa camadas (Merge Down): mescla 'Personagem' com 1 camada visível abaixo
grupo = doc.combine.merge("Personagem", name="GrupoPersonagem", count=1)

# 2. Assa o grupo em uma única camada plana substituindo-o na árvore
camada_plana = doc.combine.bake("GrupoPersonagem")

# 3. Ou achata toda a pilha do documento (Flatten Image)
camada_final = doc.combine.bake_stack(name="ArteFinal")
```

---

### 6. Organização e Inversão da Pilha (`doc.stack`)

```python
# Desloca a camada 2 posições para cima na pilha
doc.stack.move_relative(personagem, steps=2)

# Envia diretamente para o topo ou base
doc.stack.move_to_front(personagem)
doc.stack.move_to_back(fundo)

# Troca a posição de duas camadas
doc.stack.swap(personagem, fundo)

# Inverte a ordem das camadas (com recursive=True descendo em subgrupos)
doc.stack.reverse(recursive=True)
```

---

### 7. Renderização, Exportação e Visualização Interativa

```python
from anicrop import Viewer

# Renderiza a cena completa do documento em alta resolução
resultado = doc.render(format=ImageFormat.RGBA)

# Salva a imagem rasterizada no disco
resultado.save("output/composicao_final.png", quality=95)

# Abre a janela do visualizador interativo OpenCV
Viewer(doc).show()
```

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
