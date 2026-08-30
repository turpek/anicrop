# Análise Técnica: Lanczos Boundary Padding e Artefatos de Ringing nas Bordas

Este documento registra as descobertas matemáticas, comportamentais e arquiteturais realizadas durante a investigação dos artefatos de borda (linhas claras e haloing) no motor de renderização `render_patch` do **anicrop**.

---

## 1. O Problema Observado

Durante os testes visuais de renderização da Viewport (`test7` e `test8`), foi observada uma linha fina clara (esbranquiçada/cinzenta) localizada a 1 ou 2 pixels antes da borda física inferior da camada renderizada, especialmente visível quando a camada era composta sobre um fundo branco.

---

## 2. Investigação e Causa Raiz

### 2.1. O Fenômeno de Ringing do Lanczos (Overshoot)
* O filtro `cv2.INTER_LANCZOS4` utiliza um kernel baseado na função Sinc 1D estendida para 2D:

  $$\text{Lanczos}(x) = \text{sinc}(x) \cdot \text{sinc}\left(\frac{x}{a}\right)$$

* A função Sinc oscila entre valores positivos e negativos (lóbulos negativos).
* Quando uma camada é recortada e sofre padding com zeros constantes (`mode='constant', constant_values=0`), a transição brusca de cores/opacidade para o zero externo faz com que o lóbulo negativo do Lanczos cause uma oscilação na curva de interpolação (*Gibbs Phenomenon / Overshoot*).
* Essa oscilação gera um pico de luminância nos pixels situados a $1.5 \sim 2.5$ pixels de distância da borda física, originando a linha clara no antepenúltimo pixel.

### 2.2. A Matemática dos canais RGB vs Alpha
* Ao aplicar `mode='edge'` (repetição de borda) no canal **RGB**, o lóbulo negativo do Lanczos encontra valores de cor contínuos, eliminando a oscilação de cor.
* No entanto, se `mode='edge'` for aplicado no canal **Alpha** em camadas rotacionadas, os cantos externos do Bounding Box recebem Alpha opaco ($255$) em vez de transparente ($0$), violando a transparência dos cantos externos poligonais de um layer girado a 45°.

---

## 3. Comparativo dos Algoritmos de Interpolação no Downscaling

Foram realizados testes numéricos diretos sobre o buffer de pixels na antepenúltima linha (`y=464` na Viewport):

| Algoritmo | Valor BGR (`y=464`) | Comportamento na Borda |
| :--- | :--- | :--- |
| **`InterpolationOption.LANCZOS`** | `BGR = [55, 62, 78]` | ❌ Exibe linha clara por ringing Sinc (*overshoot*) |
| **`InterpolationOption.CUBIC`** | `BGR = [ 6, 37, 66]` | ✅ Transição suave e perfeita (sem linha clara) |
| **`InterpolationOption.AREA`** | `BGR = [ 6, 39, 67]` | ✅ Anti-aliasing por média de área (sem linha clara) |
| **`InterpolationOption.LINEAR`** | `BGR = [ 6, 39, 67]` | ✅ Suave (sem linha clara) |

---

## 4. Práticas Recomendadas para o Motor

1. **Separação de Contextos de Interpolação:**
   * **Viewport / Zoom-Out (Escala $< 1.0$):** Utilizar por padrão `InterpolationOption.CUBIC` ou `InterpolationOption.AREA`. Eles fornecem excelente nitidez, são mais rápidos e não geram linhas de anelamento nas bordas.
   * **Exportação em Alta Resolução / Rotações Puras:** Permitir `InterpolationOption.LANCZOS` ou `CUBIC` conforme a preferência do usuário.
2. **Paridade com Editores Profissionais (GIMP/Photoshop):**
   * Tanto o GIMP (via GEGL No-halo / Lo-halo) quanto o Photoshop (via Bicubic Sharper) utilizam variações do algoritmo Cúbico com trava de sinal (*clamping*) para prevenir o efeito de anelamento durante transformações de malha.
