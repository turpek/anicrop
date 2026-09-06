from pathlib import Path

from anicrop import Document, ImageFormat
from anicrop.content import FitContext
from anicrop.filter import BlurFilter

ASSETS_DIR = Path(__file__).parent

# 1. Abre o Documento herdando as dimensões exatas do background (1376x768)
doc = Document.open(ASSETS_DIR / "background.jpg", name="Fundo")

# 2. Carrega as camadas: Personagem e Chapéu (no topo)
personagem = doc.load_layer(ASSETS_DIR / "character.png", name="Personagem")
chapeu = doc.load_layer(ASSETS_DIR / "hat.png", name="Chapeu")

# 3. Redimensiona o chapéu diretamente via .content da camada
chapeu.content.resize(250, 250)

# 4. Alinha a base inferior do chapéu no ponto global da cabeça via .layout.pin
PONTO_CABECA = (520, 196)
chapeu.layout.pin(PONTO_CABECA, anchor_x=0.5, anchor_y=1.0)

# 5. Agrupa o chapéu com a personagem em um grupo não-destrutivo (Merge Down: Chapéu + 1 camada abaixo)
heroina = doc.combine.merge("Chapeu", name="Heroina", count=1)

# 6. Enquadra a heroína proporcionalmente para caber na altura do Canvas (fit_contain)
fit_payload = FitContext(heroina, doc.canvas).fit_contain()
heroina.content.fit(fit_payload)

# 7. Posiciona a heroína no lado direito do cenário diretamente via .layout
heroina.layout.align(doc.canvas, anchor_x=0.85, anchor_y=1.0)

# 8. Profundidade de campo: aplica desfoque suave no fundo
fundo = doc["Fundo"]
fundo.bind_effect(BlurFilter(radius=3.0))

# 9. Renderiza a cena e salva no disco
resultado = doc.render(format=ImageFormat.RGBA)
resultado.save(ASSETS_DIR / "cena_final.png")

print(f"Cena final renderizada e salva em: {ASSETS_DIR / 'cena_final.png'}")
