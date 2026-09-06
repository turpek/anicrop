from pathlib import Path

from anicrop import Document, Viewport

ASSETS_DIR = Path(__file__).parent

# 1. Carrega a imagem gerada pelo example.py herdando as dimensões exatas do Canvas (1376x768)
doc = Document.open(ASSETS_DIR / "cena_final.png", name="Cena")

# 2. Parâmetros da Viewport e Ponto Focal
VIEWPORT_SIZE = (886, 688)
FOCAL_POINT = (920, 130)  # Centro da cabeça da heroína no Canvas
ZOOM_FACTOR = 4.10  # 410% de zoom (escala 4.1x)

# 3. Inicializa a Viewport com a dimensão solicitada observando o Canvas do documento
viewport = Viewport(size=VIEWPORT_SIZE, canvas=doc.canvas)

# 4. Aplica o Zoom de 410% (4.1x) diretamente
viewport.zoom = ZOOM_FACTOR

# 5. Move a Câmera (Pan) fixando o ponto focal no centro da Viewport via .layout.pin
viewport.layout.pin(FOCAL_POINT)

# 6. Renderiza a janela de exibição da Viewport e salva no disco
preview = doc.preview(viewport)
preview.save(ASSETS_DIR / "heroina_zoom.png")

print(
    f"Preview com zoom de 410% renderizado e salvo em: {ASSETS_DIR / 'heroina_zoom.png'}"
)
