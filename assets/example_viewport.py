from pathlib import Path

from anicrop import Document, Viewport
from anicrop.spatial import Region
from anicrop.type import Scale

ASSETS_DIR = Path(__file__).parent

# 1. Carrega a imagem gerada pelo example.py herdando as dimensões exatas do Canvas (1376x768)
doc = Document.open(ASSETS_DIR / "cena_final.png", name="Cena")

# 2. Parâmetros da Viewport e Ponto Focal
VIEWPORT_SIZE = (886, 688)
FOCAL_POINT = (920, 130)  # Centro da cabeça da heroína no Canvas
ZOOM_FACTOR = 4.10  # 410% de zoom (escala 4.1x)

# 3. Inicializa a Viewport com a dimensão solicitada
viewport = Viewport(size=VIEWPORT_SIZE, fit_scale=1.0)

# 4. Aplica o Zoom de 410% (Scale 4.1x)
viewport.scale = Scale(ZOOM_FACTOR, ZOOM_FACTOR)

# 5. Move a Câmera (Pan) para centralizar o ponto focal no centro da Viewport
# Como a Viewport (fit_matrix) centraliza por padrão o Canvas no meio da janela,
# o pan (dx, dy) é o deslocamento do ponto focal em relação ao CENTRO do Canvas:
# dx = focal_x - (canvas_width / 2)
# dy = focal_y - (canvas_height / 2)
canvas_w, canvas_h = doc.canvas.size
view_w, view_h = VIEWPORT_SIZE
focal_x, focal_y = FOCAL_POINT

viewport.region = Region.from_rect(
    focal_x - (canvas_w / 2),
    focal_y - (canvas_h / 2),
    view_w,
    view_h,
)

# 6. Renderiza a janela de exibição da Viewport e salva no disco
preview = doc.preview(viewport)
preview.save(ASSETS_DIR / "heroina_zoom.png")

print(
    f"Preview com zoom de 410% renderizado e salvo em: {ASSETS_DIR / 'heroina_zoom.png'}"
)
