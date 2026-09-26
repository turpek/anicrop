import time
from pathlib import Path

from anicrop import Document, ImageFormat, LayerCache
from anicrop.content import FitContext
from anicrop.filter import BlurFilter

ASSETS_DIR = Path(__file__).parent

# 1. Abre o Documento herdando as dimensões do background (1376x768)
doc = Document.open(ASSETS_DIR / "background.jpg", name="Fundo")

# 2. Carrega as camadas: Personagem e Chapéu
personagem = doc.load_layer(ASSETS_DIR / "character.png", name="Personagem")
chapeu = doc.load_layer(ASSETS_DIR / "hat.png", name="Chapeu")

# 3. Redimensiona o chapéu e alinha na cabeça da personagem
chapeu.content.resize(250, 250)
chapeu.layout.pin((520, 196), anchor_x=0.5, anchor_y=1.0)

# 4. Agrupa o chapéu com a personagem em um grupo não-destrutivo
heroina = doc.combine.merge("Chapeu", name="Heroina", count=1)

# 5. Enquadra a heroína proporcionalmente (fit_contain) e alinha à direita
fit_payload = FitContext(heroina, doc.canvas).fit_contain()
heroina.content.fit(fit_payload)
heroina.layout.align(doc.canvas, anchor_x=0.85, anchor_y=1.0)

# 6. Aplica desfoque de profundidade de campo no fundo (filtro pesado)
fundo = doc["Fundo"]
fundo.bind_effect(BlurFilter(radius=4.0))

# 7. Inicializa o LayerCache e registra toda a pilha de camadas
cache = LayerCache()
cache.register(doc.stack)

# 8. Rotaciona o grupo da heroína em 10 graus
heroina.transform.rotate(10)

# 9. Frame 1: Primeira renderização com cache (aquece o cache e assa rotação de 10° e blur)
t0 = time.perf_counter()
frame1 = doc.render(format=ImageFormat.RGBA, cache=cache)
t1 = time.perf_counter()
tempo_frame1 = (t1 - t0) * 1000.0

frame1.save(ASSETS_DIR / "cena_cache_frame1.png")
print(f"[Frame 1] Renderizado em {tempo_frame1:.2f} ms (Heroina rotacionada 10°, Fundo com Blur)")

# 10. Prova do uso do Cache: Apenas translada o grupo da heroína no Canvas
heroina.transform.translate(-250, 0)

# 11. Valida antes do Frame 2: Nenhuma camada está dirty (a matriz 2x2 de rotação/escala é idêntica!)
print("-> Verificação de dirty antes do Frame 2:")
print(f"   Fundo dirty? {cache.is_dirty(fundo)}")
print(f"   Personagem dirty? {cache.is_dirty(doc['Personagem'])} (Warp de 10° preservado!)")
print(f"   Chapeu dirty? {cache.is_dirty(doc['Chapeu'])} (Warp de 10° preservado!)")

# 12. Frame 2: Segunda renderização com cache ativo
# Tanto o Fundo quanto a Personagem e o Chapéu reutilizam seus buffers assados!
t2 = time.perf_counter()
frame2 = doc.render(format=ImageFormat.RGBA, cache=cache)
t3 = time.perf_counter()
tempo_frame2 = (t3 - t2) * 1000.0

frame2.save(ASSETS_DIR / "cena_cache_frame2.png")
print(f"[Frame 2] Renderizado em {tempo_frame2:.2f} ms (Heroina transladada via Cache)")
print(f"-> Speedup com Cache: {tempo_frame1 / tempo_frame2:.1f}x mais rápido!")
print(f"Frames salvos em:\n - {ASSETS_DIR / 'cena_cache_frame1.png'}\n - {ASSETS_DIR / 'cena_cache_frame2.png'}")
