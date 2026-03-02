import numpy as np
import pytest
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.spatial import Region, Span
from anicrop.render import LayerRender

def make_render():
    return LayerRender()

@pytest.fixture
def red_layer():
    """Cria um layer vermelho 100x100 para testes."""
    data = np.zeros((100, 100, 4), dtype=np.uint8)
    data[:] = [255, 0, 0, 255] # Vermelho
    img = Image(data, ImageFormat.RGBA)
    return Layer(img)

def test_render_com_transform_translation_delta(red_layer):
    # Posição inicial (âncora) em (50, 50)
    red_layer.x = 50
    red_layer.y = 50
    
    # Aplica um delta de translação via matriz (+10, +10)
    red_layer.transform.translate(10, 10)
    
    # O BBox final deve ser (60, 60, 100, 100)
    region = red_layer.canvas_region
    assert region.top_left == (60, 60)
    assert region.size == (100, 100)

def test_render_transform_prioridade_sobre_estado(red_layer):
    # Define um estado estático (Rotação 90)
    red_layer.rotation = 90
    
    # Ativa o Transform, mas deixa ele como Identidade (sem mexer em nada)
    # Por padrão, se acessarmos a property, ele cria a Identidade.
    _ = red_layer.transform 
    
    # Como transform_used agora é True, a Rotação 90 estática deve ser IGNORADA
    # O BBox deve ser o original (0, 0, 100, 100) e não o rotacionado.
    region = red_layer.canvas_region
    assert region.top_left == (0, 0)
    assert region.size == (100, 100)

def test_render_transform_chaining_complexo(red_layer):
    # Testa: Translação(50,50) -> Escala(2x no Centro) -> Rotação(90 no Centro)
    # Tudo via encadeamento de métodos
    red_layer.transform.translate(50, 50).scale(2, 2, 0.5, 0.5).rotation(90, 0.5, 0.5)
    
    # Verificação geométrica do BBox:
    # 1. Base 100x100
    # 2. Translate(50,50) -> (50, 50, 100, 100)
    # 3. Scale 2x no centro original (50,50) -> O ponto (50,50) eh o pivo, nao muda.
    #    Os outros cantos expandem. BBox local vira (-50, 50, 200, 200).
    # 4. Rotação 90 no centro original (50,50) -> BBox rotaciona em torno de (50,50).
    #    O ponto (-50, 50) rotacionado 90 deg em torno de (50,50) vai para (-150, 50).
    #    (Vetor pivo->pt = (-100, 0). Rot 90 = (0, -100). Volta pivo = (50, -50)? Nao, pera.)
    #    Calculado via script: Top-Left eh (-150, 50)
    
    region = red_layer.canvas_region
    assert region.top_left == (-150, 50)
    assert region.size == (200, 200)

def test_render_transform_clear_restaura_estado(red_layer):
    # 1. Define estado: Rotação 90
    red_layer.rotation = 90
    orig_region = red_layer.canvas_region
    
    # 2. Ativa transform com Translação
    red_layer.transform.translate(100, 100)
    assert red_layer.canvas_region.top_left == (100, 100)
    
    # 3. Limpa o transform
    red_layer.transform_clear()
    
    # 4. Deve voltar a usar a Rotação 90 do estado estático
    assert red_layer.canvas_region == orig_region
    assert not red_layer.transform_used

def test_render_pixel_accuracy_com_transform(red_layer):
    # Posição (0,0), mas gira 90 graus no centro via Transform
    red_layer.transform.rotation(90, 0.5, 0.5)
    
    render = make_render()
    img = render.render(red_layer)
    
    # Se girou 90 no centro um quadrado sólido, ele continua ocupando o mesmo espaço 
    # (0,0,100,100) e deve ser todo vermelho.
    data = img[...]
    assert data.shape == (100, 100, 4)
    # Verifica o centro
    np.testing.assert_array_equal(data[50, 50], [255, 0, 0, 255])
    # Verifica um ponto interno (1,1) para evitar a ambiguidade matemática da quina 
    # exata em rotações (que pode mapear para a borda externa 100.0).
    np.testing.assert_array_equal(data[1, 1], [255, 0, 0, 255])
