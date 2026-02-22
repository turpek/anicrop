from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.render import LayerRender
from anicrop.spatial import Region, Span
from anicrop.transform import mat_final
import numpy as np
import pytest


# Fixture da classe que vamos testar
@pytest.fixture
def layer_render():
    # Tenta importar a classe que ainda não existe
    return LayerRender()


# Funções auxiliares para gerar Layers e Edits
def make_img(w=100, h=100, color=(255, 0, 0, 255)):
    # Gera uma imagem com uma cor sólida
    img_data = np.zeros((h, w, 4), dtype=np.uint8)
    img_data[:] = color
    # Converte para o objeto Image do projeto
    # Ajuste: ImageFormat.RGBA assume 4 canais
    return Image(img_data, ImageFormat.RGBA)


def make_layer(w=100, h=100, x=0, y=0, color=(255, 0, 0, 255)):
    img = make_img(w, h, color)
    layer = Layer(img)
    # Assumindo que setters de x/y existem e funcionam
    if x != 0:
        layer.x = x
    if y != 0:
        layer.y = y
    return layer


def test_LayerRender_identidade_sem_transformacao(layer_render):
    """
    Testa se um Layer sem transformações (Escala=1, Rotação=0, Pos=0,0)
    é renderizado exatamente igual à imagem original, pixel por pixel.
    """
    width, height = 50, 50
    # Cor arbitrária para garantir que pixels batem
    original_layer = make_layer(w=width, h=height, color=(100, 150, 200, 255))

    # Renderiza o layer
    rendered_image = layer_render.render(original_layer)

    # Verificações básicas
    assert rendered_image.width == width
    assert rendered_image.height == height

    # Verifica se os pixels são idênticos
    # A propriedade .array ou acesso direto [...] deve retornar o ndarray
    np.testing.assert_array_equal(rendered_image[...], original_layer.image[...])


def test_LayerRender_rotacao_expansao_segura(layer_render):
    """
    Testa se o LayerRender expande a imagem corretamente ao rotacionar
    em 45 graus, garantindo que as pontas não sejam cortadas pelo OpenCV
    e que o fundo adicionado seja totalmente transparente.
    """
    width, height = 100, 100
    cor_original = (255, 0, 0, 255)
    layer = make_layer(w=width, h=height, color=cor_original)

    # Aplica a rotação de 45 graus
    # Nota: Ajuste a atribuição abaixo conforme a API exata da sua classe Rotation
    layer.rotation.angle = 45

    # Ação: Renderiza o layer
    rendered_image = layer_render.render(layer)
    bbox = layer.canvas_region

    # 1. A imagem final TEM que ter o tamanho exato do Bounding Box calculado
    assert rendered_image.width == bbox.width
    assert rendered_image.height == bbox.height

    # 3. Verificação de pixels (O truque do Canal Alpha)
    img_array = rendered_image[...]

    # O pixel bem no centro da imagem expandida tem que fazer parte do quadrado vermelho
    centro_x, centro_y = bbox.width // 2, bbox.height // 2
    pixel_central = img_array[centro_y, centro_x]

    assert pixel_central[3] == 255
    np.testing.assert_array_equal(pixel_central, cor_original)

    # O pixel do canto extremo (0,0) agora é fundo vazio, tem que ser transparente
    pixel_canto = img_array[0, 0]
    assert pixel_canto[3] == 0


def test_LayerRender_achatar_edicoes_e_transformar(layer_render):
    """
    Testa se o renderizador consegue fazer o merge das edições (EditLayer)
    sobre a imagem base ANTES de aplicar a transformação espacial do OpenCV.
    """
    # 1. Setup do Base (100x100 totalmente Azul)
    cor_azul = (0, 0, 255, 255)
    layer = make_layer(w=100, h=100, color=cor_azul)

    # 2. Setup da Edição (20x20 totalmente Vermelha)
    cor_vermelha = (255, 0, 0, 255)
    img_edicao = make_img(w=20, h=20, color=cor_vermelha)

    # 3. Adiciona a edição no canto superior esquerdo (X: 10 a 30, Y: 10 a 30)
    regiao_edicao = Region(Span(10, 20), Span(10, 20))
    # Assumindo BlendMode.NORMAL ou um equivalente que você definiu no projeto
    layer.add_edit(img_edicao, regiao_edicao)

    # --- AÇÃO 1: Renderização Pura (Sem Rotação) ---
    img_renderizada = layer_render.render(layer)
    array_final = img_renderizada[...]

    # O canto extremo (0,0) deve ser azul (é o fundo)
    np.testing.assert_array_equal(array_final[0, 0], cor_azul)

    # O meio da área onde colamos a edição (Y=20, X=20) TEM que ser vermelho
    np.testing.assert_array_equal(array_final[20, 20], cor_vermelha)

    # --- AÇÃO 2: Renderização com Rotação (90 Graus) ---
    layer.rotation.angle = 90
    img_rotacionada = layer_render.render(layer)
    array_rotacionado = img_rotacionada[...]

    # Se a imagem girou, o ponto que antes era vermelho (20,20) agora
    # se moveu para outra posição, então esse pixel original não pode mais ser vermelho.
    assert not np.array_equal(array_rotacionado[20, 20], cor_vermelha), "A edição não girou junto com o layer base!"

    # O centro exato da imagem base (50,50) deve continuar azul, pois
    # nós colamos a figurinha vermelha no canto, e não no meio.
    np.testing.assert_array_equal(array_rotacionado[50, 50], cor_azul)


import numpy as np
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.spatial import Region, Span
from anicrop.render import LayerRender
from anicrop.transform import mat_final

def test_render_fluxo_real_com_quina(layer_render: LayerRender):
    # 1. Setup do Layer 100x100 (Azul)
    bg_data = np.zeros((100, 100, 4), dtype=np.uint8)
    bg_data[:] = [0, 0, 255, 255] # Azul sólido
    bg_image = Image(bg_data, ImageFormat.RGBA)
    layer = Layer(bg_image, Region(Span(0, 100), Span(0, 100)))

    # Gira 90 graus: O ponto local (0,0) agora está no global (100, 0)
    layer.rotation.angle = 90

    # 2. Setup do Edit 10x10 (Quinas coloridas)
    edit_data = np.zeros((10, 10, 4), dtype=np.uint8)
    # RGBA: Vermelho, Verde, Azul, Branco
    C_TL, C_TR, C_BL, C_BR = (255,0,0,255), (0,255,0,255), (0,0,255,255), (255,255,255,255)
    edit_data[0, 0] = C_TL
    edit_data[0, 9] = C_TR
    edit_data[9, 0] = C_BL
    edit_data[9, 9] = C_BR
    edit_img = Image(edit_data, ImageFormat.RGBA)

    # 3. O Clique do Usuário (Coordenadas Globais/Canvas)
    # Clicamos em (90, 0) para a figurinha de 10x10 ficar no "canto"
    # superior (espaço global) mas dentro da imagem.
    clique_region = Region(Span(90, 10), Span(0, 10))

    # 4. AÇÃO: O seu motor calcula a inversa internamente
    layer.add_edit(edit_img, clique_region)

    # 5. Stress: Gira mais -45 deg e escala 2x
    layer.rotation.angle += -45
    layer.scale = (2.0, 2.0)

    # 6. Renderização
    breakpoint()
    result_image = layer_render.render(layer)
    data = result_image[...]

    # 7. Verificação via Matriz Final
    # Calculamos onde a quina (0,0) do edit DEVERIA estar no canvas final
    m_final = mat_final(layer, *layer.canvas_region.top_left)
    edit_obj = layer._edits[0]
    full_edit_matrix = m_final @ edit_obj.local_matrix

    # Projetamos o ponto local (0,0) do Edit para o Canvas
    p = full_edit_matrix @ np.array([0, 0, 1.0])
    cx, cy = int(round(p[0])), int(round(p[1]))

    # Validação da cor
    np.testing.assert_array_equal(
        data[cy, cx], C_TL,
        err_msg=f"A cor na posição {cx},{cy} deveria ser Vermelha (TL)"
    )
