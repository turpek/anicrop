import numpy as np

from anicrop.container import GroupLayer
from anicrop.enums import ImageFormat
from anicrop.frame import ViewportFrame
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.render import SceneTraverser, ViewportRender
from anicrop.transform import mat_global
from anicrop.viewport import Viewport


def test_group_layer_integration_transforms():
    # 1. Setup Layer e Image (100x100)
    img = Image.new((100, 100), ImageFormat.RGBA)

    # Pintando um quadrado 10x10 (branco) no topo esquerdo
    # Note que img._data tem shape (100, 100, 4), onde Y, X, C
    img._data[0:10, 0:10] = [255, 255, 255, 255]
    layer = Layer(img)

    # 2. Setup dos Grupos
    parent = GroupLayer()
    root = GroupLayer()

    parent.append(layer)
    root.append(parent)

    # 3. Aplicar transformações
    # Transladar o parent 50 pixels pra direita e 50 pra baixo
    parent.transform.translate(50, 50)

    # Rotacionar a root em 180 graus no seu proprio centro (0.5, 0.5)
    root.transform.rotate(180, 0.5, 0.5)

    # 4. Renderizar tudo na Viewport
    viewport = Viewport((200, 200))
    renderer = ViewportRender()

    # Executa o render recursivo do grupo raiz via SceneTraverser
    traverser = SceneTraverser(renderer, viewport, ViewportFrame)
    images_gp = traverser.traverse(root)

    # Extrair a imagem renderizada do grupo raiz
    assert len(images_gp) == 1
    root_layer, rendered_image, root_frame = images_gp[0]

    # 5. Verificar a imagem resultante
    # Procura onde o alpha (canal 3) é 255
    coords = np.argwhere(rendered_image._data[:, :, 3] == 255)

    assert len(coords) > 0, "Nenhum pixel visível renderizado"

    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)

    # O pixel (0,0) (Top-Left original) virou a base direita apos rotacao 180
    # O centro de root era (100, 100).
    # O ponto (50, 50) global (origem transladada) -> rotacionado em 180 no (100, 100) = (150, 150)
    # Como as dimensoes originais eram 0 a 9 (10 pixels), a bounding box invertida ficara em torno de 141 a 150.

    # Fazemos a checagem com uma margem de tolerancia (por causa da interpolacao Lanczos4)
    assert 90 <= min_x <= 92
    assert 90 <= min_y <= 92
    assert 98 <= max_x <= 100
    assert 98 <= max_y <= 100


def test_layer_estresse_hierarquia_4_niveis_bisavo_avo_pai():
    img = Image.new((100, 100), ImageFormat.RGBA)
    layer = Layer(img)
    layer.transform.translate(10, 20)

    pai = GroupLayer()
    avo = GroupLayer()
    bisavo = GroupLayer()

    # Monta a árvore de 4 níveis: Bisavô -> Avô -> Pai -> Layer
    bisavo.append(avo)
    avo.append(pai)
    pai.append(layer)

    # 1. Translada o Bisavô em (100, 200)
    bisavo.transform.translate(100, 200)

    # 2. Rotaciona o Avô em 90° (pivô em 50, 50)
    avo.transform.rotate(90, 0.5, 0.5)

    # 3. Translada o Pai em (30, 40)
    pai.transform.translate(30, 40)

    pt_origem = np.array([0, 0, 1], dtype=np.float32)

    # O mat_global(layer) acumula recursivamente toda a árvore de 4 níveis:
    # Layer (10, 20) + Pai (30, 40) = (40, 60)
    # Avô Rotação 90° em (50, 50) leva (40, 60) para (40, 40)
    # Bisavô Translação (100, 200) leva (40, 40) para (140, 240)
    np.testing.assert_allclose(mat_global(layer) @ pt_origem, [140, 240, 1], atol=1e-4)

    # 4. Remove o Avô do Bisavô (desvincula o Bisavô e sua translação 100, 200)
    bisavo.remove(avo)

    # A matriz global do layer desfaz o efeito da translação do Bisavô, caindo de (140, 240) para (40, 40)
    np.testing.assert_allclose(mat_global(layer) @ pt_origem, [40, 40, 1], atol=1e-4)
