from unittest.mock import patch, MagicMock
import uuid
import pytest
import numpy as np
from anicrop.enums import ImageFormat
from anicrop.canvas import Canvas
from anicrop.container import GroupLayer, LayerStack
from anicrop.image import Image
from anicrop.layer import EditLayer, Layer
from anicrop.layout import Layout
from unittest.mock import MagicMock
from anicrop.spatial import Region


def make_layer(x=25, y=25, w=50, h=50) -> Layer:
    mock_img = MagicMock(spec=Image)
    mock_img.size = (w, h)
    layer = Layer(mock_img)
    layer._region = Region.from_rect(x, y, w, h)
    return layer


def make_edit(x=0, y=0, w=100, h=100):
    """Cria um mock de EditLayer com a property .region vinculada a ._region (Getter e Setter)."""
    edit = MagicMock(spec=EditLayer)
    edit._region = Region.from_rect(x, y, w, h)

    # Vincula dinamicamente a property .region (Getter e Setter)
    type(edit).region = property(
        lambda self: self._region,
        lambda self, val: setattr(self, '_region', val)
    )

    return edit


@pytest.mark.parametrize(
    'crop_ref,crop_expect',
    [
        pytest.param(
            (25, 25, 50, 50),
            Region.from_rect(25, 25, 50, 50),
            id='tuple',
        ),
        pytest.param(
            Region.from_rect(25, 25, 50, 50),
            Region.from_rect(25, 25, 50, 50),
            id='region',
        ),
        pytest.param(
            make_layer(25, 25, 50, 50),
            Region.from_rect(25, 25, 50, 50),
            id='layer',
        ),
        pytest.param(
            Canvas(width=50, height=50),
            Region.from_rect(0, 0, 50, 50),
            id='canvas',
        ),
    ]
)
def test_layout_crop_layer_com_variadas_referencias(mocker, crop_ref, crop_expect):
    """Testa crop de um Layer passando um objeto Region como referência."""
    layer_mock = make_layer(0, 0, 100, 100)
    layout = Layout()
    layout.crop(layer_mock, crop_ref)
    assert layer_mock.region == crop_expect


def test_layout_crop_layer_com_edits(mocker):
    """Testa crop de um Layer passando um objeto Region como referência."""
    crop_ref = Region.from_rect(25, 25, 50, 50)
    edit_mock = make_edit(50, 50, 50, 50)
    layer_mock = make_layer(0, 0, 100, 100)
    layer_mock._edits = [edit_mock]
    layout = Layout()
    layout.crop(layer_mock, crop_ref)
    assert edit_mock.region.top_left == (25, 25)


def test_layout_crop_sem_overlap_retorna_false():
    """Garante que fazer crop de um objeto quando não há sobreposição com a referência retorna False."""
    layer_mock = make_layer(0, 0, 50, 50)
    far_ref = Region.from_rect(200, 200, 50, 50)  # Sem sobreposição com (0, 0, 50, 50)

    layout = Layout()
    result = layout.crop(layer_mock, far_ref)
    assert result is False


def test_layout_crop_group_vazio_retorna_false():
    """Garante que fazer crop em um GroupLayer vazio é ignorado e retorna False."""
    group = GroupLayer(name="EmptyGroup")

    # Um grupo vazio não possui filhos (len(group) == 0)
    layout = Layout()
    ref = Region.from_rect(0, 0, 100, 100)

    # Deve retornar False já que é um No-Op
    result = layout.crop(group, ref)
    assert result is False


def test_layout_crop_group_com_filhos_diretos():
    """Valida a propagação do crop para os filhos (Layers) de um grupo."""
    group = GroupLayer(name="Root")
    layer1 = make_layer(0, 0, 50, 50)
    layer2 = make_layer(50, 50, 50, 50)
    group.append(layer1)
    group.append(layer2)

    layout = Layout()
    layout.crop(group, Region.from_rect(25, 25, 50, 50))

    assert layer1.region == Region.from_rect(25, 25, 25, 25)
    assert layer2.region == Region.from_rect(50, 50, 25, 25)


def test_layout_crop_recursao_em_arvore_profunda():
    """Garante que o crop navega recursivamente em Grupos aninhados."""
    root = GroupLayer(name="Root")
    sub_group = GroupLayer(name="SubGroup")
    leaf_layer = make_layer(0, 0, 100, 100)

    sub_group.append(leaf_layer)
    root.append(sub_group)

    layout = Layout()
    layout.crop(root, Region.from_rect(50, 50, 50, 50))

    # A folha que estava dentro do subgrupo dentro da raiz tem que ser cortada!
    assert leaf_layer.region == Region.from_rect(50, 50, 50, 50)


def test_layout_crop_raiz_mista_com_subgrupo_vazio():
    """Garante que o crop funciona quando a raiz tem um Layer na posição 0 e um subgrupo vazio."""
    root = GroupLayer(name="Root")
    leaf_layer = make_layer(0, 0, 100, 100)
    empty_sub_group = GroupLayer(name="SubGroup_Vazio")

    # Inserindo o Layer primeiro (posição 0)
    root.append(leaf_layer)
    # Inserindo o subgrupo vazio na sequência
    root.append(empty_sub_group)

    layout = Layout()
    result = layout.crop(root, Region.from_rect(50, 50, 50, 50))

    # A folha tem que ser cortada normalmente
    assert leaf_layer.region == Region.from_rect(50, 50, 50, 50)

    # Como uma camada foi processada e cortada com sucesso, a operação inteira retorna True
    assert result is True


def test_layout_crop_raiz_com_subgrupo_vazio_retorna_false():
    """Garante que se a raiz possuir apenas subgrupos vazios, o retorno bolhado será False."""
    root = GroupLayer(name="Root")
    empty_sub_group = GroupLayer(name="SubGroup_Vazio")

    # Outro subgrupo dentro do subgrupo, para aprofundar a recursão
    deep_empty_sub_group = GroupLayer(name="Deep_Vazio")
    empty_sub_group.append(deep_empty_sub_group)

    # A raiz só possui o subgrupo vazio
    root.append(empty_sub_group)

    layout = Layout()
    result = layout.crop(root, Region.from_rect(50, 50, 50, 50))

    # Nenhuma camada foi de fato cortada (no-op), então o resultado agregado deve ser False
    assert result is False


def test_layout_crop_raiz_com_subgrupo_valido_retorna_true():
    """Garante que a raiz propaga (agrega) o resultado True se um subgrupo interno cortou uma camada com sucesso."""
    root = GroupLayer(name="Root")
    sub_group = GroupLayer(name="SubGroup")
    leaf_layer = make_layer(0, 0, 100, 100)

    # O subgrupo tem uma camada
    sub_group.append(leaf_layer)

    # A raiz só tem o subgrupo
    root.append(sub_group)

    layout = Layout()
    result = layout.crop(root, Region.from_rect(50, 50, 50, 50))

    # O Layer foi cortado com sucesso (processado pelo sub_group)
    assert leaf_layer.region == Region.from_rect(50, 50, 50, 50)

    # Como houve um crop na árvore, a operação na raiz DEVE retornar True.
    # Na sua implementação sem o `if`, a raiz não vê o `True` do subgrupo e acaba retornando `False`.
    assert result is True


def make_group_com_1_layer(x=0, y=0, w=100, h=100):
    group = GroupLayer()
    layer = make_layer(x, y, w, h)
    group.append(layer)
    return group


@pytest.mark.parametrize('method_name', ['crop', 'fit'])
@pytest.mark.parametrize(
    'target_factory',
    [
        pytest.param(lambda: make_layer(0, 0, 100, 100), id='layer'),
        pytest.param(lambda: Canvas(100, 100), id='canvas'),
        pytest.param(lambda: make_group_com_1_layer(0, 0, 100, 100), id='grouplayer'),
    ]
)
def test_layout_operacao_mesma_regiao_noop(method_name, target_factory):
    """Garante que se a referência for idêntica à região atual do alvo, fit e crop atuam como no-op."""
    target = target_factory()
    layout = Layout()

    # Identificamos a 'old_region' dependendo se é container ou layer folha
    if isinstance(target, GroupLayer):
        old_region = target._children[0].region
    else:
        old_region = target.region

    # A referência é exatamente a mesma região
    metodo = getattr(layout, method_name)
    result = metodo(target, old_region)

    # Validamos que o objeto não sofreu realocação (mesma instância) e retornou False
    if isinstance(target, GroupLayer):
        assert target._children[0].region is old_region
    else:
        assert target.region is old_region
    assert result is False


@pytest.mark.parametrize('method_name', ['crop', 'fit'])
def test_layout_operacao_mista_em_grouplayer(method_name):
    """
    Testa fit e crop em uma hierarquia mista.
    A camada na raiz tem o mesmo tamanho da ref (No-Op), mas a do subgrupo tem tamanho diferente.
    Garante que o método principal reporte True devido à alteração no subgrupo,
    e preserve a instância da camada No-Op.
    """
    root = GroupLayer(name="Root")
    sub_group = GroupLayer(name="SubGroup")

    # Camada com tamanho idêntico à referência (No-Op esperado)
    layer_igual = make_layer(0, 0, 100, 100)
    region_igual_old = layer_igual.region

    # Camada que cruza a fronteira da referência (Modificação garantida tanto no crop quanto fit)
    layer_diferente = make_layer(50, 50, 100, 100)

    root.append(layer_igual)
    sub_group.append(layer_diferente)
    root.append(sub_group)

    ref_region = Region.from_rect(0, 0, 100, 100)

    layout = Layout()
    metodo = getattr(layout, method_name)

    result = metodo(root, ref_region)

    # Como o layer_diferente sofreu alteração real, o root deve propagar True
    assert result is True

    # A camada que já era idêntica não deve sofrer realocação (No-Op)
    assert layer_igual.region is region_igual_old

    if method_name == "fit":
        # Fit expande/adota a região inteira
        assert layer_diferente.region == ref_region
    else:
        # Crop pega a interseção: (50,50,100,100) & (0,0,100,100) = (50,50,50,50)
        assert layer_diferente.region == Region.from_rect(50, 50, 50, 50)


def test_layout_fit_layer_com_overlap():
    """Testa fit de um Layer forçando sua região a ser idêntica à referência, ajustando as intenções internas."""
    layer = make_layer(20, 20, 50, 50)
    edit_mock = make_edit(20, 20, 50, 50)
    layer._edits = [edit_mock]

    # A referência começa no (0, 0)
    ref_region = Region.from_rect(0, 0, 100, 100)

    layout = Layout()
    result = layout.fit(layer, ref_region)

    # A região da camada agora deve ser estritamente igual à da referência
    assert layer.region == ref_region

    # Compensação (Offset):
    # Old (20, 20) -> New (0, 0). Distância = (20, 20)
    # A posição original do edit (20, 20) deslocada por (20, 20) deve virar (40, 40)
    assert edit_mock.region.top_left == (40, 40)
    assert result is True


def test_layout_fit_sem_overlap_retorna_false():
    """Garante que fazer fit quando não há sobreposição com a referência retorna False."""
    layer = make_layer(0, 0, 50, 50)
    far_ref = Region.from_rect(200, 200, 50, 50)

    layout = Layout()
    result = layout.fit(layer, far_ref)
    assert result is False


def test_layout_fit_group_vazio_retorna_false():
    """Garante que fazer fit de um GroupLayer vazio retorna False."""
    group = GroupLayer(name="Root")
    layout = Layout()
    ref_region = Region.from_rect(0, 0, 100, 100)
    result = layout.fit(group, ref_region)
    assert result is False


def test_layout_fit_group_com_filhos_diretos():
    """Valida a propagação do fit para os filhos (Layers) de um grupo."""
    group = GroupLayer(name="Root")
    layer1 = make_layer(0, 0, 50, 50)
    layer2 = make_layer(50, 50, 50, 50)
    group.append(layer1)
    group.append(layer2)

    ref_region = Region.from_rect(0, 0, 100, 100)
    layout = Layout()
    result = layout.fit(group, ref_region)

    assert result is True
    # Ambos os filhos devem ter absorvido a nova região
    assert layer1.region == ref_region
    assert layer2.region == ref_region


def test_layout_fit_recursao_em_arvore_profunda():
    """Garante que o fit navega recursivamente em Grupos aninhados."""
    root = GroupLayer(name="Root")
    sub_group = GroupLayer(name="SubGroup")
    leaf_layer = make_layer(0, 0, 50, 50)

    sub_group.append(leaf_layer)
    root.append(sub_group)

    # Vamos dar fit com uma referência enorme para cobrir o overlap
    ref_region = Region.from_rect(0, 0, 200, 200)
    layout = Layout()
    result = layout.fit(root, ref_region)

    assert result is True
    assert leaf_layer.region == ref_region


@pytest.mark.parametrize(
    'fit_ref,fit_expect',
    [
        pytest.param(
            (0, 0, 80, 80),
            Region.from_rect(0, 0, 80, 80),
            id='tuple',
        ),
        pytest.param(
            Region.from_rect(0, 0, 80, 80),
            Region.from_rect(0, 0, 80, 80),
            id='region',
        ),
        pytest.param(
            make_layer(0, 0, 80, 80),
            Region.from_rect(0, 0, 80, 80),
            id='layer',
        ),
        pytest.param(
            Canvas(width=80, height=80),
            Region.from_rect(0, 0, 80, 80),
            id='canvas',
        ),
    ]
)
def test_layout_fit_canvas_com_variadas_referencias(mocker, fit_ref, fit_expect):
    """Testa fit de um Canvas passando vários tipos de objetos como referência."""
    canvas = Canvas(100, 100)
    layout = Layout()
    layout.fit(canvas, fit_ref)
    assert canvas.region == fit_expect


@pytest.mark.parametrize(
    'fit_ref,fit_expect',
    [
        pytest.param(
            (0, 0, 80, 80),
            Region.from_rect(0, 0, 80, 80),
            id='tuple',
        ),
        pytest.param(
            Region.from_rect(0, 0, 80, 80),
            Region.from_rect(0, 0, 80, 80),
            id='region',
        ),
        pytest.param(
            make_layer(0, 0, 80, 80),
            Region.from_rect(0, 0, 80, 80),
            id='layer',
        ),
        pytest.param(
            Canvas(width=80, height=80),
            Region.from_rect(0, 0, 80, 80),
            id='canvas',
        ),
    ]
)
def test_layout_fit_layer_com_variadas_referencias(mocker, fit_ref, fit_expect):
    """Testa fit de um Layer passando vários tipos de objetos como referência."""
    layer = make_layer(20, 20, 50, 50)
    edit_mock = make_edit(20, 20, 50, 50)
    layer._edits = [edit_mock]

    layout = Layout()
    result = layout.fit(layer, fit_ref)

    assert result is True
    assert layer.region == fit_expect

    # Offset dinâmico baseado na diferença entre a origem antiga (20, 20) e a nova (fit_expect)
    offset_x = 20 - fit_expect.top_left[0]
    offset_y = 20 - fit_expect.top_left[1]

    assert edit_mock.region.top_left == (20 + offset_x, 20 + offset_y)


@pytest.mark.parametrize(
    'crop_ref,crop_expect',
    [
        pytest.param(
            (25, 25, 50, 50),
            Region.from_rect(25, 25, 50, 50),
            id='tuple',
        ),
        pytest.param(
            Region.from_rect(25, 25, 50, 50),
            Region.from_rect(25, 25, 50, 50),
            id='region',
        ),
        pytest.param(
            make_layer(25, 25, 50, 50),
            Region.from_rect(25, 25, 50, 50),
            id='layer',
        ),
        pytest.param(
            Canvas(width=50, height=50),
            Region.from_rect(0, 0, 50, 50),
            id='canvas',
        ),
    ]
)
def test_layout_crop_canvas_com_variadas_referencias(mocker, crop_ref, crop_expect):
    """Testa crop de um Layer passando um objeto Region como referência."""
    canvas = Canvas(100, 100)
    layout = Layout()
    layout.crop(canvas, crop_ref)
    assert canvas.region == crop_expect


# ############################# Testes para Layout.align #####################################

@pytest.mark.parametrize('factor, expect_offset', [
    (0.0, -50),  # start=0 -> offset = 0 - 50 = -50
    (0.5, 30),   # start=80 -> offset = 80 - 50 = 30
    (1.0, 110),  # start=160 -> offset = 160 - 50 = 110
])
@pytest.mark.parametrize(
    'align_ref',
    [
        pytest.param((0, 0, 200, 200), id='tuple'),
        pytest.param(Region.from_rect(0, 0, 200, 200), id='region'),
        pytest.param(make_layer(0, 0, 200, 200), id='layer'),
        pytest.param(Canvas(width=200, height=200), id='canvas'),
    ]
)
def test_layout_align_layer_com_variadas_referencias_e_fatores(align_ref, factor, expect_offset):
    """Testa o alinhamento de um Layer usando vários tipos de referência e fatores."""
    layer = make_layer(50, 50, 40, 40)
    edit_mock = make_edit(50, 50, 40, 40)
    layer._edits = [edit_mock]

    layout = Layout()
    # Usando o mesmo factor para X e Y para simplificar o teste
    result = layout.align(layer, align_ref, factor, factor)

    assert result is True

    # A largura e altura do layer não podem mudar (40x40). Somente o topo-esquerdo muda.
    expect_start = 50 + expect_offset
    assert layer.region == Region.from_rect(expect_start, expect_start, 40, 40)

    # Como o align move a Layer como um todo pelo canvas,
    # as edições internas permanecem ancoradas na sua posição relativa intacta.
    assert edit_mock.region.top_left == (50, 50)


def test_layout_align_group_vazio_retorna_false():
    """Garante que alinhar um GroupLayer vazio retorna False."""
    group = GroupLayer(name="Root")
    layout = Layout()
    ref_region = Region.from_rect(0, 0, 100, 100)
    result = layout.align(group, ref_region)
    assert result is False


def test_layout_align_group_com_filhos_diretos():
    """Valida a propagação do align para os filhos (Layers) de um grupo."""
    group = GroupLayer(name="Root")
    # layer1 e layer2 começam fora da origem com tamanho 40x40.
    layer1 = make_layer(50, 50, 40, 40)
    layer2 = make_layer(60, 60, 40, 40)
    group.append(layer1)
    group.append(layer2)

    # Referência: (0, 0, 200, 200).
    # Sobra (slack) para cada camada de 40px: 200 - 40 = 160
    # Alinhamento 1.0 (Direita/Base) = novo start 160
    ref_region = Region.from_rect(0, 0, 200, 200)
    layout = Layout()

    result = layout.align(group, ref_region, 1.0, 1.0)

    assert result is True
    assert layer1.region == Region.from_rect(160, 160, 40, 40)
    assert layer2.region == Region.from_rect(160, 160, 40, 40)


def test_layout_align_recursao_em_arvore_profunda():
    """Garante que o align navega recursivamente em Grupos aninhados."""
    root = GroupLayer(name="Root")
    sub_group = GroupLayer(name="SubGroup")
    leaf_layer = make_layer(50, 50, 40, 40)

    sub_group.append(leaf_layer)
    root.append(sub_group)

    ref_region = Region.from_rect(0, 0, 200, 200)
    layout = Layout()

    # Alinhando ao centro (0.5)
    # Folga = 200 - 40 = 160. Offset 0.5 = 80. Novo start = 80.
    result = layout.align(root, ref_region, 0.5, 0.5)

    assert result is True
    assert leaf_layer.region == Region.from_rect(80, 80, 40, 40)


@pytest.mark.parametrize('factor, expect_offset', [
    (0.0, 0),    # start=0 -> offset=0 (No-Op)
    (0.5, 50),   # start=50 -> offset=50
    (1.0, 100),  # start=100 -> offset=100
])
@pytest.mark.parametrize(
    'align_ref',
    [
        pytest.param((0, 0, 200, 200), id='tuple'),
        pytest.param(Region.from_rect(0, 0, 200, 200), id='region'),
        pytest.param(make_layer(0, 0, 200, 200), id='layer'),
        pytest.param(Canvas(width=200, height=200), id='canvas'),
    ]
)
def test_layout_align_canvas_com_variadas_referencias_e_fatores(align_ref, factor, expect_offset):
    """Testa o alinhamento de um Canvas usando vários tipos de referência e fatores."""
    canvas = Canvas(100, 100)
    layout = Layout()

    # O Canvas começa naturalmente em (0,0) com tamanho 100x100.
    result = layout.align(canvas, align_ref, factor, factor)

    # Se o fator for 0.0, o start calculado será 0.
    # Como o Canvas já estava na origem 0,0 (mesma Region), será um No-Op (retorna False)
    expected_result = (expect_offset != 0)
    assert result is expected_result

    # O Canvas deverá se manter 100x100, apenas tendo seu start deslizado
    assert canvas.region == Region.from_rect(expect_offset, expect_offset, 100, 100)


@pytest.mark.parametrize('factor, target_rect', [
    (0.0, (0, 0, 100, 100)),     # Já posicionado na esquerda/topo
    (0.5, (50, 50, 100, 100)),   # Já posicionado no centro
    (1.0, (100, 100, 100, 100)),  # Já posicionado na direita/base
])
@pytest.mark.parametrize(
    'target_builder',
    [
        pytest.param(lambda rect: make_layer(*rect), id='layer'),
        pytest.param(lambda rect: make_group_com_1_layer(*rect), id='grouplayer'),
    ]
)
def test_layout_align_noop_quando_ja_alinhado(target_builder, factor, target_rect):
    """Garante que se o alvo já estiver ancorado na posição correta, a operação aborta (No-Op)."""
    target = target_builder(target_rect)

    if isinstance(target, GroupLayer):
        old_region = target._children[0].region
    else:
        old_region = target.region

    ref_region = Region.from_rect(0, 0, 200, 200)
    layout = Layout()

    result = layout.align(target, ref_region, factor, factor)

    assert result is False
    # Verifica que a mesmíssima instância da Region original foi preservada
    if isinstance(target, GroupLayer):
        assert target._children[0].region is old_region
    else:
        assert target.region is old_region


def test_layout_align_noop_tamanhos_identicos():
    """Garante que se alvo e referência são idênticos em geometria, qualquer fator é No-Op."""
    # Usando o Canvas aqui, pois ele é travado na origem 0,0, tornando o cenário perfeito
    canvas = Canvas(200, 200)
    old_region = canvas.region
    ref_region = Region.from_rect(0, 0, 200, 200)
    layout = Layout()

    # Passando fatores matematicamente "loucos".
    # Como o slack é zero, eles são ignorados e não geram movimento.
    result = layout.align(canvas, ref_region, 99.0, -50.0)

    assert result is False
    assert canvas.region is old_region


# ############################# Testes para Layout.resize_bounds #####################################

@pytest.mark.parametrize('anchor, expect_start', [
    (0.0, 50),   # Âncora no topo-esquerdo (Layer original começa no 50, o novo start será 50)
    (0.5, 20),   # Âncora no centro (Centro original era 70. 70 - 100/2 = 20)
    (1.0, -10),  # Âncora no canto direito (Fim original era 90. 90 - 100 = -10)
])
def test_layout_resize_bounds_mantem_conteudo_estatico(anchor, expect_start):
    """Garante que expandir a caixa (layer/canvas) não altera a posição global do conteúdo."""
    # Camada original: start=50, width=40. O fim é 90. O centro é 70.
    layer = make_layer(50, 50, 40, 40)

    # Nosso conteúdo (EditLayer) fictício, na posição interna (10, 10) relativa ao Layer
    edit_mock = make_edit(10, 10, 40, 40)
    layer._edits = [edit_mock]

    # A posição global do conteúdo ANTES do resize é: Layer(50) + Edit(10) = 60
    global_top_left_antes_x = layer.region.top_left[0] + edit_mock.region.top_left[0]
    global_top_left_antes_y = layer.region.top_left[1] + edit_mock.region.top_left[1]
    assert global_top_left_antes_x == 60
    assert global_top_left_antes_y == 60

    layout = Layout()

    # Vamos expandir a camada para 100x100 usando a âncora testada (mesma pra X e Y)
    # Como você ainda vai criar, isso aqui deve quebrar até implementar! (TDD Clássico)
    result = layout.resize_bounds(layer, 100, 100, anchor, anchor)

    assert result is True

    # 1. Valida se a borda da caixa realmente cresceu e foi ancorada no lugar certo
    assert layer.region == Region.from_rect(expect_start, expect_start, 100, 100)

    # 2. PROVA CABAL: Valida se a posição global do conteúdo continuou a mesma! (Não saiu voando)
    global_top_left_depois_x = layer.region.top_left[0] + edit_mock.region.top_left[0]
    global_top_left_depois_y = layer.region.top_left[1] + edit_mock.region.top_left[1]

    assert global_top_left_depois_x == global_top_left_antes_x
    assert global_top_left_depois_y == global_top_left_antes_y


def test_layout_resize_bounds_group_com_filhos_diretos():
    """Valida a propagação do resize_bounds para os filhos de um grupo."""
    group = GroupLayer(name="Root")
    layer1 = make_layer(50, 50, 40, 40)
    layer2 = make_layer(10, 10, 20, 20)
    group.append(layer1)
    group.append(layer2)

    layout = Layout()
    # Âncora 0.0 (Top-Left).
    # layer1 estava em 50, deve ficar em 50 com tamanho 100x100
    # layer2 estava em 10, deve ficar em 10 com tamanho 100x100
    result = layout.resize_bounds(group, 100, 100, 0.0, 0.0)

    assert result is True
    assert layer1.region == Region.from_rect(50, 50, 100, 100)
    assert layer2.region == Region.from_rect(10, 10, 100, 100)


def test_layout_resize_bounds_recursao_em_arvore_profunda():
    """Garante que o resize_bounds navega recursivamente em Grupos aninhados."""
    root = GroupLayer(name="Root")
    sub_group = GroupLayer(name="SubGroup")
    leaf_layer = make_layer(50, 50, 40, 40)

    sub_group.append(leaf_layer)
    root.append(sub_group)

    layout = Layout()
    # Âncora 0.5. Centro original é 70. Novo start = 70 - 100/2 = 20.
    result = layout.resize_bounds(root, 100, 100, 0.5, 0.5)

    assert result is True
    assert leaf_layer.region == Region.from_rect(20, 20, 100, 100)


def test_layout_resize_bounds_canvas():
    """Testa a expansão dos limites do Canvas."""
    canvas = Canvas(40, 40)  # Origem natural em 0,0, tamanho 40x40. Centro = 20.
    layout = Layout()
    # Expandindo para 100x100 no centro (0.5).
    # Centro é 20. Novo start = 20 - 50 = -30.
    result = layout.resize_bounds(canvas, 100, 100, 0.5, 0.5)

    assert result is True
    assert canvas.region == Region.from_rect(-30, -30, 100, 100)


# ############################# Testes para Layout.fit_content #####################################


def setup_layer_with_edits(layer_rect, edits_data):
    """Helper genérico que cria um Layer e Edits simulados, devolvendo também o side_effect para o mock."""
    layer = make_layer(*layer_rect)
    layer._edits = []

    mock_returns = {}
    for edit_rect, rect_data in edits_data:
        edit = make_edit(*edit_rect)
        edit.image = MagicMock()
        # Usa UUID para garantir que imagens diferentes de Layers diferentes não colidam no dict de mocks
        img_id = uuid.uuid4()
        edit.image._id = img_id
        layer._edits.append(edit)
        mock_returns[img_id] = Region.from_rect(*rect_data)

    def mock_rect_side_effect(img):
        return mock_returns[img._id]

    return layer, mock_rect_side_effect


def test_layout_fit_content_layer_totalmente_preenchido():
    """Testa fit_content num Layer sem bordas transparentes. Deve retornar False (No-Op)."""
    # Layer em 0,0 100x100. Edit no 0,0 local. Rect do Edit = 0,0 100x100
    layer, side_effect = setup_layer_with_edits(
        layer_rect=(0, 0, 100, 100),
        edits_data=[
            ((0, 0, 100, 100), (0, 0, 100, 100))
        ]
    )

    old_region = layer.region
    layout = Layout()

    with patch('anicrop.layout.calculate_content_rect', side_effect=side_effect):
        result = layout.fit_content(layer)

    assert result is False
    assert layer.region == old_region


def test_layout_fit_content_layer_com_bordas_transparentes():
    """Testa fit_content num Layer com bordas transparentes agressivas. O Layer deve encolher."""
    # Layer em 0,0. Edit em 0,0. Mas o conteúdo real (pixels) tá lá dentro do 30,30 com 40x40.
    layer, side_effect = setup_layer_with_edits(
        layer_rect=(0, 0, 100, 100),
        edits_data=[
            ((0, 0, 100, 100), (30, 30, 40, 40))
        ]
    )

    old_region = layer.region
    layout = Layout()

    with patch('anicrop.layout.calculate_content_rect', side_effect=side_effect):
        result = layout.fit_content(layer)

    assert result is True
    assert layer.region != old_region
    # A nova Bounding Box Global deve cravar onde estão os pixels
    assert layer.region == Region.from_rect(30, 30, 40, 40)


def test_layout_fit_content_layer_multiplos_edits_sobrepostos():
    """Testa 2 edits com bordas transparentes que se sobrepõem em eixos diferentes."""
    # Edit 1: Local (0,0). Rect relativa (20, 40, 50, 50). Global = (20, 40) até (70, 90).
    # Edit 2: Local (50,0). Rect relativa (10, 10, 50, 50). Global = (60, 10) até (110, 60).
    # UNIÃO GLOBAL ESPERADA:
    # X Min = 20. X Max = 110. (Largura 90)
    # Y Min = 10. Y Max = 90. (Altura 80)
    layer, side_effect = setup_layer_with_edits(
        layer_rect=(0, 0, 200, 200),
        edits_data=[
            ((0, 0, 100, 100), (20, 40, 50, 50)),
            ((50, 0, 100, 100), (10, 10, 50, 50))
        ]
    )

    layout = Layout()
    with patch('anicrop.layout.calculate_content_rect', side_effect=side_effect):
        result = layout.fit_content(layer)

    assert result is True
    assert layer.region == Region.from_rect(20, 10, 90, 80)


def test_layout_fit_content_layer_multiplos_edits_separados():
    """Testa 2 edits distantes que não se sobrepõem, o layer deve abraçar ambos."""
    # Edit 1: Local (10,10). Rect relativa (0, 0, 20, 20). Global = (10, 10) até (30, 30).
    # Edit 2: Local (200,200). Rect relativa (0, 0, 20, 20). Global = (200, 200) até (220, 220).
    # UNIÃO GLOBAL ESPERADA:
    # (10, 10) com tamanho 210x210
    layer, side_effect = setup_layer_with_edits(
        layer_rect=(0, 0, 300, 300),
        edits_data=[
            ((10, 10, 100, 100), (0, 0, 20, 20)),
            ((200, 200, 100, 100), (0, 0, 20, 20))
        ]
    )

    layout = Layout()
    with patch('anicrop.layout.calculate_content_rect', side_effect=side_effect):
        result = layout.fit_content(layer)

    assert result is True
    assert layer.region == Region.from_rect(10, 10, 210, 210)


def test_layout_fit_content_revela_arte_escondida_fora_da_borda():
    """Testa se fit_content expande a camada para recuperar conteúdo desenhado fora das bordas (não-destrutivo)."""
    # Camada original: Global (100, 100) com tamanho 50x50 (Vai até o 150).
    # EditLayer (arte): Movido para o Local (-20, -20).
    # Rect real da arte: Nasce no 0,0 interno do Edit com tamanho 30x30.
    # Posição Global da Arte: Layer(100) + Edit(-20) = 80. Vai do 80 até 110.
    # Ou seja: A arte está vazando violentamente pela esquerda/topo da camada!
    layer, side_effect = setup_layer_with_edits(
        layer_rect=(100, 100, 50, 50),
        edits_data=[
            ((-20, -20, 100, 100), (0, 0, 30, 30))
        ]
    )

    layout = Layout()
    with patch('anicrop.layout.calculate_content_rect', side_effect=side_effect):
        result = layout.fit_content(layer)

    assert result is True
    # A nova Bounding Box da Layer deve correr atrás da arte e revelar ela cravando no 80,80.
    assert layer.region == Region.from_rect(80, 80, 30, 30)


def test_layout_fit_content_group_layer():
    """Testa se o fit_content propaga para as camadas dentro de um GroupLayer, testando fora da origem."""
    group = GroupLayer(name="Root")

    # Layer 1: Em (50, 50). Arte em Local (10, 10). Rect (0, 0, 40, 40).
    # Posição global da arte 1: Layer(50) + Local(10) + Rect(0) = 60.
    layer1, side_effect1 = setup_layer_with_edits(
        layer_rect=(50, 50, 100, 100),
        edits_data=[((10, 10, 100, 100), (0, 0, 40, 40))]
    )

    # Layer 2: Em (200, 200). Arte em Local (-10, -10). Rect (0, 0, 50, 50).
    # Posição global da arte 2: Layer(200) + Local(-10) + Rect(0) = 190.
    layer2, side_effect2 = setup_layer_with_edits(
        layer_rect=(200, 200, 100, 100),
        edits_data=[((-10, -10, 100, 100), (0, 0, 50, 50))]
    )

    group.append(layer1)
    group.append(layer2)

    layout = Layout()

    # Combina os mocks
    def side_effect_combo(img):
        try:
            return side_effect1(img)
        except KeyError:
            return side_effect2(img)

    with patch('anicrop.layout.calculate_content_rect', side_effect=side_effect_combo):
        result = layout.fit_content(group)

    assert result is True
    assert layer1.region == Region.from_rect(60, 60, 40, 40)
    assert layer2.region == Region.from_rect(190, 190, 50, 50)


def test_layout_fit_content_group_layer_recursivo():
    """Testa se o fit_content propaga recursivamente por sub-grupos, testando fora da origem com negativos."""
    root = GroupLayer(name="Root")
    sub_group = GroupLayer(name="SubGroup")

    # Layer: Em (-100, -100). Arte em Local (50, 50). Rect real interna (10, 10, 30, 30).
    # Posição global final da arte: Layer(-100) + Local(50) + Rect(10) = -40.
    layer, side_effect = setup_layer_with_edits(
        layer_rect=(-100, -100, 200, 200),
        edits_data=[((50, 50, 100, 100), (10, 10, 30, 30))]
    )

    sub_group.append(layer)
    root.append(sub_group)

    layout = Layout()
    with patch('anicrop.layout.calculate_content_rect', side_effect=side_effect):
        result = layout.fit_content(root)

    assert result is True
    assert layer.region == Region.from_rect(-40, -40, 30, 30)


def test_layout_fit_content_group_bug_sobrescrita_resultado():
    """Testa se o retorno True é preservado mesmo se uma camada subsequente não sofrer alteração."""
    group = GroupLayer(name="Root")

    # Layer 1: PRECISA de alteração. Bordas transparentes sobrando (Retorna True).
    layer1, side_effect1 = setup_layer_with_edits(
        layer_rect=(0, 0, 100, 100),
        edits_data=[((0, 0, 100, 100), (10, 10, 40, 40))]
    )

    # Layer 2: NÃO precisa de alteração. Já está cravado (Retorna False).
    layer2, side_effect2 = setup_layer_with_edits(
        layer_rect=(0, 0, 50, 50),
        edits_data=[((0, 0, 50, 50), (0, 0, 50, 50))]
    )

    # Adicionamos na ordem: primeiro o que muda (True), depois o que NÃO muda (False).
    group.append(layer1)
    group.append(layer2)

    layout = Layout()

    def side_effect_combo(img):
        try:
            return side_effect1(img)
        except KeyError:
            return side_effect2(img)

    with patch('anicrop.layout.calculate_content_rect', side_effect=side_effect_combo):
        result = layout.fit_content(group)

    # Como a Layer 1 sofreu alteração (True), o Grupo todo como conjunto deve relatar True!
    assert result is True


@pytest.mark.parametrize('container_class', [GroupLayer, LayerStack])
@pytest.mark.parametrize(
    "layer1_rect, layer2_rect, mock_content, expected_region",
    [
        # Sem Sobreposição + Expand
        pytest.param(
            (10, 10, 50, 50), (100, 100, 50, 50),
            Region.from_rect(-20, -20, 90, 90),
            Region.from_rect(-10, -10, 180, 180),
            id="no_overlap_expand"
        ),
        # Sem Sobreposição + Shrink
        pytest.param(
            (10, 10, 50, 50), (100, 100, 50, 50),
            Region.from_rect(10, 10, 10, 10),
            Region.from_rect(20, 20, 100, 100),
            id="no_overlap_shrink"
        ),
        # Com Sobreposição + Expand
        pytest.param(
            (10, 10, 50, 50), (30, 30, 50, 50),
            Region.from_rect(-20, -20, 90, 90),
            Region.from_rect(-10, -10, 110, 110),
            id="overlap_expand"
        ),
        # Com Sobreposição + Shrink
        pytest.param(
            (10, 10, 50, 50), (30, 30, 50, 50),
            Region.from_rect(10, 10, 10, 10),
            Region.from_rect(20, 20, 30, 30),
            id="overlap_shrink"
        ),
        # Totalmente contida + Expand
        pytest.param(
            (10, 10, 100, 100), (50, 50, 30, 30),
            Region.from_rect(-20, -20, 90, 90),
            Region.from_rect(-10, -10, 130, 130),
            id="fully_contained_expand"
        ),
    ]
)
def test_layout_fit_content_canvas_integracao(
    container_class, layer1_rect, layer2_rect, mock_content, expected_region
):
    """
    Testa o fit_content no Canvas com GroupLayer e LayerStack.
    Cria uma hierarquia com 1 root e 1 sub-root, testando a adaptação do Canvas
    diretamente pelo conteúdo das layers (simulado pelo mock).
    """
    canvas = Canvas(500, 500)
    container = container_class()
    sub_container = GroupLayer()

    layer1 = make_layer(*layer1_rect)
    layer2 = make_layer(*layer2_rect)

    sub_container.append(layer2)
    container.append(layer1)
    container.append(sub_container)

    layout = Layout()

    with patch('anicrop.layout.calculate_content_rect', return_value=mock_content):
        # Ajusta o Canvas ao conteúdo dos pixels das layers
        result = layout.fit_content(canvas, container=container)

    assert result is True
    assert canvas.region == expected_region


def test_layout_fit_content_canvas_vazio():
    """Garante que se não houver layers, o canvas não altera seu tamanho e retorna False."""
    canvas = Canvas(200, 200)
    layout = Layout()

    result = layout.fit_content(canvas, container=LayerStack())

    assert result is False
    assert canvas.region == Region.from_rect(0, 0, 200, 200)


def test_layout_fit_content_canvas_com_grupo_vazio():
    """
    Testa se o fit_content suporta passar um container que contém um Sub-Grupo vazio.
    Sem a proteção no `_resolve_loop`, este teste lançará TypeError: unsupported operand type(s) for |=: 'NoneType' and 'NoneType'.
    """
    canvas = Canvas(500, 500)
    container = LayerStack()

    # Adicionamos um GroupLayer totalmente vazio
    empty_group = GroupLayer()

    container.append(empty_group)

    layout = Layout()

    # Tenta ajustar o Canvas. Como o grupo não tem nada, não há conteúdo, então deve retornar False.
    result = layout.fit_content(canvas, container=container)

    assert result is False


def test_layout_fit_content_canvas_sem_alteracao():
    """
    Testa se o fit_content do Canvas retorna False quando o conteúdo resultante
    é exatamente igual à região que o Canvas já possui (evitando re-render desnecessário).
    """
    # Canvas já começa englobando perfeitamente a área (-10, -10, 180, 180)
    canvas = Canvas(200, 200)
    canvas._region = Region.from_rect(-10, -10, 180, 180)

    container = LayerStack()
    # Adiciona layers que combinados resultam exatamente em (-10, -10, 180, 180)
    layer1 = make_layer(10, 10, 50, 50)
    layer2 = make_layer(100, 100, 50, 50)
    container.append(layer1)
    container.append(layer2)

    layout = Layout()

    # O mock dirá que as layers cresceram -20 (como no teste de no_overlap_expand)
    mock_content = Region.from_rect(-20, -20, 90, 90)

    with patch('anicrop.layout.calculate_content_rect', return_value=mock_content):
        result = layout.fit_content(canvas, container=container)

    # Como a união do conteúdo (-10, -10, 180, 180) é igual à region que o Canvas já tinha,
    # ele deve ignorar a alteração e retornar False!
    assert result is False
