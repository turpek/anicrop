import numpy as np
import pytest

from anicrop.document import Document
from anicrop.enums import BlendMode, ImageFormat, InterpMode
from anicrop.history import GlobalHistory
from anicrop.image import Image
from anicrop.layer import Layer
from anicrop.layout import Layout
from anicrop.proxy import GroupProxy, LayerStackProxy, ProxyLayer
from anicrop.render import CanvasRender, ViewportRender
from anicrop.spatial import Region
from anicrop.viewport import Viewport


def make_img(w: int = 10, h: int = 10) -> Image:
    return Image(np.zeros((h, w, 4), dtype=np.uint8), ImageFormat.RGBA)


def make_solid(color: tuple[int, int, int, int], w: int = 50, h: int = 50) -> Image:
    data = np.zeros((h, w, 4), dtype=np.uint8)
    data[:] = color
    return Image(data, ImageFormat.RGBA)


def test_document_reactive_mode():
    """Valida inicialização do documento em modo reativo com histórico e proxies."""
    doc = Document("TestDoc", 100, 100, history=True)

    assert isinstance(doc.history, GlobalHistory)
    assert isinstance(doc.stack, LayerStackProxy)

    raw_layer = Layer(make_img(), name="L1")
    added_layer = doc.add(raw_layer)

    assert isinstance(added_layer, ProxyLayer)
    assert doc[0] is added_layer

    added_layer.name = "Renamed L1"
    assert raw_layer.name == "Renamed L1"

    doc.history.undo()
    assert raw_layer.name == "L1"


def test_document_direct_mode():
    """Valida inicialização do documento em modo direto de alta performance sem proxies."""
    doc = Document("TestDoc", 100, 100, history=False)

    assert doc.history is None
    assert not isinstance(doc.stack, LayerStackProxy)

    raw_layer = Layer(make_img(), name="L1")
    added_layer = doc.add(raw_layer)

    assert not isinstance(added_layer, ProxyLayer)
    assert isinstance(added_layer, Layer)
    assert doc[0] is raw_layer


def test_document_canvas_properties():
    """Valida propriedades de dimensão no objeto Canvas do Document."""
    doc = Document("TestDoc", 800, 600)

    assert doc.canvas.width == 800
    assert doc.canvas.height == 600
    assert doc.canvas.size == (800, 600)

    doc.canvas.region = Region.from_size(1920, 1080)

    assert doc.canvas.width == 1920
    assert doc.canvas.height == 1080
    assert doc.canvas.size == (1920, 1080)


def test_document_add_and_duplicate_name_error():
    """Valida se a adição de camada com nome duplicado levanta ValueError estrito."""
    doc = Document("TestDoc", 100, 100)
    l1 = Layer(make_img(), name="camada1")
    l2 = Layer(make_img(), name="camada1")

    doc.add(l1)

    with pytest.raises(ValueError, match="A layer named 'camada1' already exists"):
        doc.add(l2)


def test_document_add_group_mandatory_name_and_duplicate_error():
    """Valida criação de grupo com nome obrigatório e prevenção de colisão de nomes."""
    doc = Document("TestDoc", 100, 100)
    g1 = doc.add_group("grupo1")

    assert isinstance(g1, GroupProxy)
    assert g1.name == "grupo1"

    with pytest.raises(ValueError, match="A layer named 'grupo1' already exists"):
        doc.add_group("grupo1")


def test_document_container_sequence_protocol():
    """Valida protocolo de sequência (__len__, __getitem__, __iter__, __contains__)."""
    doc = Document("TestDoc", 100, 100)
    l1 = doc.add(Layer(make_img(), name="fundo"))
    l2 = doc.add(Layer(make_img(), name="texto"))
    l3 = doc.add(Layer(make_img(), name="overlay"))

    assert len(doc) == 3
    assert doc[0] is l1
    assert doc[1] is l2
    assert doc[-1] is l3
    assert list(doc) == [l1, l2, l3]
    assert "fundo" in doc
    assert "inexistente" not in doc
    assert l2 in doc


def test_document_getitem_by_name_and_keyerror():
    """Valida acesso a camadas por nome via string e KeyError para chaves inexistentes."""
    doc = Document("TestDoc", 100, 100)
    l1 = doc.add(Layer(make_img(), name="fundo"))

    assert doc["fundo"] is l1

    with pytest.raises(KeyError, match="Layer named 'nao_existe' not found"):
        _ = doc["nao_existe"]


def test_document_find_layer_recursive():
    """Valida busca de camadas aninhadas dentro de grupos usando find()."""
    doc = Document("TestDoc", 100, 100)
    group = doc.add_group("grupo_pai")
    filho = Layer(make_img(), name="filho_aninhado")
    group.append(filho)

    encontrado = doc.find("filho_aninhado", recursive=True)

    assert encontrado is not None
    assert encontrado.name == "filho_aninhado"
    assert doc.find("filho_aninhado", recursive=False) is None


def test_document_delitem_by_name_and_index():
    """Valida remoção de camadas via del doc[name] e del doc[index]."""
    doc = Document("TestDoc", 100, 100)
    doc.add(Layer(make_img(), name="l1"))
    doc.add(Layer(make_img(), name="l2"))
    l3 = doc.add(Layer(make_img(), name="l3"))

    del doc["l2"]
    assert len(doc) == 2
    assert "l2" not in doc

    del doc[0]
    assert len(doc) == 1
    assert doc[0] is l3


def test_document_remove_by_name_and_instance():
    """Valida método remove aceitando nome da camada ou instância do objeto."""
    doc = Document("TestDoc", 100, 100)
    doc.add(Layer(make_img(), name="l1"))
    l2 = doc.add(Layer(make_img(), name="l2"))

    doc.remove("l1")
    assert len(doc) == 1
    assert "l1" not in doc

    doc.remove(l2)
    assert len(doc) == 0


def test_document_render_in_memory():
    """Valida renderização em alta resolução direto para objeto Image em memória."""
    doc = Document("TestDoc", 50, 50)
    img_data = np.ones((50, 50, 4), dtype=np.uint8) * 200
    doc.add(Layer(Image(img_data, ImageFormat.RGBA), name="l1"))

    rendered_img = doc.render()

    assert isinstance(rendered_img, Image)
    assert rendered_img.size == (50, 50)
    assert rendered_img.format == ImageFormat.RGBA


def test_document_preview_returns_image():
    """Valida se doc.preview() retorna uma instância de Image."""
    from anicrop.viewport import Viewport
    doc = Document("TestDoc", 100, 100)
    doc.add(Layer(make_img(100, 100), name="l1"))
    viewport = Viewport((50, 50))

    preview_img = doc.preview(viewport)

    assert isinstance(preview_img, Image)
    assert preview_img.size == (50, 50)
    assert preview_img.format == ImageFormat.RGBA


def test_document_export_saves_file(tmp_path):
    """Valida se doc.export() gera o arquivo de imagem no disco."""
    doc = Document("TestDoc", 40, 40)
    img_data = np.ones((40, 40, 4), dtype=np.uint8) * 150
    doc.add(Layer(Image(img_data, ImageFormat.RGBA), name="l1"))

    export_file = tmp_path / "export_output.png"
    doc.export(export_file)

    assert export_file.exists()
    reloaded = Image.open(export_file, ImageFormat.RGBA)
    assert reloaded.size == (40, 40)


def test_document_layout_property_integration():
    """Valida operações de layout diretamente através da propriedade doc.layout."""
    doc = Document("TestDoc", 1000, 1000)
    l1 = doc.add(Layer(make_img(200, 200), name="l1"))

    assert isinstance(doc.layout, Layout)

    doc.layout.fit(l1, (0, 0, 500, 500))
    assert l1.region == Region.from_size(500, 500)

    doc.layout.align(doc["l1"], doc.canvas.region, 1.0, 1.0)
    assert l1.region == Region.from_rect(500, 500, 500, 500)

    doc.history.undo()
    assert l1.region == Region.from_size(500, 500)


def test_document_render_stack_z_order_bottom_to_top():
    """Valida se camadas adicionadas sequencialmente na raiz respeitam a sobreposição visual (última sobre a primeira)."""
    doc = Document("TestDoc", 50, 50)
    doc.add(Layer(make_solid((255, 0, 0, 255), 50, 50), name="fundo_vermelho"))
    doc.add(Layer(make_solid((0, 0, 255, 255), 50, 50), name="topo_azul"))

    rendered = doc.render()

    np.testing.assert_array_equal(rendered[0, 0], [0, 0, 255, 255])
    np.testing.assert_array_equal(rendered[25, 25], [0, 0, 255, 255])


def test_document_render_grouplayer_z_order_bottom_to_top():
    """Valida se camadas adicionadas em um grupo respeitam a ordem de sobreposição interna."""
    doc = Document("TestDoc", 50, 50)
    group = doc.add_group("grupo")
    group.append(Layer(make_solid((255, 0, 0, 255), 50, 50), name="g_fundo"))
    group.append(Layer(make_solid((0, 255, 0, 255), 50, 50), name="g_topo"))

    rendered = doc.render()

    np.testing.assert_array_equal(rendered[0, 0], [0, 255, 0, 255])
    np.testing.assert_array_equal(rendered[25, 25], [0, 255, 0, 255])


def test_document_render_interleaved_hierarchy_z_order():
    """Valida composição correta entre camadas soltas e grupos aninhados na pilha."""
    doc = Document("TestDoc", 50, 50)
    doc.add(Layer(make_solid((255, 0, 0, 255), 50, 50), name="base_fundo"))

    group = doc.add_group("grupo_meio")
    group.append(Layer(make_solid((0, 255, 0, 255), 50, 50), name="g_fundo"))
    group.append(Layer(make_solid((255, 255, 0, 255), 50, 50), name="g_topo"))

    doc.add(Layer(make_solid((0, 0, 255, 255), 50, 50), name="topo_geral"))

    # Topo geral azul cobre tudo
    rendered_all = doc.render()
    np.testing.assert_array_equal(rendered_all[25, 25], [0, 0, 255, 255])

    # Ocultando o topo geral, o topo do grupo (amarelo) cobre o fundo
    doc["topo_geral"].visible = False
    rendered_no_top = doc.render()
    np.testing.assert_array_equal(rendered_no_top[25, 25], [255, 255, 0, 255])


def test_document_reactive_set_mask_undo_redo():
    """Valida se operações de set_mask e remove_mask suportam Undo e Redo no Document reativo."""
    doc = Document("TestDoc", 100, 100, history=True)
    layer = doc.add(Layer(make_img(50, 50), name="L1"))

    mask_img = Image(np.full((50, 50, 1), 255, dtype=np.uint8), ImageFormat.GRAY)
    layer.set_mask(mask_img, Region.from_size(50, 50))
    assert layer.mask is not None

    doc.history.undo()
    assert layer.mask is None

    doc.history.redo()
    assert layer.mask is not None

    layer.remove_mask()
    assert layer.mask is None

    doc.history.undo()
    assert layer.mask is not None


@pytest.mark.parametrize(
    "format_option, expected_format",
    [
        (ImageFormat.RGB, ImageFormat.RGB),
        (ImageFormat.GRAY, ImageFormat.GRAY),
        (ImageFormat.RGBA, ImageFormat.RGBA),
    ],
    ids=["rgb", "gray", "rgba"],
)
def test_document_load_layer_with_format(tmp_path, format_option, expected_format):
    """Valida se load_layer carrega imagens respeitando o ImageFormat fornecido."""
    file_path = tmp_path / "sample.png"
    make_solid((200, 100, 50, 255), 20, 20).save(file_path)

    doc = Document("Doc", 50, 50)
    layer = doc.load_layer(file_path, name="L1", format=format_option)

    assert layer.format == expected_format


def test_document_load_layer_with_blend_mode(tmp_path):
    """Valida se load_layer define o blend_mode e opacidade corretamente na camada criada."""
    file_path = tmp_path / "sample.png"
    make_solid((200, 100, 50, 255), 20, 20).save(file_path)

    doc = Document("Doc", 50, 50)
    layer = doc.load_layer(file_path, name="L1", opacity=0.75, blend_mode=BlendMode.MULTIPLY)

    assert layer.blend_mode == BlendMode.MULTIPLY
    assert layer.opacity == 0.75


def test_document_open_with_format(tmp_path):
    """Valida se Document.open inicializa o documento com o formato de imagem especificado."""
    file_path = tmp_path / "sample.png"
    make_solid((200, 100, 50, 255), 30, 40).save(file_path)

    doc = Document.open(file_path, name="OpenedDoc", format=ImageFormat.RGB)

    assert doc.canvas.size == (30, 40)
    assert doc[0].format == ImageFormat.RGB


def test_document_open_with_blend_mode(tmp_path):
    """Valida se Document.open inicializa a camada raiz com o blend_mode e opacidade informados."""
    file_path = tmp_path / "sample.png"
    make_solid((200, 100, 50, 255), 30, 40).save(file_path)

    doc = Document.open(file_path, name="OpenedDoc", opacity=0.5, blend_mode=BlendMode.HARD_MASKING)

    assert doc[0].blend_mode == BlendMode.HARD_MASKING
    assert doc[0].opacity == 0.5


@pytest.mark.parametrize(
    "interp_mode",
    [
        InterpMode.NEAREST,
        InterpMode.LINEAR,
        InterpMode.CUBIC,
        InterpMode.LANCZOS,
    ],
    ids=["nearest", "linear", "cubic", "lanczos"],
)
def test_document_render_with_interp_mode(interp_mode):
    """Valida se doc.render aceita diferentes modos de interpolação."""
    doc = Document("Doc", 50, 50)
    doc.add(Layer(make_solid((255, 0, 0, 255), 50, 50), name="L1"))

    rendered = doc.render(interp=interp_mode)

    assert isinstance(rendered, Image)
    assert rendered.size == (50, 50)


def test_document_preview_with_interp_mode():
    """Valida se doc.preview aceita o parâmetro interp com InterpMode."""
    doc = Document("Doc", 100, 100)
    doc.add(Layer(make_solid((0, 255, 0, 255), 100, 100), name="L1"))

    viewport = Viewport((50, 50), 1.0)
    preview_img = doc.preview(viewport, interp=InterpMode.NEAREST)

    assert isinstance(preview_img, Image)
    assert preview_img.size == (50, 50)


def test_document_export_with_interp_mode(tmp_path):
    """Valida se doc.export salva o arquivo no disco aceitando o parâmetro interp."""
    doc = Document("Doc", 50, 50)
    doc.add(Layer(make_solid((0, 0, 255, 255), 50, 50), name="L1"))

    out_file = tmp_path / "exported.png"
    doc.export(out_file, interp=InterpMode.LINEAR)

    assert out_file.exists()


@pytest.mark.parametrize(
    "fmt, num_channels",
    [
        (ImageFormat.RGB, 3),
        (ImageFormat.RGBA, 4),
        (ImageFormat.GRAY, 1),
    ],
    ids=["rgb", "rgba", "gray"],
)
def test_document_format_render_pipeline(fmt, num_channels):
    """Valida se o Document renderiza cena no formato explicitado no render."""
    doc = Document("FormatDoc", 50, 50)

    raw_data = np.full((50, 50, num_channels), 180, dtype=np.uint8)
    doc.add(Layer(Image(raw_data, fmt), name="Layer1"))

    rendered = doc.render(format=fmt)
    assert rendered.format == fmt
    assert rendered.shape == (50, 50, num_channels)


def test_document_render_properties():
    """Valida se Document expoem as properties canvas_render e viewport_render."""
    doc = Document("TestDoc", 100, 100)

    assert isinstance(doc.canvas_render, CanvasRender)
    assert isinstance(doc.viewport_render, ViewportRender)
