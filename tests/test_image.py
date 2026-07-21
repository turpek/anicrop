from anicrop.image import Image, ImageFormat
from anicrop.spatial import Region, Span
from pytest import raises
import numpy as np
import pytest
import zarr
from PIL import Image as PILImage


def region_(size=3, offset=0):
    return Region.from_size(size, size)


def make_region(w=3, h=3):
    return Region.from_size(w, h)


@pytest.mark.parametrize(
    "fmt,has_alpha,channels",
    [
        (ImageFormat.GRAY, False, 1),
        (ImageFormat.GRAY_ALPHA, True, 2),
        (ImageFormat.RGB, False, 3),
        (ImageFormat.RGBA, True, 4),
        (ImageFormat.CMYK, False, 4),
        (ImageFormat.CMYK_ALPHA, True, 5),
    ],
)
def test_image_format_contract(fmt, has_alpha, channels):
    assert fmt.has_alpha is has_alpha
    assert fmt.channels == channels


@pytest.mark.parametrize("shape", [(0, 0), (0, 1), (1, 0)], ids=["0x0", "0x1", "1x0"])
def test_Image_rejeita_dimensao_zero(shape):
    with pytest.raises(ValueError, match="image dimensions must be greater than zero"):
        Image(np.zeros(shape), ImageFormat.GRAY)


@pytest.mark.parametrize("shape", [[1], (1, 1, 1, 1)], ids=["1D", "4D"])
def test_Image_rejeita_se_nao_for_2D_ou_3D(shape):
    with raises(ValueError, match="image array must be 2D or 3D"):
        Image(np.zeros(shape), ImageFormat.GRAY)


@pytest.mark.parametrize(
    "shape, img_format",
    [
        ((1, 1), ImageFormat.GRAY),  # grayscale mínimo
        ((10, 20), ImageFormat.GRAY),  # grayscale comum
        ((1, 1, 1), ImageFormat.GRAY),  # grayscale com canal explícito
        ((10, 20, 1), ImageFormat.GRAY),  # grayscale com canal
        ((1, 1, 2), ImageFormat.GRAY_ALPHA),  # grayscale com canal alpha
        ((10, 20, 3), ImageFormat.RGB),  # RGB
        ((10, 20, 4), ImageFormat.RGBA),  # RGBA
    ],
    ids=[
        "gray-1x1",
        "gray-10x20",
        "gray-1x1x1",
        "gray-10x20x1",
        "gray-1x1x2",
        "rgb",
        "rgba",
    ],
)
def test_Image_aceita_formatos_validos(shape, img_format):
    Image(np.zeros(shape), img_format)


def test_Image_rejeita_0_canais():
    with raises(ValueError, match="image must have at least one channel"):
        Image(np.zeros((1, 1, 0)), ImageFormat.GRAY)


def test_Image_com_width_10():
    assert Image(np.zeros((10, 10)), ImageFormat.GRAY).width == 10


def test_Image_com_height_10():
    assert Image(np.zeros((10, 10)), ImageFormat.GRAY).height == 10


def test_Image_com_size_10x10():
    assert Image(np.zeros((10, 10)), ImageFormat.GRAY).size == (10, 10)


@pytest.mark.parametrize(
    "shape, expect, fmt",
    [
        ((1, 1), 1, ImageFormat.GRAY),  # Canal implícito
        ((1, 1, 1), 1, ImageFormat.GRAY),  # Canal explícito
        ((1, 1, 2), 2, ImageFormat.GRAY_ALPHA),  # Canal cinza+alpha
        ((1, 1, 3), 3, ImageFormat.RGB),  # Canal RGB
        ((1, 1, 4), 4, ImageFormat.RGBA),  # Canal RGBA
        ((1, 1, 4), 4, ImageFormat.CMYK),  # Canal CMYK
        ((1, 1, 5), 5, ImageFormat.CMYK_ALPHA),  # Canal CMYK+alpha
    ],
    ids=[
        "Grayscale_implicito",
        "Grayscale_explicito",
        "Grayscale+Alpha",
        "RGB",
        "RGBA",
        "CMYK",
        "CMYK+Alpha",
    ],
)
def test_Image_com_varios_canais(shape, fmt, expect):
    assert Image(np.zeros(shape), fmt).channels == expect


def test_Imagem_getitem_com_Region():
    region = Region.from_size(3, 3) + 3
    data = np.arange(10 * 10 * 3).reshape(10, 10, 3)
    img = Image(data, ImageFormat.RGB)
    sub = img[region]
    assert np.array_equal(sub, data[3:6, 3:6])


def test_Image_getitem_region_preserva_canais():
    img = Image(np.zeros((10, 10, 5)), ImageFormat.CMYK_ALPHA)
    region = Region.from_size(2, 2)
    assert img[region].shape[2] == 5


def test_Image_getitem_region_grayscale():
    img = Image(np.zeros((10, 10)), ImageFormat.GRAY)
    region = Region.from_size(4, 4)
    assert img[region].shape == (4, 4, 1)


def test_Image_getitem_region_com_slice_de_canais():
    img = Image(np.zeros((10, 10, 5)), ImageFormat.CMYK_ALPHA)
    region = Region.from_size(2, 2)
    assert img[region, :2].shape[2] == 2


@pytest.mark.parametrize(
    "args",
    [
        (make_region(), slice(1, 2), make_region(5, 1)),
        (slice(0, 5), slice(2, 4), slice(1, 2), make_region(5, 1)),
        (..., slice(1, 2), make_region(5, 1)),
    ],
    ids=[
        "region_slice_region",
        "slice_slice_region",
        "ellipsis_slice_region",
    ],
)
def test_Image_getitem_com_entradas_invalidas(args):
    with raises(TypeError, match="Region argument is only valid at the first position"):
        img = Image(np.zeros((10, 10, 5)), ImageFormat.CMYK_ALPHA)
        img[args]


def test_Image_setitem_com_Region():
    data = np.zeros((10, 10))
    img = Image(data, ImageFormat.GRAY)
    region = Region.from_size(2, 2)

    # Pinta de branco (1)
    img[region] = 1
    expected = np.zeros((10, 10))
    expected[0:2, 0:2] = 1
    assert np.array_equal(data, expected)


def test_Image_getitem_respeita_ordem_x_y_da_region():
    img = Image(np.zeros((10, 20)), ImageFormat.GRAY)

    # Region W=5, H=2
    # X=0..5, Y=0..2
    region = Region.from_size(5, 2)
    crop = img[region]
    assert crop.shape == (2, 5, 1)


@pytest.mark.parametrize(
    "shape,fmt",
    [
        ((10, 10, 4), ImageFormat.RGB),
        ((10, 10, 3), ImageFormat.RGBA),
        ((10, 10, 1), ImageFormat.GRAY_ALPHA),
        ((10, 10, 4), ImageFormat.CMYK_ALPHA),
    ],
)
def test_invalid_channel_count(shape, fmt):
    data = np.zeros(shape, dtype=np.uint8)
    msg = f"Image format '{fmt}' expects {fmt.channels} channels, but data has {shape[-1]}."

    with pytest.raises(ValueError, match=msg):
        Image(data, fmt)


def test_image_accepts_zarr_array():
    z_array = zarr.zeros(
        shape=(1000, 1000, 3),
        chunks=(128, 128, 3),
        dtype=np.uint8,
    )
    img = Image(z_array, ImageFormat.RGB)
    assert img.width == 1000
    assert img.height == 1000
    assert img.size == (1000, 1000)
    assert img.channels == 3
    assert img.shape == (1000, 1000, 3)


def test_image_zarr_getitem_with_region():
    z_array = zarr.zeros(shape=(100, 100, 3),
                         chunks=(50, 50, 3), dtype=np.uint8)
    z_array[10:20, 10:20, :] = 128
    img = Image(z_array, ImageFormat.RGB)
    region = Region(Span(10, 10), Span(10, 10))
    roi = img[region]
    assert isinstance(roi, np.ndarray)
    assert roi.shape == (10, 10, 3)
    assert np.all(roi == 128)


def test_image_zarr_grayscale_is_always_3d():
    z_array = zarr.zeros(shape=(100, 100, 1),
                         chunks=(50, 50, 1), dtype=np.uint8)
    z_array[5:10, 5:10, 0] = 255
    img = Image(z_array, ImageFormat.GRAY)
    assert img.shape == (100, 100, 1)
    region = Region(Span(5, 5), Span(5, 5))
    roi = img[region]
    assert isinstance(roi, np.ndarray)
    assert roi.shape == (5, 5, 1)
    assert np.all(roi == 255)


def test_image_open_routes_to_opencv(mocker):
    # Mock do Pillow para simular uma imagem 1000x1000 (normal)
    mock_img = mocker.MagicMock()
    mock_img.size = (1000, 1000)
    mocker.patch(
        "anicrop.image.PILImage.open"
    ).return_value.__enter__.return_value = mock_img

    mock_cv2 = mocker.patch(
        "anicrop.image.Image._open_with_opencv",
        return_value=mocker.MagicMock(spec=Image),
    )
    mock_zarr = mocker.patch("anicrop.image.Image._open_with_pillow_zarr")

    Image.open("dummy.png", ImageFormat.RGB)

    mock_cv2.assert_called_once_with("dummy.png", ImageFormat.RGB)
    mock_zarr.assert_not_called()


def test_image_open_routes_to_zarr(mocker):
    # Mock do Pillow para simular uma imagem 8192x8192 (limiar gigante)
    mock_img = mocker.MagicMock()
    mock_img.size = (8192, 8192)
    mocker.patch(
        "anicrop.image.PILImage.open"
    ).return_value.__enter__.return_value = mock_img

    mock_cv2 = mocker.patch("anicrop.image.Image._open_with_opencv")
    mock_zarr = mocker.patch(
        "anicrop.image.Image._open_with_pillow_zarr",
        return_value=mocker.MagicMock(spec=Image),
    )

    Image.open("giant.png", ImageFormat.RGB)

    mock_zarr.assert_called_once_with("giant.png", ImageFormat.RGB)
    mock_cv2.assert_not_called()


def test_open_with_opencv_loads_array(mocker):
    mock_data = np.zeros((100, 100, 3), dtype=np.uint8)
    mocker.patch("anicrop.image.cv2.imread", return_value=mock_data)

    img = Image._open_with_opencv("dummy.png", ImageFormat.RGB)
    assert img.shape == (100, 100, 3)
    assert img.channels == 3
    assert img.format == ImageFormat.RGB


def test_open_with_pillow_zarr_creates_3d_zarr(tmp_path):
    # Usamos o tmp_path para gerar um PNG real que o OpenCV não vai abrir
    source_path = tmp_path / "source.png"
    pil_img = PILImage.new("RGB", (600, 600), color=(255, 0, 0))
    pil_img.save(source_path)

    # Chamamos o conversor de blocos
    img = Image._open_with_pillow_zarr(str(source_path), ImageFormat.RGB)

    # Verifica se os dados se comportam corretamente no Wrapper Image
    assert img.shape == (600, 600, 3)
    assert img.channels == 3
    assert img.format == ImageFormat.RGB

    # Verifica se a região específica veio com a cor e dimensões corretas do chunk
    roi = img[Region(Span(100, 10), Span(100, 10))]
    assert isinstance(roi, np.ndarray)
    assert roi.shape == (10, 10, 3)
    assert np.all(roi == (255, 0, 0))


def test_image_open_opencv_converts_format(tmp_path):
    # Salva uma imagem RGB (vermelha) de teste
    source_path = tmp_path / "rgb.png"
    pil_img = PILImage.new("RGB", (100, 100), color=(255, 0, 0))
    pil_img.save(source_path)

    # 1. Abre solicitando GRAY (deve converter para 1 canal)
    img_gray = Image.open(source_path, ImageFormat.GRAY)
    assert img_gray.shape == (100, 100, 1)

    # 2. Abre solicitando RGBA (deve adicionar canal alfa com 255)
    img_rgba = Image.open(source_path, ImageFormat.RGBA)
    assert img_rgba.shape == (100, 100, 4)
    # A cor deve estar em RGB (255, 0, 0, 255) e não BGR (0, 0, 255, 255)
    assert np.all(img_rgba[0, 0] == (255, 0, 0, 255))


def test_image_open_zarr_converts_format(tmp_path):
    # Salva uma imagem RGB de teste gigante (simulada)
    source_path = tmp_path / "giant_rgb.png"
    pil_img = PILImage.new("RGB", (100, 100), color=(0, 255, 0))
    pil_img.save(source_path)

    # Força abertura via Zarr mockando o limiar de tamanho
    img_gray = Image._open_with_pillow_zarr(str(source_path), ImageFormat.GRAY)
    assert img_gray.shape == (100, 100, 1)

    img_rgba = Image._open_with_pillow_zarr(str(source_path), ImageFormat.RGBA)
    assert img_rgba.shape == (100, 100, 4)
    assert np.all(img_rgba[0, 0] == (0, 255, 0, 255))
