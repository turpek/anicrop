import numpy as np
import pytest
from anicrop.transform import (
    mat_translation,
    create_pivot_transform,
    calculate_new_bbox,
    mat_position,
    TransformComposer
)
from anicrop.spatial import Region, Span


def test_mat_translation_valores_basicos():
    m = mat_translation(10, 20)
    expected = np.array([
        [1, 0, 10],
        [0, 1, 20],
        [0, 0, 1]
    ], dtype=np.float32)
    np.testing.assert_array_equal(m, expected)


def test_mat_position_da_region():
    region = Region(Span(50, 100), Span(30, 80))
    m = mat_position(region)
    expected = mat_translation(50, 30)
    np.testing.assert_array_equal(m, expected)


def test_create_pivot_transform_com_identidade():
    identity = np.eye(3, dtype=np.float32)
    m = create_pivot_transform(identity, 100, 100, 0.5, 0.5)
    np.testing.assert_array_almost_equal(m, identity)


def test_create_pivot_transform_rotacao_90_graus_no_centro():
    rot_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    m = create_pivot_transform(rot_90, 100, 100, 0.5, 0.5)

    pt = np.array([0, 0, 1], dtype=np.float32)
    res = m @ pt
    np.testing.assert_array_almost_equal(res, [100, 0, 1])


def test_calculate_new_bbox_apenas_translacao():
    m = mat_translation(10, 20)
    size = (100, 50)
    bbox = calculate_new_bbox(m, size)
    np.testing.assert_allclose(bbox, (10, 20, 100, 50), atol=1e-5)


def test_calculate_new_bbox_rotacao_90_graus():
    rot_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    size = (100, 50)
    bbox = calculate_new_bbox(rot_90, size)
    np.testing.assert_allclose(bbox, (-50, 0, 50, 100), atol=1e-5)


# --- Testes TransformComposer ---

def test_transform_composer_inicializacao():
    composer = TransformComposer((100, 100))
    np.testing.assert_array_equal(composer.matrix, np.eye(3, dtype=np.float32))
    assert composer.size == (100, 100)
    assert not composer.has_rotation
    assert not composer.has_scale
    assert not composer.has_translation


def test_transform_composer_translate_simples():
    composer = TransformComposer((100, 100))
    composer.translate(10, 20)

    expected = np.array([
        [1, 0, 10],
        [0, 1, 20],
        [0, 0, 1]
    ], dtype=np.float32)

    np.testing.assert_array_equal(composer.matrix, expected)
    assert composer.has_translation


def test_transform_composer_rotate_no_centro():
    composer = TransformComposer((100, 100))
    composer.rotate(90, 0.5, 0.5)

    # Ponto (0,0) deve ir para (100, 0)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = composer.matrix @ pt

    np.testing.assert_array_almost_equal(res, [100, 0, 1])
    assert composer.has_rotation


def test_transform_composer_scale_no_centro():
    composer = TransformComposer((100, 100))
    composer.scale(2, 2, 0.5, 0.5)

    # Centro (50, 50) imóvel
    pt_centro = np.array([50, 50, 1], dtype=np.float32)
    res_centro = composer.matrix @ pt_centro
    np.testing.assert_array_almost_equal(res_centro, [50, 50, 1])

    # (0, 0) -> (-50, -50)
    pt_tl = np.array([0, 0, 1], dtype=np.float32)
    res_tl = composer.matrix @ pt_tl
    np.testing.assert_array_almost_equal(res_tl, [-50, -50, 1])
    assert composer.has_scale


def test_transform_composer_acumulacao_e_fluidez():
    composer = TransformComposer((100, 100))
    # Verifica chaining e ordem (M_new @ M_old)
    composer.translate(10, 0).scale(2, 2, 0.5, 0.5)

    # (0,0) -> translate(10,0) -> (10,0) -> scale2x(pivo 50,50) -> (-30, -50)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = composer.matrix @ pt
    np.testing.assert_array_almost_equal(res, [-30, -50, 1])
    assert composer.has_translation
    assert composer.has_scale


def test_transform_composer_ordem_inversa():
    composer = TransformComposer((100, 100))
    composer.scale(2, 2, 0.5, 0.5).translate(10, 0)

    # (0,0) -> scale2x(pivo 50,50) -> (-50, -50) -> translate(10,0) -> (-40, -50)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = composer.matrix @ pt
    np.testing.assert_array_almost_equal(res, [-40, -50, 1])


def test_transform_composer_scale_zero_raises_error():
    composer = TransformComposer((100, 100))
    with pytest.raises(ValueError, match="Scale factors cannot be zero"):
        composer.scale(sx=0)

    with pytest.raises(ValueError, match="Scale factors cannot be zero"):
        composer.scale(sy=0)


def test_transform_composer_has_flags_independencia():
    composer = TransformComposer((100, 100))

    composer.rotate(45)
    assert composer.has_rotation
    assert not composer.has_scale
    assert not composer.has_translation

    composer.scale(1.1, 1.1)
    assert composer.has_rotation
    assert composer.has_scale
    assert not composer.has_translation

    composer.translate(5, 5)
    assert composer.has_rotation
    assert composer.has_scale
    assert composer.has_translation
