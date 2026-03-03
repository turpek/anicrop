import numpy as np
import pytest
from anicrop.transform import (
    mat_translation,
    create_pivot_transform,
    calculate_new_bbox,
    mat_position,
    TransformComposer,
    # Transform  # Comentado até ser implementado
)
from anicrop.spatial import Region, Span

# Importando as intenções para teste direto se necessário
from anicrop.transform import TRotate, TScale, TTranslate


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


def test_transform_composer_translate_simples():
    composer = TransformComposer((100, 100))
    composer.translate(10, 20)

    expected = np.array([
        [1, 0, 10],
        [0, 1, 20],
        [0, 0, 1]
    ], dtype=np.float32)

    np.testing.assert_array_equal(composer.matrix, expected)


def test_transform_composer_rotate_no_centro():
    composer = TransformComposer((100, 100))
    composer.rotate(90, 0.5, 0.5)

    # Ponto (0,0) deve ir para (100, 0)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = composer.matrix @ pt
    np.testing.assert_array_almost_equal(res, [100, 0, 1])


def test_transform_composer_scale_no_centro():
    composer = TransformComposer((100, 100))
    composer.scale(2, 2, 0.5, 0.5)

    # Centro (50, 50) imóvel
    pt_centro = np.array([50, 50, 1], dtype=np.float32)
    res_centro = composer.matrix @ pt_centro
    np.testing.assert_array_almost_equal(res_centro, [50, 50, 1])


def test_transform_composer_acumulacao_e_fluidez():
    composer = TransformComposer((100, 100))
    composer.translate(10, 0).scale(2, 2, 0.5, 0.5)

    # (0,0) -> translate(10,0) -> (10,0) -> scale2x(pivo 50,50) -> (-30, -50)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = composer.matrix @ pt
    np.testing.assert_array_almost_equal(res, [-30, -50, 1])


def test_transform_composer_scale_zero_raises_error():
    composer = TransformComposer((100, 100))
    with pytest.raises(ValueError, match="Scale factors cannot be zero"):
        composer.scale(sx=0)


# --- Testes Transform (Imutável / Intenções) ---
# Nota: Estes testes falharão até que a classe Transform seja implementada

def test_transform_inicializacao():
    from anicrop.transform import Transform
    t = Transform()
    assert not t.has_rotation()
    assert not t.has_scale()
    assert not t.has_translation()
    assert len(t.intentions) == 0


def test_transform_imutabilidade():
    from anicrop.transform import Transform
    t1 = Transform()
    t2 = t1.translate(10, 10)
    t3 = t2.rotate(90)

    assert t1 is not t2
    assert t2 is not t3
    assert len(t1.intentions) == 0
    assert len(t2.intentions) == 1
    assert len(t3.intentions) == 2


def test_transform_has_flags():
    from anicrop.transform import Transform
    t = Transform()
    
    t_rot = t.rotate(45)
    assert t_rot.has_rotation()
    assert not t_rot.has_scale()
    
    t_scale = t.scale(2, 2)
    assert t_scale.has_scale()
    assert not t_scale.has_rotation()
    
    t_trans = t.translate(10, 10)
    assert t_trans.has_translation()
    assert not t_trans.has_scale()


def test_transform_with_initial_list():
    from anicrop.transform import Transform
    intentions = [TTranslate(10, 10), TRotate(90)]
    t = Transform(intentions)
    
    assert len(t.intentions) == 2
    assert t.has_translation()
    assert t.has_rotation()
    assert not t.has_scale()
