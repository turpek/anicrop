import numpy as np
import pytest
from anicrop.transform import (
    mat_translation,
    create_pivot_transform,
    calculate_new_bbox,
    mat_position,
    TransformComposer,
    Transform
)
from anicrop.spatial import Region, Span

# Importando as intenções para teste direto
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

def test_transform_inicializacao():
    t = Transform()
    assert not t.has_distortion
    # Identidade por padrão
    np.testing.assert_array_equal(t.get_matrix(
        (100, 100)), np.eye(3, dtype=np.float32))


def test_transform_imutabilidade():
    t1 = Transform()
    t2 = t1.translate(10, 10)
    t3 = t2.rotate(90)

    assert t1 is not t2
    assert t2 is not t3

    # t1 deve continuar sendo identidade
    np.testing.assert_array_equal(t1.get_matrix(
        (100, 100)), np.eye(3, dtype=np.float32))


def test_transform_has_distortion_cenarios():
    t = Transform()

    # Apenas translação NÃO é distorção
    t_trans = t.translate(10, 10)
    assert not t_trans.has_distortion

    # Rotação É distorção
    t_rot = t.rotate(45)
    assert t_rot.has_distortion

    # Escala É distorção
    t_scale = t.scale(2, 2)
    assert t_scale.has_distortion

    # Translação + Rotação É distorção
    t_complex = t.translate(5, 5).rotate(10)
    assert t_complex.has_distortion


def test_transform_get_matrix_composicao():
    # 1. Translate (10, 0) -> 2. Scale 2x no Centro
    t = Transform().translate(10, 0).scale(2, 2, 0.5, 0.5)

    matrix = t.get_matrix((100, 100))

    # Ponto (0,0) -> translate(10,0) -> (10,0) -> scale2x(pivo 50,50) -> (-30, -50)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = matrix @ pt
    np.testing.assert_array_almost_equal(res, [-30, -50, 1])


def test_transform_get_matrix_independencia_de_tamanho():
    t = Transform().rotate(90, 0.5, 0.5)

    m100 = t.get_matrix((100, 100))
    res100 = m100 @ [0, 0, 1]
    np.testing.assert_array_almost_equal(res100, [100, 0, 1])

    m200 = t.get_matrix((200, 200))
    res200 = m200 @ [0, 0, 1]
    np.testing.assert_array_almost_equal(res200, [200, 0, 1])


def test_transform_with_initial_list():
    # Simulando passagem de lista de intenções (se a classe suportar)
    # TTranslate não causa distorção, TRotate sim.
    initial_list = [TTranslate(10, 10), TRotate(90)]
    t = Transform(initial_list)

    assert t.has_distortion

    matrix = t.get_matrix((100, 100))
    res = matrix @ [0, 0, 1]
    np.testing.assert_array_almost_equal(res, [90, 10, 1])
