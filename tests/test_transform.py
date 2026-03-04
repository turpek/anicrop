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

# Tolerância padrão para float32
ATOL = 1e-4


def test_mat_translation_valores_basicos():
    m = mat_translation(10, 20)
    expected = np.array([
        [1, 0, 10],
        [0, 1, 20],
        [0, 0, 1]
    ], dtype=np.float32)
    np.testing.assert_allclose(m, expected, atol=ATOL)


def test_mat_position_da_region():
    region = Region(Span(50, 100), Span(30, 80))
    m = mat_position(region)
    expected = mat_translation(50, 30)
    np.testing.assert_allclose(m, expected, atol=ATOL)


def test_calculate_new_bbox_rotacao_90_graus():
    rot_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    size = (100, 50)
    bbox = calculate_new_bbox(rot_90, size)
    np.testing.assert_allclose(bbox, (-50, 0, 50, 100), atol=ATOL)


# --- Testes TransformComposer ---

def test_transform_composer_inicializacao():
    composer = TransformComposer((100, 100))
    np.testing.assert_allclose(composer.matrix, np.eye(3, dtype=np.float32), atol=ATOL)
    assert composer.size == (100, 100)


def test_transform_composer_translate_simples():
    composer = TransformComposer((100, 100))
    composer.translate(10, 20)

    expected = np.array([
        [1, 0, 10],
        [0, 1, 20],
        [0, 0, 1]
    ], dtype=np.float32)

    np.testing.assert_allclose(composer.matrix, expected, atol=ATOL)


def test_transform_composer_rotate_no_centro():
    composer = TransformComposer((100, 100))
    composer.rotate(90, 0.5, 0.5)

    # Ponto (0,0) deve ir para (100, 0)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = composer.matrix @ pt
    np.testing.assert_allclose(res, [100, 0, 1], atol=ATOL)


def test_transform_composer_scale_no_centro():
    composer = TransformComposer((100, 100))
    composer.scale(2, 2, 0.5, 0.5)

    # Centro (50, 50) imóvel
    pt_centro = np.array([50, 50, 1], dtype=np.float32)
    res_centro = composer.matrix @ pt_centro
    np.testing.assert_allclose(res_centro, [50, 50, 1], atol=ATOL)


def test_transform_composer_acumulacao_e_fluidez():
    composer = TransformComposer((100, 100))
    # Com a lógica de translação final:
    # translate(10,0) define o posicionamento final.
    # scale(2) atua sobre o asset na origem local.
    composer.translate(10, 0).scale(2, 2, 0.5, 0.5)

    # (0,0) -> scale2x(pivo 50,50) -> (-50, -50)
    # (-50, -50) -> translate global(10,0) -> (-40, -50)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = composer.matrix @ pt
    np.testing.assert_allclose(res, [-40, -50, 1], atol=ATOL)


def test_transform_composer_add_transform():
    composer = TransformComposer((100, 100))
    t = Transform().translate(50, 50)
    
    # Passando o size explicitamente na nova assinatura
    res = composer._add_transform(t, (100, 100))
    
    # T(50,50) @ ID = T(50,50)
    expected = mat_translation(50, 50)
    np.testing.assert_allclose(composer.matrix, expected, atol=ATOL)
    assert res is None


def test_transform_composer_add_transform_composicao():
    composer = TransformComposer((100, 100))
    composer.translate(10, 10)
    
    t2 = Transform().translate(20, 20)
    composer._add_transform(t2, (100, 100))
    
    # T(20) @ T(10) = T(30)
    expected = mat_translation(30, 30)
    np.testing.assert_allclose(composer.matrix, expected, atol=ATOL)


def test_transform_composer_scale_zero_raises_error():
    composer = TransformComposer((100, 100))
    with pytest.raises(ValueError, match="Scale factors cannot be zero"):
        composer.scale(sx=0)


# --- Testes Transform (Imutável / Intenções) ---

def test_transform_inicializacao():
    t = Transform()
    assert not t.has_distortion
    np.testing.assert_allclose(t.get_matrix((100, 100)), np.eye(3, dtype=np.float32), atol=ATOL)


def test_transform_imutabilidade():
    t1 = Transform()
    t2 = t1.translate(10, 10)
    t3 = t2.rotate(90)

    assert t1 is not t2
    assert t2 is not t3

    np.testing.assert_allclose(t1.get_matrix((100, 100)), np.eye(3, dtype=np.float32), atol=ATOL)


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
    # Transform imutável aplica T_global @ M_distortion
    t = Transform().translate(10, 0).scale(2, 2, 0.5, 0.5)
    matrix = t.get_matrix((100, 100))

    # (0,0) -> scale2x(pivo 50,50) -> (-50, -50)
    # (-50, -50) -> translate global(10,0) -> (-40, -50)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = matrix @ pt
    np.testing.assert_allclose(res, [-40, -50, 1], atol=ATOL)


def test_transform_get_matrix_independencia_de_tamanho():
    t = Transform().rotate(90, 0.5, 0.5)

    m100 = t.get_matrix((100, 100))
    res100 = m100 @ [0, 0, 1]
    np.testing.assert_allclose(res100, [100, 0, 1], atol=ATOL)

    m200 = t.get_matrix((200, 200))
    res200 = m200 @ [0, 0, 1]
    np.testing.assert_allclose(res200, [200, 0, 1], atol=ATOL)


def test_transform_with_initial_list():
    # Usando os novos argumentos do __init__
    intentions = [TRotate(90)]
    translation = [TTranslate(10, 10)]
    t = Transform(intentions=intentions, translate=translation)

    assert t.has_distortion
    matrix = t.get_matrix((100, 100))
    
    # (0,0) -> rotate90(centro 50,50) -> (100,0)
    # (100,0) -> translate global(10,10) -> (110, 10)
    res = matrix @ [0, 0, 1]
    np.testing.assert_allclose(res, [110, 10, 1], atol=ATOL)


def test_transform_validation_wrong_types_in_lists():
    with pytest.raises(TypeError, match="intentions list can only contain TRotate or TScale"):
        Transform(intentions=[TTranslate(10, 10)])

    with pytest.raises(TypeError, match="translate list can only contain TTranslate"):
        Transform(translate=[TRotate(45)])


# --- Testes de Estresse (Sequências Complexas) ---

def test_transform_stress_translation_rotation_interleaved():
    size = (100, 100)
    t = Transform().translate(10, 10).rotate(45).scale(2, 2).translate(5, 5).rotate(-30)
    
    res_matrix = t.get_matrix(size)
    
    # T_total global @ (R_last @ S @ R_first)
    t_total = mat_translation(15, 15)
    m_rs = TRotate(-30).matrix(size) @ TScale(2, 2).matrix(size) @ TRotate(45).matrix(size)
    expected = t_total @ m_rs
    
    np.testing.assert_allclose(res_matrix, expected, atol=ATOL)


def test_transform_stress_multiple_rotations_accumulation():
    size = (100, 100)
    t_step = Transform()
    for _ in range(9):
        t_step = t_step.rotate(10, 0.5, 0.5)
        
    t_90 = Transform().rotate(90, 0.5, 0.5)
    
    np.testing.assert_allclose(t_step.get_matrix(size), t_90.get_matrix(size), atol=ATOL)


def test_transform_stress_translation_only_no_orbit():
    size = (100, 100)
    t = Transform().translate(50, 50)
    for _ in range(36):
        t = t.rotate(10, 0.5, 0.5)
        
    res_matrix = t.get_matrix(size)
    expected = mat_translation(50, 50)
    
    np.testing.assert_allclose(res_matrix, expected, atol=ATOL)
