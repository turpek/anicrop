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


def test_transform_composer_dynamic_pivot_no_shift():
    # ESTE TESTE DEVE SER RED (FALHAR) NO TRANSFORMCOMPOSER ATUAL
    # Ele exige que o composer rastreie o BBox dinamicamente como a classe Transform
    composer = TransformComposer((100, 100))

    # 1. Rotaciona 45 graus no centro
    composer.rotate(45, 0.5, 0.5)
    m_rot = composer.matrix
    x_ref, _, _, _ = calculate_new_bbox(m_rot, (100, 100))

    # 2. Escala 2x no pivô visual 0.
    # No sistema inteligente, o Top-Left deve permanecer estável (~ -20.71)
    composer.scale(2, 1, 0, 0)
    m_complex = composer.matrix
    x_res, _, _, _ = calculate_new_bbox(m_complex, (100, 100))

    # Se falhar aqui, é porque o pivô inteligente não está implementado no Composer
    assert abs(x_res - x_ref) <= 1


def test_transform_composer_add_transform():
    composer = TransformComposer((100, 100))
    t = Transform().translate(50, 50)
    res = composer._add_transform(t, (100, 100))

    expected = mat_translation(50, 50)
    np.testing.assert_allclose(composer.matrix, expected, atol=ATOL)
    assert res is None


# --- Testes Transform (Imutável / Pivôs Dinâmicos) ---

def test_transform_inicializacao():
    t = Transform()
    assert not t.has_distortion
    np.testing.assert_allclose(t.get_matrix((100, 100)), np.eye(3, dtype=np.float32), atol=ATOL)


def test_transform_dynamic_pivot_no_shift():
    # Este teste já deve ser GREEN (Passar) no Transform
    size = (100, 100)

    t_rot = Transform().rotate(45, 0.5, 0.5)
    m_rot = t_rot.get_matrix(size)
    x_ref, _, _, _ = calculate_new_bbox(m_rot, size)

    t_complex = Transform().rotate(45, 0.5, 0.5).scale(2, 1, 0, 0)
    m_complex = t_complex.get_matrix(size)
    x_res, _, _, _ = calculate_new_bbox(m_complex, size)

    assert abs(x_res - x_ref) <= 1


def test_transform_validation_errors():
    with pytest.raises(TypeError):
        Transform(intentions=[TTranslate(10, 10)])
    with pytest.raises(TypeError):
        Transform(translate=[TRotate(45)])


# --- Testes de Estresse ---

def test_transform_stress_interleaved():
    size = (100, 100)
    t = Transform().translate(10, 10).rotate(45).scale(2, 2).translate(5, 5).rotate(-30)
    res_matrix = t.get_matrix(size)

    x, y, w, h = calculate_new_bbox(res_matrix, size)
    assert x != 0
    assert w > 100
