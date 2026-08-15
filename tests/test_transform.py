import numpy as np
import pytest
from anicrop.transform import (
    mat_translation,
    calculate_new_rect,
    mat_position,
    ComposerRel,
    TransformRel,
    ComposerAbs,
    TransformAbs,
    Transform,
    has_distortion,
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


def test_calculate_new_rect_rotacao_90_graus():
    rot_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    size = (100, 50)
    rect = calculate_new_rect(rot_90, size)
    np.testing.assert_allclose(rect, (-50, 0, 50, 100), atol=ATOL)


# --- Testes ComposerRel ---

def test_composer_rel_inicializacao():
    composer = ComposerRel((100, 100))
    np.testing.assert_allclose(composer.matrix, np.eye(3, dtype=np.float32), atol=ATOL)
    assert composer.size == (100, 100)


def test_composer_rel_translate_simples():
    composer = ComposerRel((100, 100))
    composer.translate(10, 20)

    expected = np.array([
        [1, 0, 10],
        [0, 1, 20],
        [0, 0, 1]
    ], dtype=np.float32)

    np.testing.assert_allclose(composer.matrix, expected, atol=ATOL)


def test_composer_rel_rotate_no_centro():
    composer = ComposerRel((100, 100))
    composer.rotate(90, 0.5, 0.5)

    # Ponto (0,0) deve ir para (100, 0)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = composer.matrix @ pt
    np.testing.assert_allclose(res, [100, 0, 1], atol=ATOL)


def test_composer_rel_scale_no_centro():
    composer = ComposerRel((100, 100))
    composer.scale(2, 2, 0.5, 0.5)

    # Centro (50, 50) imóvel
    pt_centro = np.array([50, 50, 1], dtype=np.float32)
    res_centro = composer.matrix @ pt_centro
    np.testing.assert_allclose(res_centro, [50, 50, 1], atol=ATOL)


def test_composer_rel_dynamic_pivot_no_shift():
    # ESTE TESTE DEVE SER RED (FALHAR) NO COMPOSERREL ATUAL
    # Ele exige que o composer rastreie o Rect dinamicamente como a classe TransformRel
    composer = ComposerRel((100, 100))

    # 1. Rotaciona 45 graus no centro
    composer.rotate(45, 0.5, 0.5)
    m_rot = composer.matrix
    x_ref, _, _, _ = calculate_new_rect(m_rot, (100, 100))

    # 2. Escala 2x no pivô visual 0.
    # No sistema inteligente, o Top-Left deve permanecer estável (~ -20.71)
    composer.scale(2, 1, 0, 0)
    m_complex = composer.matrix
    x_res, _, _, _ = calculate_new_rect(m_complex, (100, 100))

    # Se falhar aqui, é porque o pivô inteligente não está implementado no Composer
    assert abs(x_res - x_ref) <= 1


def test_composer_rel_add_transform():
    composer = ComposerRel((100, 100))
    t = TransformRel().translate(50, 50)
    res = composer.add_transform(t, (100, 100))

    expected = mat_translation(50, 50)
    np.testing.assert_allclose(composer.matrix, expected, atol=ATOL)
    assert res is composer


# --- Testes TransformRel (Imutável / Pivôs Dinâmicos) ---

def test_transform_rel_inicializacao():
    t = TransformRel()
    assert not t.has_distortion
    np.testing.assert_allclose(t.get_matrix((100, 100)),
                               np.eye(3, dtype=np.float32), atol=ATOL)


def test_transform_rel_dynamic_pivot_no_shift():
    # Este teste já deve ser GREEN (Passar) no TransformRel
    size = (100, 100)

    t_rot = TransformRel().rotate(45, 0.5, 0.5)
    m_rot = t_rot.get_matrix(size)
    x_ref, _, _, _ = calculate_new_rect(m_rot, size)

    t_complex = TransformRel().rotate(45, 0.5, 0.5).scale(2, 1, 0, 0)
    m_complex = t_complex.get_matrix(size)
    x_res, _, _, _ = calculate_new_rect(m_complex, size)

    assert abs(x_res - x_ref) <= 1


def test_transform_rel_validation_errors():
    with pytest.raises(TypeError):
        TransformRel(intentions=[TTranslate(10, 10)])
    with pytest.raises(TypeError):
        TransformRel(translate=[TRotate(45)])


# --- Testes de Estresse ---

def test_transform_rel_stress_interleaved():
    size = (100, 100)
    t = TransformRel().translate(10, 10).rotate(45).scale(2, 2).translate(5, 5).rotate(-30)
    res_matrix = t.get_matrix(size)

    x, y, w, h = calculate_new_rect(res_matrix, size)
    assert x != 0
    assert w > 100


# --- Testes ComposerAbs ---

def test_composer_abs_inicializacao():
    composer = ComposerAbs((100, 100))
    np.testing.assert_allclose(composer.matrix, np.eye(3, dtype=np.float32), atol=ATOL)
    assert composer.size == (100, 100)


def test_composer_abs_translate_simples():
    composer = ComposerAbs((100, 100))
    composer.translate(10, 20)

    expected = mat_translation(10, 20)
    np.testing.assert_allclose(composer.matrix, expected, atol=ATOL)


def test_composer_abs_rotate_no_pivo_absoluto():
    composer = ComposerAbs((100, 100))
    composer.rotate(90, px=50, py=50)

    # Ponto (0,0) rodando 90° em torno de (50,50) vai para (100,0)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = composer.matrix @ pt
    np.testing.assert_allclose(res, [100, 0, 1], atol=ATOL)


def test_composer_abs_scale_no_pivo_absoluto():
    composer = ComposerAbs((100, 100))
    composer.scale(2, 2, px=50, py=50)

    # Centro (50, 50) imóvel
    pt_centro = np.array([50, 50, 1], dtype=np.float32)
    res_centro = composer.matrix @ pt_centro
    np.testing.assert_allclose(res_centro, [50, 50, 1], atol=ATOL)


def test_composer_abs_add_transform():
    composer = ComposerAbs((100, 100))
    t = TransformAbs().translate(50, 50)
    res = composer.add_transform(t)

    expected = mat_translation(50, 50)
    np.testing.assert_allclose(composer.matrix, expected, atol=ATOL)
    assert res is composer


# --- Testes TransformAbs ---

def test_transform_abs_inicializacao():
    t = TransformAbs()
    assert not t.has_distortion
    np.testing.assert_allclose(t.get_matrix(), np.eye(3, dtype=np.float32), atol=ATOL)


def test_transform_abs_rotate_e_scale_pivo_absoluto():
    t_rot = TransformAbs().rotate(90, px=50, py=50)
    m_rot = t_rot.get_matrix()
    pt = np.array([0, 0, 1], dtype=np.float32)
    np.testing.assert_allclose(m_rot @ pt, [100, 0, 1], atol=ATOL)

    t_scale = TransformAbs().scale(2, 2, px=50, py=50)
    m_scale = t_scale.get_matrix()
    pt_centro = np.array([50, 50, 1], dtype=np.float32)
    np.testing.assert_allclose(m_scale @ pt_centro, [50, 50, 1], atol=ATOL)


def test_transform_abs_validation_errors():
    with pytest.raises(TypeError):
        TransformAbs(intentions=[TTranslate(10, 10)])
    with pytest.raises(TypeError):
        TransformAbs(translate=[TRotate(45)])


def test_transform_abs_stress_interleaved():
    t = TransformAbs().translate(10, 10).rotate(45, px=50, py=50).scale(
        2, 2, px=10, py=10).translate(5, 5).rotate(-30, px=0, py=0)
    res_matrix = t.get_matrix()

    x, y, w, h = calculate_new_rect(res_matrix, (100, 100))
    assert x != 0
    assert w > 100


def test_transform_factory_classmethods():
    t_rel = Transform.relative()
    assert isinstance(t_rel, TransformRel)

    t_abs = Transform.absolute()
    assert isinstance(t_abs, TransformAbs)


def test_composer_sync_region():
    """Valida se o método sync_region atualiza a região de referência do Composer."""
    composer = ComposerRel((50, 50))
    assert composer.size == (50, 50)
    assert composer.region == Region.from_size(50, 50)

    # Sincroniza para uma nova região de 100x200 em (20, 30)
    new_region = Region.from_rect(20, 30, 100, 200)
    composer.sync_region(new_region)

    assert composer.size == (100, 200)
    assert composer.region == new_region


@pytest.mark.parametrize(
    "matrix, expected_has_distortion",
    [
        pytest.param(np.eye(3, dtype=np.float32), False, id="identidade_sem_distorcao"),
        pytest.param(mat_translation(10, 20), False, id="translacao_sem_distorcao"),
        pytest.param(TransformRel().rotate(45).get_matrix((100, 100)), True, id="rotacao_com_distorcao"),
        pytest.param(TransformRel().scale(2.0, 2.0).get_matrix((100, 100)), True, id="escala_com_distorcao"),
        pytest.param(TransformRel().translate(15, -30).get_matrix((100, 100)), False, id="translacao_relativa_sem_distorcao"),
    ],
)
def test_has_distortion_function(matrix, expected_has_distortion):
    """Valida se a função has_distortion identifica qualquer distorção afim (rotação/escala) na matriz."""
    assert has_distortion(matrix) is expected_has_distortion
