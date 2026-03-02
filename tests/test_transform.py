import numpy as np
from anicrop.transform import (
    mat_translation,
    create_pivot_transform,
    calculate_new_bbox,
    mat_position,
    Transform
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
    # Uma matriz identidade não deve mudar nada mesmo com pivô
    identity = np.eye(3, dtype=np.float32)
    m = create_pivot_transform(identity, 100, 100, 0.5, 0.5)
    np.testing.assert_array_almost_equal(m, identity)


def test_create_pivot_transform_rotacao_90_graus_no_centro():
    # Rotacionar 90 graus no centro (50, 50) de um quadrado 100x100
    # O ponto (0,0) deve ir para (100, 0) se o eixo Y for para baixo e girarmos sentido horário
    # R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    rot_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    m = create_pivot_transform(rot_90, 100, 100, 0.5, 0.5)

    # Projetar o ponto (0,0) local
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = m @ pt

    # No centro (50,50), (0,0) está em (-50, -50) relativo ao pivô.
    # Rotacionado 90 deg: x' = -(-50) = 50, y' = -50.
    # Voltando do pivô: x = 50 + 50 = 100, y = -50 + 50 = 0.
    np.testing.assert_array_almost_equal(res, [100, 0, 1])


def test_calculate_new_bbox_apenas_translacao():
    m = mat_translation(10, 20)
    size = (100, 50)
    bbox = calculate_new_bbox(m, size)
    # x, y, w, h
    # Corners: (0,0)->(10,20), (99,0)->(109,20), (99,49)->(109,69), (0,49)->(10,69)
    # MinX: 10, MaxX: 109. W: 109-10+1 = 100.
    # MinY: 20, MaxY: 69. H: 69-20+1 = 50.
    np.testing.assert_allclose(bbox, (10.0, 20.0, 100.0, 50.0), atol=1e-5)


def test_calculate_new_bbox_rotacao_90_graus():
    # Rotacao de 90 graus na origem
    rot_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    size = (100, 50)
    bbox = calculate_new_bbox(rot_90, size)

    # Nova lógica (espaço contínuo):
    # (0, 0)    -> (0, 0)
    # (100, 0)  -> (0, 100)
    # (100, 50) -> (-50, 100)
    # (0, 50)   -> (-50, 0)

    # Min: (-50, 0), Max: (0, 100)
    # Largura (W): Max X - Min X = 0 - (-50) = 50
    # Altura (H):  Max Y - Min Y = 100 - 0 = 100
    np.testing.assert_allclose(bbox, (-50.0, 0.0, 50.0, 100.0), atol=1e-5)


def test_calculate_new_bbox_com_escala():
    # Escala de 2x
    scale = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    size = (100, 50)
    bbox = calculate_new_bbox(scale, size)
    np.testing.assert_allclose(bbox, (0.0, 0.0, 200.0, 100.0), atol=1e-5)


def test_calculate_new_bbox_com_flip_horizontal():
    # Escala de -1 no X (Flip Horizontal)
    flip_h = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    size = (100, 50)
    # Com pivô na origem (0,0), o (99, 0) vai para (-100, 0)
    bbox = calculate_new_bbox(flip_h, size)
    # MinX: -100, MaxX: 0. W: 100.
    np.testing.assert_allclose(bbox, (-100.0, 0.0, 100.0, 50.0), atol=1e-5)


def test_calculate_new_bbox_escala_zero():
    # Escala zero deve resultar em largura e altura 1 (ponto singular)
    scale_zero = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    size = (100, 50)
    bbox = calculate_new_bbox(scale_zero, size)
    np.testing.assert_allclose(bbox, (0.0, 0.0, 1.0, 1.0), atol=1e-5)


def test_calculate_new_bbox_escala_negativa_extrema():
    # Escala -2.5x em ambos os eixos
    scale_neg = np.array([
        [-2.5, 0, 0],
        [0, -2.5, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    size = (100, 50)

    # Canto (0, 0)     -> (0, 0)
    # Canto (100, 0)   -> (-250, 0)
    # Canto (100, 50)  -> (-250, -125)
    # Canto (0, 50)    -> (0, -125)

    # Min X: floor(-250) = -250
    # Min Y: floor(-125) = -125
    # Max X: ceil(0)     = 0
    # Max Y: ceil(0)     = 0

    # Largura (W): 0 - (-250) = 250
    # Altura (H):  0 - (-125) = 125
    bbox = calculate_new_bbox(scale_neg, size)
    np.testing.assert_allclose(bbox, (-250.0, -125.0, 250.0, 125.0), atol=1e-5)


def test_create_pivot_transform_pivo_excentrico():
    # Rotação de 90° em torno do canto inferior direito (100, 50)
    rot_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # px_rel=1.0 (100px), py_rel=1.0 (50px)
    m = create_pivot_transform(rot_90, 100, 50, 1.0, 1.0)

    # O próprio pivô não deve se mover
    pivo_pt = np.array([100, 50, 1], dtype=np.float32)
    res_pivo = m @ pivo_pt
    np.testing.assert_array_almost_equal(res_pivo, [100, 50, 1])

    # O topo-esquerdo (0,0) deve girar em torno de (100,50)
    # Vetor Pivo->TL = (-100, -50)
    # Rot 90°: x' = -(-50) = 50, y' = -100
    # Pos-Pivo: x = 100+50 = 150, y = 50-100 = -50
    res_tl = m @ [0, 0, 1]
    np.testing.assert_array_almost_equal(res_tl, [150, -50, 1])


def test_composicao_ordem_matrizes():
    # Testar se T @ R @ S se comporta como esperado
    # 1. Escala 2x -> 2. Rot 90° -> 3. Translação (10, 10)
    S = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float32)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    T = mat_translation(10, 10)

    M = T @ R @ S

    # Ponto (10, 0) local:
    # Scale 2x -> (20, 0)
    # Rot 90 -> (0, 20)
    # Trans (10, 10) -> (10, 30)
    pt = np.array([10, 0, 1], dtype=np.float32)
    res = M @ pt
    np.testing.assert_array_almost_equal(res, [10, 30, 1])


def test_estabilidade_ciclo_rotacao():
    # Girar 90 graus 4 vezes deve resultar na identidade (ou muito próximo)
    rot_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    m = rot_90 @ rot_90 @ rot_90 @ rot_90
    np.testing.assert_array_almost_equal(m, np.eye(3))


def test_transform_inicializacao():
    t = Transform((100, 100))
    # Identidade por padrão
    np.testing.assert_array_equal(t.matrix, np.eye(3, dtype=np.float32))
    assert t.size == (100, 100)


def test_transform_translacao_simples():
    t = Transform((100, 100))
    t.translate(10, 20)

    expected = np.array([
        [1, 0, 10],
        [0, 1, 20],
        [0, 0, 1]
    ], dtype=np.float32)

    np.testing.assert_array_equal(t.matrix, expected)


def test_transform_rotacao_no_centro():
    t = Transform((100, 100))
    # Rotaciona 90 graus no centro (50, 50)
    t.rotation(90, 0.5, 0.5)

    # Ponto (0,0) deve ir para (100, 0)
    pt = np.array([0, 0, 1], dtype=np.float32)
    res = t.matrix @ pt

    np.testing.assert_array_almost_equal(res, [100, 0, 1])


def test_transform_escala_no_centro():
    t = Transform((100, 100))
    # Escala 2x no centro (50, 50)
    t.scale(2, 2, 0.5, 0.5)

    # O centro (50, 50) deve permanecer imóvel
    pt_centro = np.array([50, 50, 1], dtype=np.float32)
    res_centro = t.matrix @ pt_centro
    np.testing.assert_array_almost_equal(res_centro, [50, 50, 1])

    # O ponto (0, 0) deve se afastar do centro
    # Vetor Pivo->TL = (-50, -50)
    # Escala 2x: (-100, -100)
    # Pos-Pivo: (50-100, 50-100) = (-50, -50)
    pt_tl = np.array([0, 0, 1], dtype=np.float32)
    res_tl = t.matrix @ pt_tl
    np.testing.assert_array_almost_equal(res_tl, [-50, -50, 1])


def test_transform_acumulacao_e_fluidez():
    # Testa a ordem de acumulação: M_new @ M_old
    # 1. Translate (10, 0)
    # 2. Scale 2x no Centro
    t = Transform((100, 100))
    t.translate(10, 0).scale(2, 2, 0.5, 0.5)

    # Como a classe faz M_new @ M_old:
    # O resultado deve ser Escala @ Translacao
    # Ponto (0, 0) -> Translacao -> (10, 0)
    # (10, 0) -> Escala 2x (pivo 50,50)
    # Vetor Pivo->Pt = (10-50, 0-50) = (-40, -50)
    # Escala 2x = (-80, -100)
    # Volta Pivo = (50-80, 50-100) = (-30, -50)

    pt = np.array([0, 0, 1], dtype=np.float32)
    res = t.matrix @ pt
    np.testing.assert_array_almost_equal(res, [-30, -50, 1])


def test_transform_ordem_inversa():
    # 1. Scale 2x no Centro
    # 2. Translate (10, 0)
    t = Transform((100, 100))
    t.scale(2, 2, 0.5, 0.5).translate(10, 0)

    # Resultado deve ser Translacao @ Escala
    # Ponto (0, 0) -> Escala 2x (pivo 50,50) -> (-50, -50)
    # (-50, -50) -> Translacao (10, 0) -> (-40, -50)

    pt = np.array([0, 0, 1], dtype=np.float32)
    res = t.matrix @ pt
    np.testing.assert_array_almost_equal(res, [-40, -50, 1])
