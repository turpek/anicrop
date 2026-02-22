import numpy as np
from anicrop.transform import (
    mat_translation,
    create_pivot_transform,
    calculate_new_bbox,
    mat_position
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

    # (0,0) -> (0,0)
    # (99, 0) -> (0, 99)
    # (99, 49) -> (-49, 99)
    # (0, 49) -> (-49, 0)
    # Min: (-49, 0), Max: (0, 99)
    # W: 0 - (-49) + 1 = 50, H: 99 - 0 + 1 = 100
    np.testing.assert_allclose(bbox, (-49.0, 0.0, 50.0, 100.0), atol=1e-5)


def test_calculate_new_bbox_com_escala():
    # Escala de 2x
    scale = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    size = (100, 50)
    bbox = calculate_new_bbox(scale, size)
    # (0,0)->(0,0), (99,49)->(198,98)
    # W: 198-0+1 = 199, H: 98-0+1 = 99
    np.testing.assert_allclose(bbox, (0.0, 0.0, 199.0, 99.0), atol=1e-5)


def test_calculate_new_bbox_com_flip_horizontal():
    # Escala de -1 no X (Flip Horizontal)
    flip_h = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    size = (100, 50)
    # Com pivô na origem (0,0), o (99, 0) vai para (-99, 0)
    bbox = calculate_new_bbox(flip_h, size)
    # MinX: -99, MaxX: 0. W: 100.
    np.testing.assert_allclose(bbox, (-99.0, 0.0, 100.0, 50.0), atol=1e-5)


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
    # (0,0) -> (0,0)
    # (99, 49) -> (-247.5, -122.5) -> Round -> (-248, -122)
    # Min: (-248, -122), Max: (0, 0)
    # W: 0 - (-248) + 1 = 249
    # H: 0 - (-122) + 1 = 123
    bbox = calculate_new_bbox(scale_neg, size)
    np.testing.assert_allclose(bbox, (-248.0, -122.0, 249.0, 123.0), atol=1e-5)


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
