import pytest
import numpy as np
from anicrop.image import Image, ImageFormat
from anicrop.layer import Layer
from anicrop.spatial import Region, Span


def make_canvas(w=100, h=100, color=(255, 0, 0, 255)):
    return Image.new((h, w), ImageFormat.RGBA, color=color)


@pytest.fixture
def red_layer():
    return Layer(make_canvas())


def test_render_transform_translation_simples(red_layer):
    red_layer.transform.translate(10, 20)
    region = red_layer.global_region
    assert region.top_left == (10, 20)
    assert region.size == (100, 100)


def test_render_transform_rotation_90_centro(red_layer):
    # Rotacionar 90 graus no centro mantém o BBox se a imagem for quadrada
    red_layer.transform.rotate(90, 0.5, 0.5)
    region = red_layer.global_region
    assert region.top_left == (0, 0)
    assert region.size == (100, 100)


def test_render_transform_scale_2x_centro(red_layer):
    # Escala 2x no centro (50, 50) de uma imagem 100x100
    # O ponto (0,0) vai para -50, o ponto (100,100) vai para 150
    # Novo BBox: (-50, -50, 200, 200)
    red_layer.transform.scale(2, 2, 0.5, 0.5)
    region = red_layer.global_region
    assert region.top_left == (-50, -50)
    assert region.size == (200, 200)


def test_render_transform_chaining_complexo(red_layer):
    # Testa: Translação(50,50) -> Escala(2x no Centro) -> Rotação(90 no Centro)
    # Com a lógica de Translação Final Acumulada:
    # 1. Distorção local (Escala 2x + Rotação 90 no centro 50,50):
    #    BBox local expande para (-50, -50, 200, 200)
    # 2. Translação global final (+50, +50):
    #    Top-Left: (-50+50, -50+50) = (0, 0)
    #    Size: (200, 200)
    red_layer.transform.translate(50, 50).scale(
        2, 2, 0.5, 0.5).rotate(90, 0.5, 0.5)

    region = red_layer.global_region
    assert region.top_left == (0, 0)
    assert region.size == (200, 200)


def test_render_transform_pivo_canto(red_layer):
    # Rotação 90 no canto (0,0)
    # (0,0) -> (0,0)
    # (100,0) -> (0, 100)
    # (100,100) -> (-100, 100)
    # (0,100) -> (-100, 0)
    # BBox: (-100, 0, 100, 100)
    red_layer.transform.rotate(90, 0, 0)
    region = red_layer.global_region
    assert region.top_left == (-100, 0)
    assert region.size == (100, 100)
