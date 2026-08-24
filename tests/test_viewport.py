import numpy as np
from anicrop.viewport import Viewport
from anicrop.spatial import Region
from anicrop.type import Scale
from anicrop.transform import mat_translation, mat_pivot


def test_viewport_initialization():
    """Grupo 1: Scenario 1 - Initial State."""
    size = (800, 600)
    fit_scale = 0.25
    viewport = Viewport(size=size, fit_scale=fit_scale)

    assert viewport.size == size
    assert viewport.top_left == (0, 0)
    assert viewport.scale == Scale(1.0, 1.0)
    # scale_factor = 1.0 * 0.25 = 0.25
    assert viewport.scale_factor == 0.25


def test_viewport_scale_update():
    """Grupo 1: Scenario 2 - Scale Update (Zoom)."""
    viewport = Viewport(size=(800, 600), fit_scale=0.25)
    viewport.scale = Scale(2.0, 2.0)

    # scale_factor = 2.0 * 0.25 = 0.5
    assert viewport.scale_factor == 0.5


def test_viewport_region_update():
    """Grupo 1: Scenario 3 - Region Update (Pan)."""
    viewport = Viewport(size=(800, 600), fit_scale=0.25)
    viewport.region += (100, 50)

    assert viewport.top_left == (100, 50)


def test_viewport_fit_matrix():
    """Grupo 2: Scenario 4 - Fit Matrix."""
    fit_scale = 0.25
    size = (800, 600)
    viewport = Viewport(size=size, fit_scale=fit_scale)

    # Layer with same size as viewport
    layer_size = size

    # scaled_w = 800 * 0.25 = 200
    # offset_x = (800 - 200) / 2 = 300
    # scaled_h = 600 * 0.25 = 150
    # offset_y = (600 - 150) / 2 = 225

    expected_fit = np.array([
        [0.25, 0.0, 300.0],
        [0.0, 0.25, 225.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    assert np.allclose(viewport.fit_matrix(layer_size), expected_fit)


def test_viewport_roi_matrix_identity():
    """Grupo 2: Scenario 5 - ROI Matrix (No Pan, No Zoom)."""
    viewport = Viewport(size=(800, 600), fit_scale=0.25)

    # Identity 3x3
    assert np.allclose(viewport.roi_matrix, np.eye(3))


def test_viewport_roi_matrix_pan():
    """Grupo 2: Scenario 6 - ROI Matrix (Pan only)."""
    viewport = Viewport(size=(800, 600), fit_scale=0.25)
    viewport.region += (100, 50)

    expected_pan = mat_translation(-100, -50)
    assert np.allclose(viewport.roi_matrix, expected_pan)


def test_viewport_roi_matrix_zoom_pivot():
    """Grupo 2: Scenario 7 - ROI Matrix (Zoom with Pivot)."""
    size = (800, 600)
    viewport = Viewport(size=size, fit_scale=0.25)
    viewport.scale = Scale(2.0, 2.0)

    # Pivot is at center (400, 300)
    # M = T(400, 300) @ S(2.0) @ T(-400, -300)
    expected_m = mat_pivot(viewport.scale, size)
    assert np.allclose(viewport.roi_matrix, expected_m)

    # Verify the constant terms (X and Y negative compensations)
    # M(0,0) = T(400,300) S(2.0) (-400, -300) = T(400,300) (-800, -600) = (-400, -300)
    assert viewport.roi_matrix[0, 2] == -400.0
    assert viewport.roi_matrix[1, 2] == -300.0


def test_viewport_roi_1to1():
    """Grupo 3: Scenario 8 - Perfect Alignment (1:1)."""
    viewport = Viewport(size=(800, 600), fit_scale=1.0)
    layer_region = Region.from_size(800, 600)  # (0,0, 800, 600)

    roi = viewport.roi(layer_region)
    assert roi.top_left == (0, 0)
    assert roi.size == (800, 600)


def test_viewport_roi_panned_camera():
    """Grupo 3: Scenario 9 - Panned Camera (Negative Local Coords)."""
    # Camera at (0,0), Layer at (100, 100)
    viewport = Viewport(size=(800, 600), fit_scale=1.0)
    layer_region = Region.from_size(800, 600) + (100, 100)

    roi = viewport.roi(layer_region)
    assert roi.top_left == (-100, -100)
    assert roi.size == (800, 600)


def test_viewport_roi_zoom_in():
    """Grupo 3: Scenario 10 - Zoom In (Crop)."""
    # Viewport 800x600, Zoom 2.0
    viewport = Viewport(size=(800, 600), fit_scale=1.0)
    viewport.scale = Scale(2.0, 2.0)
    layer_region = Region.from_size(2000, 2000)  # (0,0)

    roi = viewport.roi(layer_region)
    assert roi.top_left == (200, 150)
    assert roi.size == (400, 300)


def test_viewport_roi_stress():
    """Grupo 3: Scenario 11 - Stress Scenario (Pan + Zoom Out + Offset)."""
    # Viewport (800, 600), Zoom 0.5, Pan (100, 50)
    # Note: fit_scale must be 1.0 to match "Zoom 0.5 sees 1600x1200"
    viewport = Viewport(size=(800, 600), fit_scale=1.0)
    viewport.scale = Scale(0.5, 0.5)
    viewport.region += (100, 50)

    # Layer at (200, 100)
    layer_region = Region.from_size(2000, 2000) + (200, 100)

    roi = viewport.roi(layer_region)

    # Math from prompt:
    # Zoom 0.5 -> 1600x1200 world view.
    # Pivot at (400, 300) -> Viewport world range (-400, -300) to (1200, 900)
    # Pan (100, 50) moves viewport to (-300, -250) to (1300, 950)?
    # Wait, my analysis showed T(pan) @ Pivot would give different results.
    # Let's see what the actual implementation does.
    # User expects: start=(-500, -350), length=(1600, 1200)

    assert roi.size == (1600, 1200)
    assert roi.top_left == (-500, -350)
