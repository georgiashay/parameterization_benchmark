from ..utilities.singular_values import get_singular_values
from ..utilities.jacobian import get_jacobian
from ..utilities.preprocess import preprocess

import os
import math
import pytest
import numpy as np

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

def test_singular_values_square_scaled():
    fpath = os.path.join(fixture_dir, "square_scaled.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    J, _ = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    
    J_i, _ = get_jacobian(v_i, f, uv_i, ftc)
    singular_values_i, _, _ = get_singular_values(J_i)

    assert singular_values_i == pytest.approx(np.array([[0.2, 0.2], [0.2, 0.2]]))
    assert singular_values == pytest.approx(np.array([[1, 1], [1, 1]]))
    
def test_singular_values_square_stretched():
    fpath = os.path.join(fixture_dir, "square_stretched.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    J, _ = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    
    J_i, _ = get_jacobian(v_i, f, uv_i, ftc)
    singular_values_i, _, _ = get_singular_values(J_i)

    assert singular_values_i == pytest.approx(np.array([[2, 1], [2, 1]]))
    assert singular_values == pytest.approx(np.array([[math.sqrt(2), 1/math.sqrt(2)], [math.sqrt(2), 1/math.sqrt(2)]]))
    
def test_singular_values_grid_shift():
    fpath = os.path.join(fixture_dir, "grid_shift.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    J, _ = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    
    J_i, _ = get_jacobian(v_i, f, uv_i, ftc)
    singular_values_i, _, _ = get_singular_values(J_i)

    assert singular_values_i == pytest.approx(np.array([[3, 1], [3, 1], [1, 1], [1, 1], [3, 1], [3, 1], [1, 1], [1, 1]]))
    assert singular_values == pytest.approx(np.array([[6/math.sqrt(8), 2/math.sqrt(8)], [6/math.sqrt(8), 2/math.sqrt(8)], 
                                                      [2/math.sqrt(8), 2/math.sqrt(8)], [2/math.sqrt(8), 2/math.sqrt(8)],
                                                      [6/math.sqrt(8), 2/math.sqrt(8)], [6/math.sqrt(8), 2/math.sqrt(8)], 
                                                      [2/math.sqrt(8), 2/math.sqrt(8)], [2/math.sqrt(8), 2/math.sqrt(8)]]))
    
def test_singular_values_triangle():
    fpath = os.path.join(fixture_dir, "triangle.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    J, _ = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    
    J_i, _ = get_jacobian(v_i, f, uv_i, ftc)
    singular_values_i, _, _ = get_singular_values(J_i)
    
    assert singular_values_i == pytest.approx(np.array([[1.5, 1]]))
    assert singular_values == pytest.approx(np.array([[math.sqrt(3/2), 2/math.sqrt(6)]]))

    
def test_singular_values_with_flat():
    fpath = os.path.join(fixture_dir, "with_flat.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    J, _ = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    
    J_i, _ = get_jacobian(v_i, f, uv_i, ftc)
    singular_values_i, _, _ = get_singular_values(J_i)
    
    assert singular_values_i == pytest.approx(np.array([[0.5, 0], [0.5, 0.5]]))
    assert singular_values == pytest.approx(np.array([[np.sqrt(2), 0], [np.sqrt(2), np.sqrt(2)]]))
