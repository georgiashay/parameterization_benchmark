from ..utilities.jacobian import get_jacobian
from ..utilities.preprocess import preprocess

import os
import math
import pytest
import numpy as np

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

# Note: the following are regression tests

def test_regression_jacobian_square_scaled():
    fpath = os.path.join(fixture_dir, "square_scaled.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    J = get_jacobian(v, f, uv, ftc)
    J_i = get_jacobian(v_i, f, uv_i, ftc)

    assert J == pytest.approx(np.array([[[1, 0], [0, 1]], 
                                        [[math.sqrt(2)/2, -math.sqrt(2)/2], [math.sqrt(2)/2, math.sqrt(2)/2]]]))
    assert J_i == pytest.approx(np.array([[[0.2, 0], [0, 0.2]], 
                                          [[math.sqrt(2)/10, -math.sqrt(2)/10], [math.sqrt(2)/10, math.sqrt(2)/10]]]))

    
def test_regression_jacobian_square_stretched():
    fpath = os.path.join(fixture_dir, "square_stretched.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    J = get_jacobian(v, f, uv, ftc)
    J_i = get_jacobian(v_i, f, uv_i, ftc)
    
    assert J == pytest.approx(np.array([[[math.sqrt(2), 0], [0, math.sqrt(2)/2]], [[1, -1], [0.5, 0.5]]]))
    assert J_i == pytest.approx(np.array([[[2, 0], [0, 1]], [[math.sqrt(2), -math.sqrt(2)], [math.sqrt(2)/2, math.sqrt(2)/2]]]))

    
def test_regression_jacobian_grid_shift():
    fpath = os.path.join(fixture_dir, "grid_shift.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    J = get_jacobian(v, f, uv, ftc)    
    J_i = get_jacobian(v_i, f, uv_i, ftc)
    
    assert J == pytest.approx(np.array([[[1.5*math.sqrt(2), 0], [0, math.sqrt(2)/2]],
                                         [[1.5, -1.5], [0.5, 0.5]],
                                         [[math.sqrt(2)/2, 0], [0, math.sqrt(2)/2]],
                                         [[0.5, -0.5], [0.5, 0.5]],
                                         [[1.5*math.sqrt(2), 0], [0, math.sqrt(2)/2]],
                                         [[1.5, -1.5], [0.5, 0.5]],
                                         [[math.sqrt(2)/2, 0], [0, math.sqrt(2)/2]],
                                         [[0.5, -0.5], [0.5, 0.5]]]))
    assert J_i == pytest.approx(np.array([[[3, 0], [0, 1]],
                                          [[1.5*math.sqrt(2), -1.5*math.sqrt(2)], [math.sqrt(2)/2, math.sqrt(2)/2]],
                                          [[1, 0], [0, 1]],
                                          [[math.sqrt(2)/2, -math.sqrt(2)/2], [math.sqrt(2)/2, math.sqrt(2)/2]],
                                          [[3, 0], [0, 1]],
                                          [[1.5*math.sqrt(2), -1.5*math.sqrt(2)], [math.sqrt(2)/2, math.sqrt(2)/2]],
                                          [[1, 0], [0, 1]],
                                          [[math.sqrt(2)/2, -math.sqrt(2)/2], [math.sqrt(2)/2, math.sqrt(2)/2]]]))
    
        
def test_regression_jacobian_triangle():
    fpath = os.path.join(fixture_dir, "triangle.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    J = get_jacobian(v, f, uv, ftc)    
    J_i = get_jacobian(v_i, f, uv_i, ftc)
    
    assert J == pytest.approx(np.array([[[0.5*math.sqrt(6), 0], [0, math.sqrt(6)/3]]]))
    assert J_i == pytest.approx(np.array([[[1.5, 0], [0, 1]]]))
    
     
def test_regression_jacobian_with_flat():
    fpath = os.path.join(fixture_dir, "with_flat.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    J = get_jacobian(v, f, uv, ftc)    
    J_i = get_jacobian(v_i, f, uv_i, ftc)
    
    assert J == pytest.approx(np.array([[[1, 1], [0, 0]], [[1, 1], [-1, 1]]]))
    assert J_i == pytest.approx(np.array([[[math.sqrt(2)/4, math.sqrt(2)/4], [0, 0]], 
                                          [[math.sqrt(2)/4, math.sqrt(2)/4], [-math.sqrt(2)/4, math.sqrt(2)/4]]]))

