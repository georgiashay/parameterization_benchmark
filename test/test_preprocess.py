from ..utilities.preprocess import preprocess 

import os
import math
import pytest
import numpy as np

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

def test_preprocess_square():
    fpath = os.path.join(fixture_dir, "square.obj")
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    assert mesh_areas == pytest.approx(np.array([[0.5, 0.5]]))
    assert uv_areas == pytest.approx(np.array([[0.5, 0.5]]))
    assert v_i == pytest.approx(v)
    assert uv_i == pytest.approx(uv)
    assert v == pytest.approx(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))
    assert uv == pytest.approx(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    
def test_preprocess_square_scaled():
    fpath = os.path.join(fixture_dir, "square_scaled.obj")
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    assert mesh_areas == pytest.approx(np.array([[0.5, 0.5]]))
    assert uv_areas == pytest.approx(np.array([[0.5, 0.5]]))
    assert v_i == pytest.approx(np.array([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]]))
    assert uv_i == pytest.approx(np.array([[0, 0], [2, 0], [2, 2], [0, 2]]))
    assert v == pytest.approx(np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]]))
    assert uv == pytest.approx(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    
def test_preprocess_square_stretched():
    fpath = os.path.join(fixture_dir, "square_stretched.obj")
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    assert mesh_areas == pytest.approx(np.array([[0.5, 0.5]]))
    assert uv_areas == pytest.approx(np.array([[0.5, 0.5]]))
    assert v_i == pytest.approx(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))
    assert uv_i == pytest.approx(np.array([[0, 0], [2, 0], [2, 1], [0, 1]]))
    assert v == pytest.approx(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))
    assert uv == pytest.approx(np.array([[0, 0], [math.sqrt(2), 0], 
                                         [math.sqrt(2), 1/math.sqrt(2)], [0, 1/math.sqrt(2)]]))
    
def test_preprocess_grid_shift():
    fpath = os.path.join(fixture_dir, "grid_shift.obj")
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    assert mesh_areas == pytest.approx(np.array([[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]]))
    assert uv_areas == pytest.approx(np.array([[0.1875, 0.1875, 0.0625, 0.0625, 0.1875, 0.1875, 0.0625, 0.0625]]))
    assert v_i == pytest.approx(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], 
                                          [2, 1, 0], [0, 2, 0], [1, 2, 0], [2, 2, 0]]))
    assert uv_i == pytest.approx(np.array([[0, 0], [3, 0], [4, 0], [0, 1], [3, 1], 
                                           [4, 1], [0, 2], [3, 2], [4, 2]]))
    assert v == pytest.approx(np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0],
                                        [1, 0.5, 0], [0, 1, 0], [0.5, 1, 0], [1, 1, 0]]))
    assert uv == pytest.approx(np.array([[0, 0], [3/math.sqrt(8), 0], [4/math.sqrt(8), 0], 
                                         [0, 1/math.sqrt(8)], [3/math.sqrt(8), 1/math.sqrt(8)],
                                         [4/math.sqrt(8), 1/math.sqrt(8)], [0, 2/math.sqrt(8)], 
                                         [3/math.sqrt(8), 2/math.sqrt(8)], 
                                         [4/math.sqrt(8), 2/math.sqrt(8)]]))
    
def test_preprocess_no_uv_area():
    fpath = os.path.join(fixture_dir, "flat_triangle.obj")
    
    with pytest.raises(ValueError) as e:
        preprocess(fpath)
        
    assert str(e.value) == "No UV area"
    