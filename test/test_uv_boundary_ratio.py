from ..utilities.uv_boundary_ratio import get_uv_boundary_length
from ..utilities.preprocess import preprocess

import os
import math
import pytest

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

def test_uv_boundary_ratio_square():
    fpath = os.path.join(fixture_dir, "square.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    uv_boundary_length = get_uv_boundary_length(uv, ftc)
    
    assert uv_boundary_length == pytest.approx(4)
    
def test_uv_boundary_ratio_square_scaled():
    fpath = os.path.join(fixture_dir, "square_scaled.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    uv_boundary_length = get_uv_boundary_length(uv, ftc)
    
    assert uv_boundary_length == pytest.approx(4)
    
def test_uv_boundary_ratio_grid_shift():
    fpath = os.path.join(fixture_dir, "grid_shift.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    uv_boundary_length = get_uv_boundary_length(uv, ftc)
    
    assert uv_boundary_length == pytest.approx(12/math.sqrt(8))
    
def test_uv_boundary_ratio_cone_disk():
    fpath = os.path.join(fixture_dir, "cone_disk.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    uv_boundary_length = get_uv_boundary_length(uv, ftc)
    
    s = 1/math.sqrt(2 * (1 + math.sqrt(2)))
    
    assert uv_boundary_length == pytest.approx(8*s)

def test_uv_boundary_ratio_cone_pizza():
    fpath = os.path.join(fixture_dir, "cone_pizza.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    uv_boundary_length = get_uv_boundary_length(uv, ftc)
    
    s = 1/math.sqrt(2 * (1 + math.sqrt(2)))
    d = (s/2)/math.sin(math.pi/8)
    
    assert uv_boundary_length == pytest.approx(8*s + 8*d)

def test_uv_boundary_ratio_cube():
    fpath = os.path.join(fixture_dir, "cube.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    uv_boundary_length = get_uv_boundary_length(uv, ftc)
    
    assert uv_boundary_length == pytest.approx(14/math.sqrt(6))

def test_uv_boundary_ratio_cone_tri():
    fpath = os.path.join(fixture_dir, "cone_tri.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    uv_boundary_length = get_uv_boundary_length(uv, ftc)
    
    b = 2
    a = math.sqrt(2)
    
    assert uv_boundary_length == pytest.approx(b + 2*a)