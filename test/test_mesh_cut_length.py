from ..utilities.mesh_cut_length import get_mesh_cut_length
from ..utilities.v_uv_map import get_v_uv_map
from ..utilities.preprocess import preprocess

import os
import math
import pytest
import numpy as np

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")


def test_mesh_cut_length_square():
    fpath = os.path.join(fixture_dir, "square.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f, ftc)
    mesh_cut_length = get_mesh_cut_length(uv_to_v_arr, v, f, ftc)
    
    assert mesh_cut_length == pytest.approx(0)
    
def test_mesh_cut_length_square_scaled():
    fpath = os.path.join(fixture_dir, "square_scaled.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f, ftc)
    mesh_cut_length = get_mesh_cut_length(uv_to_v_arr, v, f, ftc)
    
    assert mesh_cut_length == pytest.approx(0)
    
def test_meh_cut_length_grid_shift():
    fpath = os.path.join(fixture_dir, "grid_shift.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f, ftc)
    mesh_cut_length = get_mesh_cut_length(uv_to_v_arr, v, f, ftc)
    
    assert mesh_cut_length == pytest.approx(0)
        
def test_mesh_cut_length_cone_disk():
    fpath = os.path.join(fixture_dir, "cone_disk.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f, ftc)
    mesh_cut_length = get_mesh_cut_length(uv_to_v_arr, v, f, ftc)
        
    assert mesh_cut_length == pytest.approx(0)

def test_mesh_cut_length_cone_pizza():
    fpath = os.path.join(fixture_dir, "cone_pizza.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f, ftc)
    mesh_cut_length = get_mesh_cut_length(uv_to_v_arr, v, f, ftc)

    r = math.sqrt(4 + 2 * math.sqrt(2))/2
    h = math.sqrt(2*(r**2) - 1/4)
    area = 1/2 * h * 8
    
    a = 1/math.sqrt(area)
    r = a * math.sqrt(4 + 2 * math.sqrt(2))/2
    h =  math.sqrt(2*(r**2) - (a/2)**2)
    d = math.sqrt((a/2)**2 + h**2)
    
    assert mesh_cut_length == pytest.approx(4*d)

def test_mesh_cut_length_cube():
    fpath = os.path.join(fixture_dir, "cube.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f, ftc)
    mesh_cut_length = get_mesh_cut_length(uv_to_v_arr, v, f, ftc)
    
    assert mesh_cut_length == pytest.approx(7/math.sqrt(6))

def test_mesh_cut_length_cone_tri():
    fpath = os.path.join(fixture_dir, "cone_tri.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f, ftc)
    mesh_cut_length = get_mesh_cut_length(uv_to_v_arr, v, f, ftc)
  
    assert mesh_cut_length == pytest.approx(0)
