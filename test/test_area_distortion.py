from ..utilities.area_distortion import get_area_distortion
from ..utilities.preprocess import preprocess 

import os
import math
import pytest

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

def test_area_distortion_square():
    fpath = os.path.join(fixture_dir, "square.obj")
    _, _, _, _, _, _, mesh_areas, uv_areas = preprocess(fpath)
    
    area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
    
    assert total_area_distortion == pytest.approx(0)
    assert max_area_distortion == pytest.approx(0)
    
def test_area_distortion_scaled_square():
    fpath = os.path.join(fixture_dir, "square_scaled.obj")
    _, _, _, _, _, _, mesh_areas, uv_areas = preprocess(fpath)
    
    area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
    
    assert total_area_distortion == pytest.approx(0)
    assert max_area_distortion == pytest.approx(0)
    
def test_area_distortion_stretched_square():
    fpath = os.path.join(fixture_dir, "square_stretched.obj")
    _, _, _, _, _, _, mesh_areas, uv_areas = preprocess(fpath)
    
    area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
    
    assert total_area_distortion == pytest.approx(0)
    assert max_area_distortion == pytest.approx(0)

def test_area_distortion_grid_shift():
    fpath = os.path.join(fixture_dir, "grid_shift.obj")
    _, _, _, _, _, _, mesh_areas, uv_areas = preprocess(fpath)
    
    area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
    
    assert total_area_distortion == pytest.approx(0.5)
    assert max_area_distortion == pytest.approx(0.5)
    
def test_area_distortion_cone_disk():
    fpath = os.path.join(fixture_dir, "cone_disk.obj")
    _, _, _, _, _, _, mesh_areas, uv_areas = preprocess(fpath)
    
    area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
    
    assert total_area_distortion == pytest.approx(0, abs=1e-6)
    assert max_area_distortion == pytest.approx(0)
    
def test_area_distortion_cone_pizza():
    fpath = os.path.join(fixture_dir, "cone_pizza.obj")
    _, _, _, _, _, _, mesh_areas, uv_areas = preprocess(fpath)
    
    area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
    
    assert total_area_distortion == pytest.approx(0, abs=1e-4)
    assert max_area_distortion == pytest.approx(0, abs=1e-6)
    
def test_area_distortion_cone_tri():
    fpath = os.path.join(fixture_dir, "cone_tri.obj")
    _, _, _, _, _, _, mesh_areas, uv_areas = preprocess(fpath)
    
    area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
    
    area_factor1 = math.cos(math.pi/8)
    area_factor2 = math.cos(3*math.pi/8)
    sum_area_factors = area_factor1 + area_factor2

    expected = abs(0.5 - area_factor1/sum_area_factors) + abs(0.5 - area_factor2/sum_area_factors)
    
    assert total_area_distortion == pytest.approx(expected)
    assert max_area_distortion == pytest.approx((area_factor2/sum_area_factors)/0.5 + 0.5/(area_factor2/sum_area_factors) - 2)
                                                 
    