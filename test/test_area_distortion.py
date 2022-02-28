from ..utilities.area_distortion import get_area_distortion
from ..utilities.preprocess import preprocess 

import os

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

def test_area_distortion_square():
    fpath = os.path.join(fixture_dir, "square.obj")
    v, uv, f, ftc, mesh_areas, uv_areas = preprocess(fpath)
    
    area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
    
    assert total_area_distortion == 0
    
def test_area_distortion_scaled_square():
    fpath = os.path.join(fixture_dir, "square_scaled.obj")
    v, uv, f, ftc, mesh_areas, uv_areas = preprocess(fpath)
    
    area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
    
    assert total_area_distortion == 0
    
def test_area_distortion_stretched_square():
    fpath = os.path.join(fixture_dir, "square_stretched.obj")
    v, uv, f, ftc, mesh_areas, uv_areas = preprocess(fpath)
    
    area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
    
    assert total_area_distortion == 0

def test_area_distortion_grid_shift():
    fpath = os.path.join(fixture_dir, "grid_shift.obj")
    v, uv, f, ftc, mesh_areas, uv_areas = preprocess(fpath)
    
    area_distortions, max_area_distortion, total_area_distortion = get_area_distortion(uv_areas, mesh_areas)
    
    assert total_area_distortion == 0.5