from ..utilities.artist_match_area import get_artist_area_match
from ..utilities.preprocess import preprocess 

import os
import math
import pytest

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

def test_artist_area_match_square():
    ofpath = os.path.join(fixture_dir, "square_stretched.obj")
    fpath = os.path.join(fixture_dir, "square_scaled.obj")
    
    _, _, _, _, _, _, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    _, _, _, _, _, _, mesh_areas, uv_areas = preprocess(fpath)
    
    _, _, artist_area_match = get_artist_area_match(mesh_areas, uv_areas_o, uv_areas)
    
    assert artist_area_match == pytest.approx(0)
    
def test_artist_area_match_symmetric_shifts():
    ofpath = os.path.join(fixture_dir, "right_shift.obj")
    fpath = os.path.join(fixture_dir, "left_shift.obj")
    
    _, _, _, _, _, _, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    _, _, _, _, _, _, mesh_areas, uv_areas = preprocess(fpath)
    
    _, _, artist_area_match = get_artist_area_match(mesh_areas, uv_areas_o, uv_areas)
    
    assert artist_area_match == pytest.approx(0)
    
def test_artist_area_match_diff_shifts():
    ofpath = os.path.join(fixture_dir, "right_shift.obj")
    fpath = os.path.join(fixture_dir, "big_left_shift.obj")
    
    _, _, _, _, _, _, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    _, _, _, _, _, _, mesh_areas, uv_areas = preprocess(fpath)
    
    _, _, artist_area_match = get_artist_area_match(mesh_areas, uv_areas_o, uv_areas)
    
    assert artist_area_match == pytest.approx(6/10)
    
def test_artist_area_match_other_diff_shifts():
    ofpath = os.path.join(fixture_dir, "quarters1.obj")
    fpath = os.path.join(fixture_dir, "quarters2.obj")
    
    _, _, _, _, _, _, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    _, _, _, _, _, _, mesh_areas, uv_areas = preprocess(fpath)
    
    _, _, artist_area_match = get_artist_area_match(mesh_areas, uv_areas_o, uv_areas)
    
    expected = (1/8) + (1/16) + (3/16)
    
    assert artist_area_match == pytest.approx(expected)
    
