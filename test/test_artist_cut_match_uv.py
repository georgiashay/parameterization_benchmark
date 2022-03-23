from ..utilities.artist_cut_match_uv import get_artist_cut_match_uv
from ..utilities.jacobian import get_jacobian
from ..utilities.singular_values import get_singular_values
from ..utilities.preprocess import preprocess 

import os
import math
import pytest

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")


def test_artist_cut_match_uv_cube():
    ofpath = os.path.join(fixture_dir, "cube.obj")
    fpath = os.path.join(fixture_dir, "cube_cutoff.obj")
    
    v_io, uv_io, f_o, ftc_o, v_o, uv_o, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    artist_cut_match_uv = get_artist_cut_match_uv(uv_o, ftc_o, uv, ftc)
    
    assert artist_cut_match_uv == pytest.approx(4/math.sqrt(6))
    
def test_artist_cut_match_uv_icosa():
    ofpath = os.path.join(fixture_dir, "icosahedron_hemi.obj")
    fpath = os.path.join(fixture_dir, "icosahedron.obj")
    
    v_io, uv_io, f_o, ftc_o, v_o, uv_o, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    artist_cut_match_uv = get_artist_cut_match_uv(uv_o, ftc_o, uv, ftc)
    
    scaled_edge_length_measure = math.sqrt((4/math.sqrt(3)) * (1/20))
    measure_length = 22 * scaled_edge_length_measure
    
    scaled_edge_length_artist = 0.447785
    artist_length = 12 * scaled_edge_length_artist
  
    assert artist_cut_match_uv == pytest.approx(measure_length - artist_length, abs=1e-3)