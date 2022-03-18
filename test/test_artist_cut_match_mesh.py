from ..utilities.artist_cut_match_mesh import get_artist_cut_match_mesh
from ..utilities.jacobian import get_jacobian
from ..utilities.singular_values import get_singular_values
from ..utilities.preprocess import preprocess 
from ..utilities.v_uv_map import get_v_uv_map

import os
import math
import pytest

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")


def test_artist_cut_match_mesh_cube():
    ofpath = os.path.join(fixture_dir, "cube.obj")
    fpath = os.path.join(fixture_dir, "cube_cutoff.obj")
    
    v_io, uv_io, f_o, ftc_o, v_o, uv_o, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    v_to_uv_o, uv_to_v_o, uv_to_v_arr_o = get_v_uv_map(f_o, ftc_o)
    v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f, ftc)
    
    artist_cut_match_mesh = get_artist_cut_match_mesh(uv_to_v_arr_o, v_o, ftc_o, uv_to_v_arr, v, ftc)
    
    assert artist_cut_match_mesh == pytest.approx(2/math.sqrt(6))
    
def test_artist_cut_match_mesh_icosa():
    ofpath = os.path.join(fixture_dir, "icosahedron_hemi.obj")
    fpath = os.path.join(fixture_dir, "icosahedron.obj")
    
    v_io, uv_io, f_o, ftc_o, v_o, uv_o, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    
    v_to_uv_o, uv_to_v_o, uv_to_v_arr_o = get_v_uv_map(f_o, ftc_o)
    v_to_uv, uv_to_v, uv_to_v_arr = get_v_uv_map(f, ftc)
    
    artist_cut_match_mesh = get_artist_cut_match_mesh(uv_to_v_arr_o, v_o, ftc_o, uv_to_v_arr, v, ftc)
    
    scaled_edge_length = math.sqrt((4/math.sqrt(3)) * (1/20))
    
    measure_length = 11 * scaled_edge_length
    artist_length = 6 * scaled_edge_length
    
    assert artist_cut_match_mesh == pytest.approx(measure_length - artist_length)
    
    
