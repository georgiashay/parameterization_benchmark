from ..utilities.artist_match_angle import get_artist_angle_match
from ..utilities.angle_distortion import get_angle_distortion
from ..utilities.jacobian import get_jacobian
from ..utilities.singular_values import get_singular_values
from ..utilities.preprocess import preprocess 

import os
import math
import pytest
import numpy as np

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

def test_artist_match_angle_square_scaled():
    ofpath = os.path.join(fixture_dir, "square.obj")
    fpath = os.path.join(fixture_dir, "square_scaled.obj")
    
    _, _, f_o, ftc_o, v_o, uv_o, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    J_o = get_jacobian(v_o, f_o, uv_o, ftc_o)
    singular_values_o, _, _ = get_singular_values(J_o)
    _, angle_errors_o, _, _ = get_angle_distortion(singular_values_o, mesh_areas_o, v_o, f_o, uv_o, ftc_o)
    
    _, _, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    _, angle_errors, _, _ = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
    
    _, _, artist_match_angle = get_artist_angle_match(angle_errors_o, angle_errors, mesh_areas)
    assert artist_match_angle == pytest.approx(0)
    
def test_artist_match_angle_square_stretched():   
    ofpath = os.path.join(fixture_dir, "square_scaled.obj")
    fpath = os.path.join(fixture_dir, "square_stretched.obj")
    
    _, _, f_o, ftc_o, v_o, uv_o, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    J_o = get_jacobian(v_o, f_o, uv_o, ftc_o)
    singular_values_o, _, _ = get_singular_values(J_o)
    _, angle_errors_o, _, _ = get_angle_distortion(singular_values_o, mesh_areas_o, v_o, f_o, uv_o, ftc_o)
    
    _, _, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    _, angle_errors, _, _ = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
    
    _, _, artist_match_angle = get_artist_angle_match(angle_errors_o, angle_errors, mesh_areas)
    
    expected = abs(math.pi/4 - math.atan2(1, 2)) + abs(math.pi/4 - math.atan2(2, 1))
    
    assert artist_match_angle == pytest.approx(expected)
    
def test_artist_match_angle_diff_shifts():
    ofpath = os.path.join(fixture_dir, "right_shift.obj")
    fpath = os.path.join(fixture_dir, "left_shift.obj")
    
    _, _, f_o, ftc_o, v_o, uv_o, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    J_o = get_jacobian(v_o, f_o, uv_o, ftc_o)
    singular_values_o, _, _ = get_singular_values(J_o)
    _, angle_errors_o, _, _ = get_angle_distortion(singular_values_o, mesh_areas_o, v_o, f_o, uv_o, ftc_o)
    
    _, _, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    _, angle_errors, _, _ = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
    
    _, _, artist_match_angle = get_artist_angle_match(angle_errors_o, angle_errors, mesh_areas)
    
    expected = 0.5 * (abs(math.atan2(1, 3) - math.atan2(1, 2)) + abs(math.atan2(3, 1)  - math.atan2(2, 1)))
    
    assert artist_match_angle == pytest.approx(expected)

def test_artist_match_angle_flattened_triangle():
    ofpath = os.path.join(fixture_dir, "without_flat.obj")
    fpath = os.path.join(fixture_dir, "with_flat.obj")
    
    _, _, f_o, ftc_o, v_o, uv_o, mesh_areas_o, uv_areas_o = preprocess(ofpath)
    J_o = get_jacobian(v_o, f_o, uv_o, ftc_o)
    singular_values_o, _, _ = get_singular_values(J_o)
    _, angle_errors_o, _, _ = get_angle_distortion(singular_values_o, mesh_areas_o, v_o, f_o, uv_o, ftc_o)
    
    _, _, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    _, angle_errors, _, _ = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
    
    _, _, artist_match_angle = get_artist_angle_match(angle_errors_o, angle_errors, mesh_areas)
        
    assert artist_match_angle == pytest.approx(math.pi/2)