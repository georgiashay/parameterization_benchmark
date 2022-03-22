from ..utilities.angle_distortion import get_angle_distortion
from ..utilities.jacobian import get_jacobian
from ..utilities.singular_values import get_singular_values
from ..utilities.preprocess import preprocess 

import os
import math
import pytest

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

def test_angle_distortion_square():
    fpath = os.path.join(fixture_dir, "square.obj")
    _, _, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, min_singular_value, max_singular_value = get_singular_values(J)
    
    angle_distortions, max_angle_distortion, total_angle_distortion = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
    
    assert total_angle_distortion == pytest.approx(0)
    assert max_angle_distortion == pytest.approx(0)
    
def test_angle_distortion_scaled_square():
    fpath = os.path.join(fixture_dir, "square_scaled.obj")
    _, _, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, min_singular_value, max_singular_value = get_singular_values(J)
    
    angle_distortions, max_angle_distortion, total_angle_distortion = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
    
    assert total_angle_distortion == pytest.approx(0)
    assert max_angle_distortion == pytest.approx(0)
    
def test_angle_distortion_stretched_square():
    fpath = os.path.join(fixture_dir, "square_stretched.obj")
    _, _, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, min_singular_value, max_singular_value = get_singular_values(J)
    
    angle_distortions, max_angle_distortion, total_angle_distortion = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
    
    expected = abs(math.pi/4 - math.atan2(1, 2)) + abs(math.pi/4 - math.atan2(2, 1))
    
    assert max_angle_distortion == pytest.approx(0.5)
    assert total_angle_distortion == pytest.approx(expected)
    
def test_angle_distortion_grid_shift():
    fpath = os.path.join(fixture_dir, "grid_shift.obj")
    _, _, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, min_singular_value, max_singular_value = get_singular_values(J)
    
    angle_distortions, max_angle_distortion, total_angle_distortion = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
    
    expected_1 = abs(math.pi/4 - math.atan2(1, 3)) + abs(math.pi/4 - math.atan2(3, 1))
    expected_2 = abs(math.pi/4 - math.atan2(1, 1)) + abs(math.pi/4 - math.atan2(1, 1))
    
    assert max_angle_distortion == pytest.approx(4/3)
    assert total_angle_distortion == pytest.approx((expected_1 + expected_2)/2)
    
def test_angle_distortion_triangle():
    fpath = os.path.join(fixture_dir, "triangle.obj")
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, min_singular_value, max_singular_value = get_singular_values(J)
    
    angle_distortions, max_angle_distortion, total_angle_distortion = get_angle_distortion(singular_values, mesh_areas, v, f, uv, ftc)
    
    sigma_1 = 2/math.sqrt(6)
    sigma_2 = 3/math.sqrt(6)
    
    angle_1 = abs(math.pi/4 - math.atan2(3, 2))
    angle_2 = abs(math.pi/4 - math.atan2(2, 3))
    
    assert max_angle_distortion == pytest.approx((sigma_1/sigma_2) + (sigma_2/sigma_1) - 2)
    assert total_angle_distortion == pytest.approx(angle_1 + angle_2)