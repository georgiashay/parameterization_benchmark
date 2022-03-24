from ..utilities.overlap_area import get_overlap_area
from ..utilities.singular_values import get_singular_values
from ..utilities.jacobian import get_jacobian
from ..utilities.preprocess import preprocess

import os
import math
import pytest

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

def test_overlap_area_square():
    fpath = os.path.join(fixture_dir, "square.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    
    overlap_area = get_overlap_area(ftc, uv, singular_values)
       
    assert overlap_area == pytest.approx(0)
    
def test_overlap_area_folded():
    fpath = os.path.join(fixture_dir, "folded.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    
    overlap_area = get_overlap_area(ftc, uv, singular_values)
       
    assert overlap_area == pytest.approx(1)
    
def test_overlap_area_half_folded():
    fpath = os.path.join(fixture_dir, "half_folded.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    
    overlap_area = get_overlap_area(ftc, uv, singular_values)
       
    assert overlap_area == pytest.approx(0.5)
    
def test_overlap_area_half_strip_fold():
    fpath = os.path.join(fixture_dir, "strip_fold.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    
    overlap_area = get_overlap_area(ftc, uv, singular_values)
       
    assert overlap_area == pytest.approx(0.7)
    
def test_overlap_area_half_strip_fold_partial():
    fpath = os.path.join(fixture_dir, "strip_fold_partial.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J = get_jacobian(v, f, uv, ftc)
    singular_values, _, _ = get_singular_values(J)
    
    overlap_area = get_overlap_area(ftc, uv, singular_values)
       
    assert overlap_area == pytest.approx(0.55)