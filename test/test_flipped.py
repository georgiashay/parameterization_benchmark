from ..utilities.flipped import get_flipped
from ..utilities.jacobian import get_jacobian
from ..utilities.preprocess import preprocess 

import os
import pytest

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

def test_flipped_square():
    fpath = os.path.join(fixture_dir, "square.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J, _ = get_jacobian(v, f, uv, ftc)
    
    flipped = get_flipped(J)
        
    assert flipped == pytest.approx(0)

def test_flipped_square_stretched():
    fpath = os.path.join(fixture_dir, "square_stretched.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J, _ = get_jacobian(v, f, uv, ftc)
    
    flipped = get_flipped(J)
        
    assert flipped == pytest.approx(0)

def test_flipped_one_flipped_tri():
    fpath = os.path.join(fixture_dir, "flipped_tri.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J, _ = get_jacobian(v, f, uv, ftc)
    
    flipped = get_flipped(J)
        
    assert flipped == pytest.approx(1)
    
def test_flipped_folded():
    fpath = os.path.join(fixture_dir, "folded.obj")
    
    v_i, uv_i, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
    J, _ = get_jacobian(v, f, uv, ftc)
    
    flipped = get_flipped(J)
        
    assert flipped == pytest.approx(0.5)