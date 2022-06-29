from ..utilities.artist_correlation import get_artist_correlation
from ..utilities.jacobian import get_jacobian
from ..utilities.singular_values import get_singular_values
from ..utilities.preprocess import preprocess 

import os
import math
import pytest
import numpy as np

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")
    
def test_artist_match_angle_self_comparison():
    meshes = ["icosahedron_hemi.obj", "icosahedron.obj", "cone_pizza.obj"]

    for m in meshes:
        fpath = os.path.join(fixture_dir, m)
    
        _, _, f, ftc, v, uv, mesh_areas, uv_areas = preprocess(fpath)
        J, _ = get_jacobian(v, f, uv, ftc)
        singular_values, _, _ = get_singular_values(J)
        rho = get_artist_angle_match(rho, rho, mesh_areas)
     
        assert rho == pytest.approx(0.)
