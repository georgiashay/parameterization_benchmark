from ..utilities.jacobian import get_jacobian
from ..utilities.preprocess import preprocess
from ..utilities.resolution import get_resolution

import os, glob
import igl
import math
import pytest
import scipy
from scipy.spatial.transform import Rotation as R
import numpy as np

test_dir = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")

def test_rigid_transformations():
    #Translations & scaling of unit-sized flat things should be resolution 1.
    V = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
    Vt = V[:,0:2]
    F = np.array([[0,1,2]])
    Ft = F

    ntests = 100
    rng = np.random.default_rng()
    for i in range(ntests):
        #Random 3D translation
        t3 = rng.uniform(-2.,2.,(3))
        Vl = V + t3

        #Random 2D translation
        t2 = rng.uniform(-2.,2.,(2))
        s = rng.uniform(0.1,2.,(1))
        Vtl = Vt*s[0] + t2

        r = get_resolution(Vl,F,Vtl,Ft)
        assert abs(r-1.) < 1e-8

    #Resizing the axes uniformly, but randomly, should return the appropriate resolution
    Vt = Vt
    r0 = get_resolution(V,F,Vt,Ft)
    for i in range(ntests):
        scale = rng.uniform(0.1,2.,(2))
        r = get_resolution(V,F,Vt*scale,Ft)

        assert abs(r0*max(scale[1]/scale[0], scale[0]/scale[1]) - r) < 1e-8



