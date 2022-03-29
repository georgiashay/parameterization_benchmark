import sys
import os

import argparse
import pandas as pd
import numpy as np
import scipy
import igl

from utilities.area_distortion import get_area_distortion
from utilities.preprocess import preprocess
from utilities.jacobian import get_jacobian
from utilities.uv_coordinates import get_uv_coordinates
from utilities.singular_values import get_singular_values
from utilities.flipped import get_flipped
from utilities.angle_distortion import get_angle_distortion
from utilities.overlap_area import get_overlap_area
from utilities.resolution import get_resolution
from selected_plots import selected_plots

# selected_plots('/home/ostein/Desktop/new_result.csv', '/home/ostein/Desktop/new_artist.csv')
selected_plots('/Users/ostein/Desktop/new_result.csv', '/Users/ostein/Desktop/new_artist.csv')


