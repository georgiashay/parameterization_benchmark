## Script to generate a bunch of plots from the singular value CSV file.
# Feel free to move this around to whatever folder you need, Georgia.

import sys
import os
import csv
import math
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pb



def selected_plots(filepath,artist_filepath=None):
    artist = artist_filepath != None

    read_first_row = False

    object_id = []
    filename = []
    max_area_distortion = []
    total_area_distortion = []
    min_singular_value = []
    max_singular_value = []
    percentage_flipped_triangles = []
    bijectivity_violation_area = []
    max_angle_distortion = []
    total_angle_distortion = []
    resolution = []
    artist_area_match = []
    artist_angle_match = []

    with open(filepath, newline='') as parsing:
        reader = csv.reader(parsing)
        for row in reader:
            assert len(row) == 13
            if not read_first_row:
                read_first_row = True
                #Make sure all columns have the expected headers
                assert row[1] == 'Filename'
                assert row[2] == 'Max Area Distortion'
                assert row[3] == 'Total Area Distortion'
                assert row[4] == 'Min Singular Value'
                assert row[5] == 'Max Singular Value'
                assert row[6] == 'Percentage Flipped Triangles'
                assert row[7] == 'Bijectivity Violation Area'
                assert row[8] == 'Max Angle Distortion'
                assert row[9] == 'Total Angle Distortion'
                assert row[10] == 'Resolution'
                assert row[11] == 'Artist Area Match'
                assert row[12] == 'Artist Angle Match'
            else:
                #Given the namings above, put everything into the correct array.
                object_id.append(to_int(row[0]))
                filename.append(row[1])
                max_area_distortion.append(to_float(row[2]))
                total_area_distortion.append(to_float(row[3]))
                min_singular_value.append(to_float(row[4]))
                max_singular_value.append(to_float(row[5]))
                percentage_flipped_triangles.append(to_float(row[6]))
                bijectivity_violation_area.append(to_float(row[7]))
                max_angle_distortion.append(to_float(row[8]))
                total_angle_distortion.append(to_float(row[9]))
                resolution.append(to_float(row[10]))
                artist_area_match.append(to_float(row[11]))
                artist_angle_match.append(to_float(row[12]))

    #Creating nice plots using the guide from
    #https://towardsdatascience.com/a-simple-guide-to-beautiful-visualizations-in-python-f564e6b9d392
    sb.set_style('white')
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    plt.rc('font', size=12)
    palette = 'Pastel1'

    #Percentage flipped triangles histogram
    nbins=10
    plt.figure(figsize=(10,4), tight_layout=True)
    axis = sb.histplot(data=percentage_flipped_triangles,
        bins=10,
        palette=palette,
        linewidth=2)
    axis.set(title='Percentage of flipped triangles, log scale',
        yscale='log',
        xlabel='',
        ylabel='count')
    axis.set_xticks(np.linspace(0., 1., num=nbins+1, endpoint=True))

    axis.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    plt.show()




#Custom int and float conversions
def to_float(val):
    if val == '':
        return None
    try:
        f = float(val)
        return f
    except:
        return None
def to_int(val):
    if val == '':
        return None
    try:
        i = int(val)
        return i
    except:
        return None


