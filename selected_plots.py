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
import pandas as pd
from types import SimpleNamespace


def selected_plots(path1,
    path2=None,
    name1='dataset 1',
    name2='dataset 2',
    out_dir='.'):
    #Read in the CSV files to create a namespace
    data1 = read_csv(path1)
    data2 = None if path2==None else read_csv(path2)

    # data2 = None

    #Creating nice plots using the guide from
    #https://towardsdatascience.com/a-simple-guide-to-beautiful-visualizations-in-python-f564e6b9d392
    sb.set_style('white')
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    plt.rc('font', size=12)

    #Load packaged Linux Libertine
    fontdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        'utilities', 'assets', 'fonts')
    for font in matplotlib.font_manager.findSystemFonts(fontdir):
        matplotlib.font_manager.fontManager.addfont(font)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Linux Libertine')

    #Percentage flipped triangles histogram
    if data2==None:
        axis = percentage_flipped_triangles_hist(data1.percentage_flipped_triangles,
            name1)
    else:
        axis = percentage_flipped_triangles_hist(data1.percentage_flipped_triangles,
            name1,
            data2.percentage_flipped_triangles,
            name2)
    percentage_flipped_path = os.path.join(out_dir, 'percentage_flipped.pdf')
    plt.savefig(percentage_flipped_path)
    plt.clf()

    #Bijectivity area violation histogram
    if data2==None:
        axis = overlap_area_hist(data1.bijectivity_violation_area,
            name1)
    else:
        axis = overlap_area_hist(data1.bijectivity_violation_area,
            name1,
            data2.bijectivity_violation_area,
            name2)
    bijectivity_violation_path = os.path.join(out_dir, 'bijectivity_violation.pdf')
    plt.savefig(bijectivity_violation_path)
    plt.clf()



#Percentage flipped triangles histogram
def percentage_flipped_triangles_hist(data1,
    name1,
    data2=None,
    name2=None,
    nbins=11,
    palette = 'Pastel1'):

    c = 1e-8
    def handle_extremes(data):
        lo = [-1./nbins for x in data if x<c]
        hi = [x for x in data if x>=c]
        if len(lo)==0:
            lo = pd.Series([],dtype=pd.Float64Dtype())
        if len(hi)==0:
            hi = pd.Series([],dtype=pd.Float64Dtype()) 
        return lo,hi
    def remove_nones(data):
        return [1. if x==None else x for x in data]
    def preproc(data):
        return handle_extremes(remove_nones(data))

    to_plot1_lo,to_plot1 = preproc(data1)
    if data2 == None:
        to_plot_lo = {name1 : to_plot1_lo}
        to_plot = {name1 : to_plot1}
        names = [name1]
    else:
        to_plot2_lo,to_plot2 = preproc(data2)
        to_plot_lo = {name1 : to_plot1_lo, name2 : to_plot2_lo}
        to_plot = {name1 : to_plot1, name2 : to_plot2}
        names = [name1,name2]
    plt.figure(figsize=(10,4), tight_layout=True)
    axis = sb.histplot(data=to_plot,
        bins=nbins-1,
        palette=palette,
        shrink = 0.75,
        linewidth=2,
        legend=True,
        hue_order=names,
        multiple='dodge')

    #Zero bin
    axis = sb.histplot(data=to_plot_lo,
        bins=1,
        palette=palette,
        shrink = 0.75/nbins,
        linewidth=2,
        legend=True,
        hue_order=names,
        multiple='dodge',
        ax=axis)

    #Draw dotted line separating left and right bins
    axis.plot([-0.3/nbins,-0.3/nbins], [0.,1.2*axis.get_ylim()[1]],
        linestyle='dashed',
        dashes=(10,5),
        linewidth=0.5,
        color=(0.2,0.2,0.2))

    #Deal with labels
    axis.set(title='Percentage of flipped triangles',
        yscale='log',
        xlabel='(failed parametrizations reported as 100% flipped)',
        ylabel='count (log scale)')
    ticks = [-1./nbins, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    ticklabels = ['0%', '>0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
    axis.set_xticks(ticks,ticklabels)
    axis.margins(y=0)
    sb.move_legend(axis, "upper center")

    return axis

#Bijectivity area violation histogram
def overlap_area_hist(data1,
    name1,
    data2=None,
    name2=None,
    nbins=10,
    palette = 'Pastel1'):

    c = 1e-8
    def non_none_max(data):
        m = 0.
        for x in data:
            if x != None:
                if math.isfinite(x):
                    if x>m:
                        m = x
        return m
    cmax = non_none_max(data1) if data2==None else max(non_none_max(data1),non_none_max(data2))
    zero_tick = - 1.5*cmax/nbins
    inf_tick = cmax*(1. + 1.5/nbins)
    def handle_extremes(data):
        lo = [zero_tick for x in data if x<c]
        mid = [x for x in data if (x>=c and x<math.inf)]
        hi = [inf_tick for x in data if x==math.inf]
        if len(lo)==0:
            lo = pd.Series([],dtype=pd.Float64Dtype()) 
        if len(mid)==0:
            mid = pd.Series([],dtype=pd.Float64Dtype()) 
        if len(hi)==0:
            hi = pd.Series([],dtype=pd.Float64Dtype()) 
        return lo,mid,hi
    def remove_nones(data):
        return [math.inf if x==None else x for x in data]
    def preproc(data):
        return handle_extremes(remove_nones(data))

    to_plot1_lo,to_plot1,to_plot1_hi = preproc(data1)
    if data2 == None:
        to_plot_lo = {name1 : to_plot1_lo}
        to_plot = {name1 : to_plot1}
        to_plot_hi = {name1 : to_plot1_hi}
        names = [name1]
    else:
        to_plot2_lo,to_plot2,to_plot2_hi = preproc(data2)
        to_plot_lo = {name1 : to_plot1_lo, name2 : to_plot2_lo}
        to_plot = {name1 : to_plot1, name2 : to_plot2}
        to_plot_hi = {name1 : to_plot1_hi, name2 : to_plot2_hi}
        names = [name1,name2]
    plt.figure(figsize=(10,4), tight_layout=True)
    axis = sb.histplot(data=to_plot,
        bins=nbins-2,
        palette=palette,
        shrink = 0.75,
        linewidth=2,
        legend=True,
        hue_order=names,
        multiple='dodge')

    #Zero bin
    axis = sb.histplot(data=to_plot_lo,
        bins=1,
        palette=palette,
        shrink = 0.75/nbins,
        linewidth=2,
        legend=True,
        hue_order=names,
        multiple='dodge',
        ax=axis)

    #Inf bin
    axis = sb.histplot(data=to_plot_hi,
        bins=1,
        palette=palette,
        shrink = 0.75/nbins,
        linewidth=2,
        legend=True,
        hue_order=names,
        multiple='dodge',
        ax=axis)

    #Draw dotted line separating left and middle bins
    axis.plot([-0.5/nbins,-0.5/nbins], [0.,1.2*axis.get_ylim()[1]],
        linestyle='dashed',
        dashes=(10,5),
        linewidth=0.5,
        color=(0.2,0.2,0.2))

    #Draw dotted line separating middle and right bins
    axis.plot([(1.+0.5/nbins)*cmax,(1.+0.5/nbins)*cmax], [0.,1.2*axis.get_ylim()[1]],
        linestyle='dashed',
        dashes=(10,5),
        linewidth=0.5,
        color=(0.2,0.2,0.2))

    #Deal with labels
    axis.set(title='Overlapping area in UV map',
        yscale='log',
        xlabel='(failed parametrizations reported as ∞)',
        ylabel='count (log scale)')
    ticks = np.concatenate(([zero_tick], np.linspace(0., cmax, num=nbins-1, endpoint=True), [inf_tick]))
    ticklabels = ["0"] + ["%.1e"%a for a in np.linspace(c, cmax, num=nbins-1, endpoint=True)] + ["∞"]
    axis.set_xticks(ticks,ticklabels)
    axis.margins(y=0)
    sb.move_legend(axis, "upper center")

    return axis


#Read in data from CSV
def read_csv(path):

    data = SimpleNamespace()
    data.object_id = []
    data.nfaces = []
    data.nvertices = []
    data.filename = []
    data.max_area_distortion = []
    data.total_area_distortion = []
    data.min_singular_value = []
    data.max_singular_value = []
    data.percentage_flipped_triangles = []
    data.bijectivity_violation_area = []
    data.max_angle_distortion = []
    data.total_angle_distortion = []
    data.resolution = []
    data.artist_area_match = []
    data.artist_angle_match = []

    read_first_row = False
    with open(path, newline='') as parsing:
        reader = csv.reader(parsing)
        for row in reader:
            assert len(row) == 15
            if not read_first_row:
                read_first_row = True
                #Make sure all columns have the expected headers
                assert row[1] == 'Filename'
                assert row[2] == 'Faces'
                assert row[3] == 'Vertices'
                assert row[4] == 'Max Area Distortion'
                assert row[5] == 'Total Area Distortion'
                assert row[6] == 'Min Singular Value'
                assert row[7] == 'Max Singular Value'
                assert row[8] == 'Percentage Flipped Triangles'
                assert row[9] == 'Bijectivity Violation Area'
                assert row[10] == 'Max Angle Distortion'
                assert row[11] == 'Total Angle Distortion'
                assert row[12] == 'Resolution'
                assert row[13] == 'Artist Area Match'
                assert row[14] == 'Artist Angle Match'
            else:
                #Given the namings above, put everything into the correct array.
                data.object_id.append(to_int(row[0]))
                data.filename.append(row[1])
                data.nfaces.append(to_int(row[2]))
                data.nvertices.append(to_int(row[3]))
                data.max_area_distortion.append(to_float(row[4]))
                data.total_area_distortion.append(to_float(row[5]))
                data.min_singular_value.append(to_float(row[6]))
                data.max_singular_value.append(to_float(row[7]))
                data.percentage_flipped_triangles.append(to_float(row[8]))
                data.bijectivity_violation_area.append(to_float(row[9]))
                data.max_angle_distortion.append(to_float(row[10]))
                data.total_angle_distortion.append(to_float(row[11]))
                data.resolution.append(to_float(row[12]))
                data.artist_area_match.append(to_float(row[13]))
                data.artist_angle_match.append(to_float(row[14]))

    return data


#Custom int and float conversions
def to_float(val):
    if val == '':
        return None
    try:
        f = float(val)
        if not math.isfinite(f):
            return None
        return f
    except:
        return None
def to_int(val):
    if val == '':
        return None
    try:
        i = int(val)
        if not math.isfinite(i):
            return None
        return i
    except:
        return None

