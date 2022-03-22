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
    produce_scatter=True,
    out_dir='.'):
    #Read in the CSV files to create a namespace
    data1 = read_csv(path1)
    data2 = None if path2==None else read_csv(path2)
    if data2 != None:
        assert data1.object_id == data2.object_id
        assert data1.nfaces == data2.nfaces
        assert data1.nvertices == data2.nvertices

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
        axis = percent_hist(data1.percentage_flipped_triangles,
            name1,
            title='Percentage of flipped triangles',
            comment='(failed parametrizations reported as 100% flipped)')
    else:
        axis = percent_hist(data1.percentage_flipped_triangles,
            name1,
            data2.percentage_flipped_triangles,
            name2,
            title='Percentage of flipped triangles',
            comment='(failed parametrizations reported as 100% flipped)')
    percentage_flipped_path = os.path.join(out_dir, 'percentage_flipped.pdf')
    plt.savefig(percentage_flipped_path)
    plt.clf()

    #Bijectivity area violation histogram
    if data2==None:
        axis = hist(data1.bijectivity_violation_area,
            name1,
            title='Overlapping area in UV map',
            comment='(failed parametrizations reported as ∞)')
    else:
        axis = hist(data1.bijectivity_violation_area,
            name1,
            data2.bijectivity_violation_area,
            name2,
            title='Overlapping area in UV map',
            comment='(failed parametrizations reported as ∞)')
    bijectivity_violation_path = os.path.join(out_dir, 'bijectivity_violation.pdf')
    plt.savefig(bijectivity_violation_path)
    plt.clf()

    #Compare max angle distortion
    if produce_scatter:
        if data2!=None:
            axis = scatter_comparison(data1.max_angle_distortion,
                name1,
                data2.max_angle_distortion,
                name2,
                huedata=data1.nfaces,
                huename="#faces",
                title="Maximal angle distortion")
            max_angle_distortion_path = os.path.join(out_dir, 'max_angle_distortion_comp.pdf')
            plt.savefig(max_angle_distortion_path)
            plt.clf()

        if data2==None:
            axis = scatter_vs_property(data1.nfaces,
                "#faces",
                data1.max_angle_distortion,
                name1,
                title="Maximal angle distortion")
        else:
            axis = scatter_vs_property(data1.nfaces,
                "#faces",
                data1.max_angle_distortion,
                name1,
                data2.max_angle_distortion,
                name2,
                title="Maximal angle distortion")
        max_angle_distortion_path = os.path.join(out_dir, 'max_angle_distortion.pdf')
        plt.savefig(max_angle_distortion_path)
        plt.clf()


    #Compare total angle distortion
    if produce_scatter:
        if data2!=None:
            axis = scatter_comparison(data1.max_angle_distortion,
                name1,
                data2.max_angle_distortion,
                name2,
                huedata=data1.nfaces,
                huename="#faces",
                title="Total angle distortion")
            total_angle_distortion_path = os.path.join(out_dir, 'total_angle_distortion_comp.pdf')
            plt.savefig(total_angle_distortion_path)
            plt.clf()

        if data2==None:
            axis = scatter_vs_property(data1.nfaces,
                "#faces",
                data1.max_angle_distortion,
                name1,
                title="Total angle distortion")
        else:
            axis = scatter_vs_property(data1.nfaces,
                "#faces",
                data1.max_angle_distortion,
                name1,
                data2.max_angle_distortion,
                name2,
                title="Total angle distortion")
        total_angle_distortion_path = os.path.join(out_dir, 'total_angle_distortion.pdf')
        plt.savefig(total_angle_distortion_path)
        plt.clf()




                # assert row[4] == 'Max Area Distortion'
                # assert row[6] == 'Min Singular Value'
                # assert row[7] == 'Max Singular Value'
                # assert row[8] == 'Percentage Flipped Triangles'
                # assert row[9] == 'Bijectivity Violation Area'
                # assert row[11] == 'Total Angle Distortion'
                # assert row[12] == 'Resolution'
                # assert row[13] == 'Artist Area Match'
                # assert row[14] == 'Artist Angle Match'



#Percent histogram with special zero bin
def percent_hist(data1,
    name1,
    data2=None,
    name2=None,
    title='',
    comment='',
    zero_bin=True,
    palette = 'Pastel1',):

    assert all(x==None or x>=0 for x in data1)
    assert data2==None or all(x==None or x>=0 for x in data2)

    nbins = 11
    rnbins = nbins - (zero_bin==True)

    c = 1e-8
    def handle_extremes(data):
        if zero_bin:
            lo = [-1./nbins for x in data if x<c]
            hi = [x for x in data if x>=c]
        else:
            lo = []
            hi = data
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
    draw_zero_bin = zero_bin
    if data2 == None:
        to_plot_lo = (to_plot1_lo)
        if len(to_plot1_lo)==0:
            draw_zero_bin = False
        to_plot = (to_plot1)
        names = [name1]
        colors = plt.cm.get_cmap(palette).colors[0]
    else:
        to_plot2_lo,to_plot2 = preproc(data2)
        if len(to_plot1_lo)==0 and len(to_plot2_lo)==0:
            draw_zero_bin = False
        to_plot_lo = (to_plot1_lo, to_plot2_lo)
        to_plot = (to_plot1, to_plot2)
        names = [name1,name2]
        colors = (plt.cm.get_cmap(palette).colors[0], plt.cm.get_cmap(palette).colors[1])
    plt.figure(figsize=(10,4), tight_layout=True)
    axis = plt.gca()
    axis.hist(to_plot,
        bins=rnbins,
        rwidth=0.75,
        label=names,
        color=colors)
    axis.legend()

    #Deal with labels
    axis.set(title=title,
        yscale='log',
        xlabel=comment,
        ylabel='count (log scale)')

    if draw_zero_bin:
        #Zero bin
        axis.hist(to_plot_lo,
            bins=1,
            rwidth=0.75/(nbins-1),
            label=names,
            color=colors)

        #Draw dotted line separating left and right bins
        axis.plot([-0.3/nbins,-0.3/nbins], [axis.get_ylim()[0],1.2*axis.get_ylim()[1]],
            linestyle='dashed',
            dashes=(10,5),
            linewidth=0.5,
            color=(0.2,0.2,0.2))

    if draw_zero_bin:
        ticks = [-1./nbins, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        ticklabels = ['0%', '>0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
        axis.margins(y=0)
    else:
        ticks = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        ticklabels = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
        axis.margins(y=0.1)
    axis.set_xticks(ticks,ticklabels)
    sb.move_legend(axis, "upper center")

    return axis

#Histogram with zero and inf bins
def hist(data1,
    name1,
    data2=None,
    name2=None,
    nbins=10,
    title='',
    comment='',
    zero_bin=True,
    inf_bin=True,
    palette = 'Pastel1'):

    assert all(x==None or x>=0 for x in data1)
    assert data2==None or all(x==None or x>=0 for x in data2)
    assert nbins>=3
    rnbins = nbins - (zero_bin==True) - (inf_bin==True)

    c = 1e-8
    def non_none_max(data):
        m = 1.
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
    draw_zero_bin = zero_bin
    draw_inf_bin = inf_bin
    if data2 == None:
        to_plot_lo = [to_plot1_lo]
        if len(to_plot1_lo)==0:
            draw_zero_bin = False
        if len(to_plot1_hi)==0:
            draw_inf_bin = False
        to_plot = [to_plot1]
        to_plot_hi = [to_plot1_hi]
        names = [name1]
        colors = plt.cm.get_cmap(palette).colors[0]
    else:
        to_plot2_lo,to_plot2,to_plot2_hi = preproc(data2)
        if len(to_plot1_lo)==0 and len(to_plot2_lo)==0:
            draw_zero_bin = False
        if len(to_plot1_hi)==0 and len(to_plot2_hi)==0:
            draw_inf_bin = False
        to_plot_lo = [to_plot1_lo, to_plot2_lo]
        to_plot = [to_plot1, to_plot2]
        to_plot_hi = [to_plot1_hi, to_plot2_hi]
        names = [name1,name2]
        colors = (plt.cm.get_cmap(palette).colors[0], plt.cm.get_cmap(palette).colors[1])

    plt.figure(figsize=(10,4), tight_layout=True)
    axis = plt.gca()
    axis.hist(to_plot,
        bins=rnbins,
        rwidth=0.75,
        label=names,
        color=colors)
    axis.legend()

    #Zero bin
    if draw_zero_bin:
        axis.hist(to_plot_lo,
            bins=1,
            rwidth=0.75/(nbins-2),
            label=names,
            color=colors)

    #Inf bin
    if draw_inf_bin:
        axis.hist(to_plot_hi,
            bins=1,
            rwidth=0.75/(nbins-2),
            label=names,
            color=colors)

    #Deal with labels
    axis.set(title=title,
        yscale='log',
        xlabel=comment,
        ylabel='count (log scale)')

    axlo = axis.get_ylim()[0]
    axhi = axis.get_ylim()[1]
    #Draw dotted line separating left and middle bins
    if draw_zero_bin:
        axis.plot([-0.5/nbins,-0.5/nbins], [axlo,1.2*axhi],
            linestyle='dashed',
            dashes=(10,5),
            linewidth=0.5,
            color=(0.2,0.2,0.2))

    #Draw dotted line separating middle and right bins
    if draw_inf_bin:
        axis.plot([(1.+0.5/nbins)*cmax,(1.+0.5/nbins)*cmax], [axlo,1.2*axhi],
            linestyle='dashed',
            dashes=(10,5),
            linewidth=0.5,
            color=(0.2,0.2,0.2))

    if draw_zero_bin and not draw_inf_bin:
        ticks = np.concatenate(([zero_tick], np.linspace(0., cmax, num=rnbins+1, endpoint=True)))
        ticklabels = ["0"] + ["%.1e"%a for a in np.linspace(c, cmax, num=rnbins+1, endpoint=True)]
        axis.margins(y=0)
    elif draw_inf_bin and not draw_zero_bin:
        ticks = np.concatenate((np.linspace(0., cmax, num=rnbins+1, endpoint=True), [inf_tick]))
        ticklabels = ["%.1e"%a for a in np.linspace(c, cmax, num=rnbins+1, endpoint=True)] + ["∞"]
        axis.margins(y=0)
    elif draw_inf_bin and draw_zero_bin:
        ticks = np.concatenate(([zero_tick], np.linspace(0., cmax, num=rnbins+1, endpoint=True), [inf_tick]))
        ticklabels = ["0"] + ["%.1e"%a for a in np.linspace(c, cmax, num=rnbins+1, endpoint=True)] + ["∞"]
        axis.margins(y=0)
    else:
        ticks = np.linspace(0., cmax, num=rnbins+1, endpoint=True)
        ticklabels = ["%.1e"%a for a in np.linspace(c, cmax, num=rnbins+1, endpoint=True)]
        axis.margins(y=0.1)

    axis.set_xticks(ticks,ticklabels)
    sb.move_legend(axis, "upper center")

    return axis

#Scatter comparison plot
def scatter_comparison(data1,
    name1,
    data2,
    name2,
    huedata=None,
    huename=None,
    title='',
    nticks=10,
    nhueticks=6,
    logx=True,
    logy=True,
    loghue=True,
    palette = 'plasma'):

    assert all(x==None or x>=0 for x in data1)
    assert all(x==None or x>=0 for x in data2)
    if huedata!=None:
        assert all(x!=None and math.isfinite(x) for x in huedata)
    base = 10.

    def non_none_max(data):
        m = 0.
        for x in data:
            if x != None:
                if math.isfinite(x):
                    if x>m:
                        m = x
        return m
    def non_none_min(data):
        m = math.inf
        for x in data:
            if x != None:
                if math.isfinite(x):
                    if x<m:
                        m = x
        return m
    smallestlognum = 1e-16
    def handle_extremes(data,log):
        cmax = non_none_max(data)
        cmin = non_none_min(data)
        if log:
            inf_tick = base ** (math.log(cmax,base)*(1. + 1./nticks))
        else:
            inf_tick = cmax*(1. + 1./nticks)
        lo = [x for x in data if x<math.inf]
        hi = [inf_tick for x in data if x==math.inf]
        data = lo + hi
        if log:
            if smallestlognum > cmin:
                minlabel = 0
                cmin = smallestlognum
            else:
                minlabel = cmin
            data = [smallestlognum if x<smallestlognum else x for x in data]
        else:
            minlabel = None
        return data,cmin,cmax,minlabel,inf_tick
    def remove_nones(data):
        return [math.inf if x==None else x for x in data]
    def preproc(data,log):
        return handle_extremes(remove_nones(data),log)

    to_plot1,min1,max1,minlabel1,inf1 = preproc(data1,logx)
    to_plot2,min2,max2,minlabel2,inf2 = preproc(data2,logy)

    plt.figure(figsize=(10,4), tight_layout=True)
    axis = plt.gca()
    if huedata == None:
        axis.scatter(to_plot1,
            to_plot2,
            s=3.,
            cmap=palette,
            alpha=0.5)
    else:
        max_hue = max(huedata)
        min_hue = min(huedata)
        if(loghue):
            min_hue = max(smallestlognum,min_hue)
            normalizer = matplotlib.colors.SymLogNorm(linthresh=smallestlognum, vmin=min_hue, vmax=max_hue, base=base, clip=False)
        else:
            normalizer = matplotlib.colors.Normalize(vmin=min_hue, vmax=max_hue, clip=False)
        sc = axis.scatter(to_plot1,
            to_plot2,
            c=huedata,
            s= 3.,
            norm=normalizer,
            cmap=palette,
            alpha=0.5)
        cbar = plt.colorbar(sc)
        cbar.set_label(huename)

    #Deal with labels
    axis.set(title=title,
        xscale='log' if logx else 'linear',
        yscale='log' if logy else 'linear',
        xlabel=name1,
        ylabel=name2)
    if logx:
        loglabels = np.logspace(math.log(min1,base), math.log(max1,base), num=nticks-1, endpoint=True, base=base)
        xticks = np.concatenate((loglabels, [inf1]))
        xticklabels = ["%.1e"%minlabel1] + ["%.1e"%a for a in loglabels[1:]] + ["∞"]
    else:
        xticks = np.concatenate((np.linspace(0, max1, num=nticks-1), [inf1]))
        xticklabels = ["%.1e"%a for a in np.linspace(0, max1, num=nticks-1, endpoint=True)] + ["∞"]
    if logy:
        loglabels = np.logspace(math.log(min2,base), math.log(max2,base), num=nticks-1, endpoint=True, base=base)
        yticks = np.concatenate((loglabels, [inf2]))
        yticklabels = ["%.1e"%minlabel2] + ["%.1e"%a for a in loglabels[1:]] + ["∞"]
    else:
        yticks = np.concatenate((np.linspace(0, max2, num=nticks-1), [inf2]))
        yticklabels = ["%.1e"%a for a in np.linspace(0, max2, num=nticks-1, endpoint=True)] + ["∞"]
    axis.set_xticks(xticks,xticklabels)
    axis.set_yticks(yticks,yticklabels)
    axis.margins(x=0.005, y=0.01)

    axis.plot([min1 if logx else 0., inf1], [min2 if logy else 0., inf2],
        linestyle='dashed',
        dashes=(10,5),
        linewidth=0.5,
        alpha=0.5,
        color=(0.,0.,0.))
    axis.text(max1, max2, 'x=y     ', horizontalalignment='right', fontstyle='italic')

    return axis

#Scatter cs property plot
def scatter_vs_property(prop,
    propname,
    data1,
    name1,
    data2=None,
    name2=None,
    title='',
    nxticks=10,
    nyticks=10,
    logx=True,
    logy=True,
    palette = 'Pastel1'):

    assert all(x>0 for x in prop)
    assert len(prop)==len(data1)
    assert all(x==None or x>=0 for x in data1)
    if data2 != None:
        assert len(prop)==len(data1)
        assert all(x==None or x>=0 for x in data2)
    base = 10.

    def non_none_max(data):
        m = 0.
        for x in data:
            if x != None:
                if math.isfinite(x):
                    if x>m:
                        m = x
        return m
    def non_none_min(data):
        m = math.inf
        for x in data:
            if x != None:
                if math.isfinite(x):
                    if x<m:
                        m = x
        return m
    smallestlognum = 1e-16
    def handle_extremes(data,log,inf_provided=None,relabel_infs=True):
        cmax = non_none_max(data)
        cmin = non_none_min(data)
        if inf_provided==None:
            if log:
                inf_tick = base ** (math.log(cmax,base)*(1. + 1./nyticks))
            else:
                inf_tick = cmax*(1. + 1./nyticks)
        else:
            inf_tick = inf_provided
        lo = [x for x in data if x<math.inf]
        if relabel_infs:
            hi = [inf_tick for x in data if x==math.inf]
        else:
            hi = [x for x in data if x==math.inf]
        data = lo + hi
        if log:
            if smallestlognum > cmin:
                minlabel = 0
                cmin = smallestlognum
            else:
                minlabel = cmin
            data = [smallestlognum if x<smallestlognum else x for x in data]
        else:
            minlabel = None
        return data,cmin,cmax,minlabel,inf_tick
    def remove_nones(data):
        return [math.inf if x==None else x for x in data]
    def preproc(data,log,inf_provided=None,relabel_infs=True):
        return handle_extremes(remove_nones(data),log,
            inf_provided=inf_provided,relabel_infs=relabel_infs)

    if data2==None:
        to_plot1,min1,max1,minlabel1,inf1 = preproc(data1,logx)
        global_min,global_minlabel = min1,minlabel1
        global_max,global_inf = max1,inf1
    if data2!=None:
        to_plot1,min1,max1,minlabel1,inf1 = preproc(data1,logx,relabel_infs=False)
        to_plot2,min2,max2,minlabel2,inf2 = preproc(data2,logy,relabel_infs=False)
        if min1<min2:
            global_min,global_minlabel = min1,minlabel1
        else:
            global_min,global_minlabel = min2,minlabel2
        if max1>max2:
            global_max,global_inf = max1,inf1
        else:
            global_max,global_inf = max2,inf2
        to_plot1,_,_,_,_ = preproc(data1,logx,relabel_infs=True,inf_provided=global_inf)
        to_plot2,_,_,_,_ = preproc(data2,logy,relabel_infs=True,inf_provided=global_inf)

    max_prop = max(prop)
    min_prop = min(prop)

    plt.figure(figsize=(10,4), tight_layout=True)
    axis = plt.gca()
    if data2==None:
        print(plt.cm.get_cmap(palette).colors[0])
        axis.scatter(prop,
            to_plot1,
            color=plt.cm.get_cmap(palette).colors[0],
            s=3.)
        axis.legend(labels=[name1])
    else:
        val = np.concatenate((np.concatenate([prop,prop])[:,None],
            np.concatenate([to_plot1,to_plot2])[:,None],
            np.concatenate((np.zeros((len(prop),1)), np.ones((len(prop),1))), axis=0)), axis=1)
        rng = np.random.default_rng()
        rng.shuffle(val)
        sc = axis.scatter(val[:,0],
            val[:,1],
            c=val[:,2],
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list(palette, plt.cm.get_cmap(palette).colors[0:2]),
            s=3.,
            alpha=0.5)
        # axis.scatter(prop,
        #     to_plot1,
        #     color=plt.cm.get_cmap(palette).colors[0],
        #     s=3.,
        #     alpha=0.5)
        # axis.scatter(prop,
        #     to_plot2,
        #     color=plt.cm.get_cmap(palette).colors[1],
        #     s=3.,
        #     alpha=0.5)
        axis.legend(handles=sc.legend_elements()[0], labels=[name1, name2])

    #Deal with labels
    axis.set(title=title,
        xscale='log' if logx else 'linear',
        yscale='log' if logy else 'linear',
        xlabel=propname,
        ylabel='')
    if logx:
        xticks = np.logspace(math.log(min_prop,base), math.log(max_prop,base), num=nxticks, endpoint=True, base=base)
        xticklabels = ["%.1e"%a for a in xticks]
    else:
        xticks = np.linspace(0, max_prop, num=nxticks)
        xticklabels = ["%.1e"%a for a in np.linspace(0, max1, num=nticks-1, endpoint=True)]
    if logy:
        loglabels = np.logspace(math.log(global_min,base), math.log(global_max,base), num=nyticks-1, endpoint=True, base=base)
        yticks = np.concatenate((loglabels, [global_inf]))
        yticklabels = ["%.1e"%a for a in yticks[:-1]] + ["∞"]
    else:
        yticks = np.concatenate((np.linspace(0, global_max, num=nyticks-1), [global_inf]))
        yticklabels = ["%.1e"%a for a in np.linspace(0, max2, num=nyticks-1, endpoint=True)] + ["∞"]
    axis.set_xticks(xticks,xticklabels)
    axis.set_yticks(yticks,yticklabels)
    axis.margins(x=0.005, y=0.01)

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
            # For now, no special treatment of inf in plot
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


