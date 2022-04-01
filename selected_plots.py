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
        # assert data1.nfaces == data2.nfaces
        # assert data1.nvertices == data2.nvertices
        data1.nfaces = data2.nfaces
        data1.nvertices = data2.nvertices

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
    data1_flipped = [None if x==None else min(1.-x,x) for x in data1.percentage_flipped_triangles]
    if data2==None:
        axis = hist(data1_flipped,
            name1,
            title='Percentage of flipped triangles',
            comment='(reporting smaller of flipped and 100%-flipped, failed parametrizations are ∞)',
            percentx=True)
    else:
        data2_flipped = [None if x==None else min(1.-x,x) for x in data2.percentage_flipped_triangles]
        axis = hist(data1_flipped,
            name1,
            data2_flipped,
            name2,
            title='Percentage of flipped triangles',
            comment='(reporting smaller of flipped and 100%-flipped, failed parametrizations are ∞)',
            percentx=True)
    percentage_flipped_path = os.path.join(out_dir, 'percentage_flipped.pdf')
    plt.savefig(percentage_flipped_path)
    plt.close()

    #Bijectivity area violation histogram
    # if data2==None:
    #     axis = hist(data1.bijectivity_violation_area,
    #         name1,
    #         logx=True,
    #         title='Overlapping area in UV map',
    #         comment='(failed parametrizations are ∞)',
    #         zero_bin=False)
    # else:
    #     axis = hist(data1.bijectivity_violation_area,
    #         name1,
    #         data2.bijectivity_violation_area,
    #         name2,
    #         logx=True,
    #         title='Overlapping area in UV map',
    #         comment='(failed parametrizations are ∞)',
    #         zero_bin=False)
    # bijectivity_violation_path = os.path.join(out_dir, 'bijectivity_violation.pdf')
    # plt.savefig(bijectivity_violation_path)
    # plt.close()

    def make_graphs_for_prop(prop,
        title,
        produce_scatter=True,
        plot_data2=True):

        plot_2 = plot_data2 
        if data2==None:
            plot_2 = False

        if plot_2:
            axis = hist(getattr(data1, prop),
                name1,
                getattr(data2, prop),
                name2,
                title=title,
                comment='(failed parametrizations are ∞)',
                logx=True,
                zero_bin=False,
                inf_bin=True)
        else:
            axis = hist(getattr(data1, prop),
                name1,
                title=title,
                comment='(failed parametrizations are ∞)',
                logx=True,
                zero_bin=False,
                inf_bin=True)
        percentage_flipped_path = os.path.join(out_dir, 'percentage_flipped.pdf')
        plt.savefig(percentage_flipped_path)
        plt.close()

        if produce_scatter:
            if plot_2:
                axis = scatter_comparison(getattr(data1, prop),
                    name1,
                    getattr(data2, prop),
                    name2,
                    huedata=data1.nfaces,
                    huename="#faces",
                    title=title)
                scatter_comp_path = os.path.join(out_dir, f'{prop}_scatter_comp.pdf')
                plt.savefig(scatter_comp_path)
                plt.close()

            # if not plot_2:
            #     axis = scatter_vs_property(data1.nfaces,
            #         "#faces",
            #         getattr(data1, prop),
            #         name1,
            #         title=title)
            # else:
            #     axis = scatter_vs_property(data1.nfaces,
            #         "#faces",
            #         getattr(data1, prop),
            #         name1,
            #         getattr(data2, prop),
            #         name2,
            #         title=title)
            # scatter_path = os.path.join(out_dir, f'{prop}_scatter.pdf')
            # plt.savefig(scatter_path)
            # plt.close()


    #Make plots for all properties
    make_graphs_for_prop('max_angle_distortion',
        'Maximal angle distortion',
        produce_scatter=produce_scatter)
    make_graphs_for_prop('total_angle_distortion',
        'Total angle distortion',
        produce_scatter=produce_scatter)
    make_graphs_for_prop('max_area_distortion',
        'Max area distortion',
        produce_scatter=produce_scatter)
    make_graphs_for_prop('total_area_distortion',
        'Total area distortion',
        produce_scatter=produce_scatter)
    make_graphs_for_prop('min_singular_value',
        'Min. singular value',
        produce_scatter=produce_scatter)
    make_graphs_for_prop('max_singular_value',
        'Max. singular value',
        produce_scatter=produce_scatter)
    make_graphs_for_prop('resolution',
        'Pixel resolution needed for display',
        produce_scatter=produce_scatter)
    make_graphs_for_prop('artist_area_match',
        'Matching the artist\'s area distortion',
        produce_scatter=produce_scatter,
        plot_data2=False)
    make_graphs_for_prop('artist_angle_match',
        'Matching the artist\'s angle distortion',
        produce_scatter=produce_scatter,
        plot_data2=False)




                # assert row[4] == 'Max Area Distortion'
                # assert row[6] == 'Min Singular Value'
                # assert row[7] == 'Max Singular Value'
                # assert row[8] == 'Percentage Flipped Triangles'
                # assert row[9] == 'Bijectivity Violation Area'
                # assert row[11] == 'Total Angle Distortion'
                # assert row[12] == 'Resolution'
                # assert row[13] == 'Artist Area Match'
                # assert row[14] == 'Artist Angle Match'

    # data = SimpleNamespace()
    # data.object_id = []
    # data.nfaces = []
    # data.nvertices = []
    # data.filename = []
    # data.max_area_distortion = []
    # data.total_area_distortion = []
    # data.min_singular_value = []
    # data.max_singular_value = []
    # data.percentage_flipped_triangles = []
    # data.bijectivity_violation_area = []
    # data.max_angle_distortion = []
    # data.total_angle_distortion = []
    # data.resolution = []
    # data.artist_area_match = []
    # data.artist_angle_match = []

# Histogram with data
def hist(data1,
    name1,
    data2=None,
    name2=None,
    nbins=10,
    title='',
    comment='',
    zero_bin=True,
    inf_bin=True,
    logx=False,
    logy=True,
    percentx=False,
    palette='Pastel1'):
    
    bar_width = 0.4
    zero_cutoff = 1e-8
    smallestlognum = zero_cutoff
    if percentx:
        inf_cutoff = 1.
    else:
        inf_cutoff = 1e18

    assert all(x==None or x>=0 for x in data1)
    assert data2==None or all(x==None or x>=0 for x in data2)
    assert nbins>=3
    rnbins = nbins - (zero_bin==True) - (inf_bin==True)

    def preprocess_zeros_infs(data):
        xn = [math.inf if x==None else x for x in data]

        zeros = [zero_cutoff for x in xn if x<=zero_cutoff]
        mids = [x for x in xn if x>zero_cutoff and x<=inf_cutoff]
        infs = [inf_cutoff for x in xn if x>inf_cutoff]

        if not zero_bin:
            mids = zeros + mids
            zeros = None
        if not inf_bin:
            mids = mids + infs
            infs = None

        return zeros,mids,infs
    def handle_logheights(bins):
        if logx:
            return [smallestlognum if x<smallestlognum else x for x in bins]
        else:
            return bins

    #Find out which bins to plot
    zeros1,mids1,infs1 = preprocess_zeros_infs(data1)
    plotmax = max(mids1) if len(mids1)>0 else 2*zero_cutoff
    if data2!=None:
        zeros2,mids2,infs2 = preprocess_zeros_infs(data2)
        plotmax = max(plotmax,max(mids2)) if len(mids2)>0 else plotmax


    labelcoords = []
    labeltexts = []
    plt.figure(figsize=(10,4), tight_layout=True)
    axis = plt.gca()

    #Zero bin
    zerooffset = -0.5
    if zero_bin:
        if data2==None:
            zerobins1 = [zerooffset]
            zeroheights1 = handle_logheights([len(zeros1)])
        else:
            zerobins1 = [zerooffset - bar_width/2]
            zeroheights1 = handle_logheights([len(zeros1)])
            zerobins2 = [zerooffset + bar_width/2]
            zeroheights2 = handle_logheights([len(zeros2)])

        labelcoords.append(zerooffset)
        labeltexts.append("0%" if percentx else "0")

        axis.bar(zerobins1,
            zeroheights1,
            color=plt.cm.get_cmap(palette).colors[0],
            width=bar_width)
        if data2!=None:
            axis.bar(zerobins2,
                zeroheights2,
                color=plt.cm.get_cmap(palette).colors[1],
                width=bar_width)

    #Normal bins
    if data2==None:
        bins1 = [x for x in range(1,rnbins+1)]
        if logx:
            histbins = np.logspace(np.log10(zero_cutoff),
                np.log10(inf_cutoff),
                num=rnbins+1,
                endpoint=True)
        else:
            histbins = np.linspace(zero_cutoff,
                inf_cutoff,
                num=rnbins+1,
                endpoint=True)
        hist1,edges1 = np.histogram(mids1, bins=histbins)
        heights1 = handle_logheights([x for x in hist1])
    else:
        bins1 = [x - bar_width/2 for x in range(1,rnbins+1)]
        bins2 = [x + bar_width/2 for x in range(1,rnbins+1)]
        if logx:
            histbins = np.logspace(np.log10(zero_cutoff),
                np.log10(plotmax),
                num=rnbins+1,
                endpoint=True)
        else:
            histbins = np.linspace(zero_cutoff,
                plotmax,
                num=rnbins+1,
                endpoint=True)
        hist1,edges1 = np.histogram(mids1, bins=histbins)
        heights1 = handle_logheights([x for x in hist1])
        hist2,edges2 = np.histogram(mids2, bins=histbins)
        heights2 = handle_logheights([x for x in hist2])
        assert np.linalg.norm(edges1 - edges2) < 1e-8
    labelcoords += [x for x in np.linspace(0.5, rnbins+0.5, num=rnbins+1)]
    if percentx:
        if logx or plotmax<0.05:
            labeltexts += [">0%" if zero_bin else "0%"] + ["%.1e%%"%(x*100.) for x in edges1[1:]]
        else:
            labeltexts += [">0%" if zero_bin else "0%"] + ["{0:.0%}".format(x) for x in edges1[1:]]
    else:
        labeltexts += [">0" if zero_bin else "0"] + ["%.1e"%x for x in edges1[1:]]

    bar1 = axis.bar(bins1,
        heights1,
        color=plt.cm.get_cmap(palette).colors[0],
        width=bar_width)
    if data2!=None:
        bar2 = axis.bar(bins2,
            heights2,
            color=plt.cm.get_cmap(palette).colors[1],
            width=bar_width)

    #Infinity bins
    infoffset = rnbins + 1.5
    if inf_bin:
        if data2==None:
            infbins1 = [infoffset]
            infheights1 = handle_logheights([len(infs1)])
        else:
            infbins1 = [infoffset - bar_width/2]
            infheights1 = handle_logheights([len(infs1)])
            infbins2 = [infoffset + bar_width/2]
            infheights2 = handle_logheights([len(infs2)])

        labelcoords.append(infoffset)
        labeltexts.append("∞")

        axis.bar(infbins1,
            infheights1,
            color=plt.cm.get_cmap(palette).colors[0],
            width=bar_width)
        if data2!=None:
            axis.bar(infbins2,
                infheights2,
                color=plt.cm.get_cmap(palette).colors[1],
                width=bar_width)

    #Legend
    if data2==None:
        axis.legend([bar1],[name1], loc='upper center')
    else:
        axis.legend([bar1,bar2],[name1,name2], loc='upper center')

    #Draw dotted lines
    axlo = axis.get_ylim()[0]
    axhi = axis.get_ylim()[1]
    if zero_bin:
        axis.plot([0.5*(zerooffset+1.), 0.5*(zerooffset+1.)], [axlo,1.2*axhi],
            linestyle='dashed',
            dashes=(10,5),
            linewidth=0.5,
            color=(0.2,0.2,0.2))
    if inf_bin:
        axis.plot([0.5*(infoffset+rnbins), 0.5*(infoffset+rnbins)], [axlo,1.2*axhi],
            linestyle='dashed',
            dashes=(10,5),
            linewidth=0.5,
            color=(0.2,0.2,0.2))
    if zero_bin or inf_bin:
        axis.margins(y=0)

    #Deal with labels
    axis.set(title=title,
        xscale='linear',
        yscale='log' if logy else 'linear',
        xlabel=comment,
        ylabel='count')
    axis.set_xticks(labelcoords,labeltexts)

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
        m = 1.
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
            smallest_tick = max(cmin, smallestlognum)
            inf_tick = base ** ((math.log(cmax,base)-math.log(smallest_tick,base))/nticks + math.log(cmax,base))
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

#Scatter vs property plot
def scatter_vs_property(prop,
    propname,
    data1,
    name1,
    data2=None,
    name2=None,
    title='',
    nxticks=10,
    nyticks=6,
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
        m = 1.
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
    def handle_extremes(data,log,nticks,inf_provided=None,relabel_infs=True):
        cmax = non_none_max(data)
        cmin = non_none_min(data)
        if inf_provided==None:
            if log:
                cmax = non_none_max(data)
                cmin = non_none_min(data)
                if log:
                    smallest_tick = max(cmin, smallestlognum)
                    inf_tick = base ** ((math.log(cmax,base)-math.log(smallest_tick,base))/nticks + math.log(cmax,base))
                else:
                    inf_tick = cmax*(1. + 1./nticks)
            else:
                inf_tick = cmax*(1. + 1./nticks)
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
    def preproc(data,log,nticks,inf_provided=None,relabel_infs=True):
        return handle_extremes(remove_nones(data),log,nticks,
            inf_provided=inf_provided,relabel_infs=relabel_infs)

    if data2==None:
        to_plot1,min1,max1,minlabel1,inf1 = preproc(data1,logx,nxticks)
        global_min,global_minlabel = min1,minlabel1
        global_max,global_inf = max1,inf1
    if data2!=None:
        to_plot1,min1,max1,minlabel1,inf1 = preproc(data1,logx,nxticks,relabel_infs=False)
        to_plot2,min2,max2,minlabel2,inf2 = preproc(data2,logy,nyticks,relabel_infs=False)
        if min1<min2:
            global_min,global_minlabel = min1,minlabel1
        else:
            global_min,global_minlabel = min2,minlabel2
        if max1>max2:
            global_max,global_inf = max1,inf1
        else:
            global_max,global_inf = max2,inf2
        to_plot1,_,_,_,_ = preproc(data1,logx,nxticks,relabel_infs=True,inf_provided=global_inf)
        to_plot2,_,_,_,_ = preproc(data2,logy,nyticks,relabel_infs=True,inf_provided=global_inf)

    max_prop = max(prop)
    min_prop = min(prop)

    plt.figure(figsize=(10,4), tight_layout=True)
    axis = plt.gca()
    if data2==None:
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
        yticklabels = ["%.1e"%global_minlabel] + ["%.1e"%a for a in yticks[1:-1]] + ["∞"]
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
        return max(f,0) #TODO: Remove
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


