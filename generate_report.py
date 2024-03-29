## Script to generate a bunch of plots from the singular value CSV file.

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
import argparse
import shutil
import fpdf
from fpdf import FPDF

fpdf.set_global("FPDF_CACHE_MODE", 1)


def generate_report(data1, data2, folder1, folder2, name1, name2, output_folder, produce_scatter, remeshed=False, comp_artist=False):
    is_comparison = (data2 is not None)
    plot_folder = os.path.join(output_folder, "plots")
    interesting_mesh_folder = os.path.join(plot_folder, "interesting_meshes")
        
    pdf = FPDF(orientation="P", unit="pt", format="letter")
    pdf.add_page()
    
    font_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'utilities', 'assets', 'fonts')
    pdf.add_font('Libertine', '', os.path.join(font_dir, "LinLibertine_Rah.ttf"), uni=True)
    pdf.set_font('Libertine','', 24)

    pdf.set_title("Benchmark Report")
    pdf.set_x(0)
    pdf.set_y(24)
    pdf.write(24, "Benchmark Report")
    pdf.set_x(0)
    pdf.set_y(48)
    pdf.set_font_size(18)
    if is_comparison:
        pdf.write(18, name1 + " vs " + name2)
    else:
        pdf.write(18, name1)
        
    def print_stats(data,title):
        pdf.set_x(0)
        pdf.set_y(80)
        pdf.set_font_size(15)
        pdf.write(15, "Statistics for " + title)

        pdf.set_x(0)
        pdf.set_y(100)
        pdf.set_font_size(9)
        
        def without_nans(arr):
            return arr[np.where(np.logical_not(np.isnan(arr)))]
        

        max_area_distortion = np.array(data.max_area_distortion).astype(float)
        max_area_distortion = without_nans(max_area_distortion)
        avg_non_inf_max_area_distortion = np.mean(max_area_distortion[np.where(np.logical_not(np.isinf(max_area_distortion)))])

        max_angle_distortion = np.array(data.max_angle_distortion).astype(float)
        max_angle_distortion = without_nans(max_angle_distortion)
        avg_non_inf_max_angle_distortion = np.mean(max_angle_distortion[np.where(np.logical_not(np.isinf(max_angle_distortion)))])
        
        average_area_discrepancy = np.array(data.average_area_discrepancy).astype(float)
        proportion_failures = np.sum(np.isnan(average_area_discrepancy))/len(average_area_discrepancy)
        average_area_discrepancy = without_nans(average_area_discrepancy)
        
        average_angle_discrepancy = np.array(data.average_angle_discrepancy).astype(float)
        average_angle_discrepancy = without_nans(average_angle_discrepancy)
        
        percent_flipped = 100. * np.array(data.proportion_flipped_triangles).astype(float)
        percent_flipped = without_nans(percent_flipped)
        
        if not remeshed:
            artist_correlation = np.array(data.artist_correlation).astype(float)
            artist_correlation = without_nans(artist_correlation)
                
        if remeshed:
            stats1 = [
                ("Avg Non-Inf Max Area Distortion", avg_non_inf_max_area_distortion),
                ("Avg Area Discrepancy", np.mean(average_area_discrepancy)),
                ("Avg Percentage Flipped Triangles", np.mean(percent_flipped)),
                ("Avg Non-Inf Max Angle Distortion", avg_non_inf_max_angle_distortion),
                ("Avg Angle Error", np.mean(average_angle_discrepancy)),
                ("Proportion of Parameterization Failures", proportion_failures)
            ]
        else:
            stats1 = [
                ("Avg Non-Inf Max Area Distortion", avg_non_inf_max_area_distortion),
                ("Avg Area Discrepancy", np.mean(average_area_discrepancy)),
                ("Avg Percentage Flipped Triangles", np.mean(percent_flipped)),
                ("Avg Non-Inf Max Angle Distortion", avg_non_inf_max_angle_distortion),
                ("Avg Angle Discrepancy", np.mean(average_angle_discrepancy)),
                ("Avg Artist Correlation (small = more corr.)", np.mean(artist_correlation)),
                ("Proportion of Parameterization Failures", proportion_failures)
            ]
            
        if not data.cut:
            mesh_cut_length = np.array(data.mesh_cut_length).astype(float)
            mesh_cut_length = without_nans(mesh_cut_length)
            
            artist_mesh_cut_length_match = np.array(data.artist_mesh_cut_length_match).astype(float)
            artist_mesh_cut_length_match = without_nans(artist_mesh_cut_length_match)
            
            stats1.extend([
                ("Avg Mesh Cut Length", np.mean(mesh_cut_length)),
                ("Avg Artist Mesh Cut Length Match", np.mean(artist_mesh_cut_length_match))
            ])
         
        def n_sig_figs(x, n):
            if x == 0:
                return 0
            
            magnitude = int(math.floor(math.log10(abs(x))))
            return round(x, -magnitude + n - 1)
                
        pdf.set_x(0)
        pdf.set_y(100)
        pdf.set_font_size(9)
        pdf.write(9, "\n".join([stat[0] for stat in stats1]))
        pdf.set_left_margin(pdf.l_margin + 160)
        pdf.set_x(0)
        pdf.set_y(100)
        pdf.write(9, "\n".join([str(n_sig_figs(stat[1], 6)) for stat in stats1]))
        
    print_stats(data1,name1)
    
    if is_comparison:
        pdf.set_left_margin(306)
        print_stats(data2,name2)
        
    pdf.set_left_margin(28.35)
    
    pdf.image(os.path.join(plot_folder, "percentage_flipped.png"), \
              x=28.35, y=200, w=555.3, h = 222.12, type = '', link = '')

    #Put all the histograms here
    
    if is_comparison:
        pdf.image(os.path.join(plot_folder, "resolution.png"), \
                  x=28.35, y=425, w=555.3, h = 222.12, type = '', link = '')       
    
    if produce_scatter:
        pdf.add_page()
        if is_comparison:
            pdf.image(os.path.join(plot_folder, "symmetric_dirichlet_energy_scatter_comp.png"), \
                      x=28.35, y=30, w=555.3, h = 222.12, type = '', link = '')
            pdf.image(os.path.join(plot_folder, "max_angle_distortion_scatter_comp.png"), \
                      x=28.35, y=255, w=555.3, h = 222.12, type = '', link = '')
            pdf.image(os.path.join(plot_folder, "average_angle_discrepancy_scatter_comp.png"), \
                      x=28.35, y=480, w=555.3, h = 222.12, type = '', link = '')
            
            pdf.add_page()
            pdf.image(os.path.join(plot_folder, "max_area_distortion_scatter_comp.png"), \
                      x=28.35, y=50, w=555.3, h = 222.12, type = '', link = '')
            pdf.image(os.path.join(plot_folder, "average_area_discrepancy_scatter_comp.png"), \
                      x=28.35, y=275, w=555.3, h = 222.12, type = '', link = '')
        else:
            pdf.image(os.path.join(plot_folder, "symmetric_dirichlet_energy_scatter.png"), \
                      x=28.35, y=30, w=555.3, h = 222.12, type = '', link = '')
            pdf.image(os.path.join(plot_folder, "max_angle_distortion_scatter.png"), \
                      x=28.35, y=255, w=555.3, h = 222.12, type = '', link = '')
            pdf.image(os.path.join(plot_folder, "average_angle_discrepancy_scatter.png"), \
                      x=28.35, y=480, w=555.3, h = 222.12, type = '', link = '')
            
            pdf.add_page()
            pdf.image(os.path.join(plot_folder, "max_area_distortion.png"), \
                      x=28.35, y=30, w=555.3, h = 222.12, type = '', link = '')
            pdf.image(os.path.join(plot_folder, "average_area_discrepancy.png"), \
                      x=28.35, y=255, w=555.3, h = 222.12, type = '', link = '')

    if not remeshed:
        if produce_scatter:
            pdf.image(os.path.join(plot_folder, "artist_correlation.png"), \
                      x=28.35, y=495, w=555.3, h = 222.12, type = '', link = '')
        else:
            pdf.add_page()
            pdf.image(os.path.join(plot_folder, "artist_correlation.png"), \
                      x=28.35, y=50, w=555.3, h = 222.12, type = '', link = '')
        
    if not data1.cut:
        pdf.add_page()
        pdf.image(os.path.join(plot_folder, "mesh_cut_length.png"), \
                  x=28.35, y=50, w=555.3, h = 222.12, type = '', link = '')
        pdf.image(os.path.join(plot_folder, "artist_mesh_cut_length_match.png"), \
                  x=28.35, y=275, w=555.3, h = 222.12, type = '', link = '')

    
    interesting_mesh_files = sorted([(f,) + tuple(f.split("__")) for f in os.listdir(interesting_mesh_folder) \
                                     if f.endswith(".png")])
    interesting_mesh_files = [(f, " ".join(reason.split("_")), filename, " ".join(prop.split("_"))) \
                              for f, reason, filename, prop in interesting_mesh_files]
    
    
    if len(interesting_mesh_files) > 0:
        pdf.add_page()

        pdf.set_x(0)
        pdf.set_y(24)
        pdf.set_font_size(18)
        pdf.write(18, "Interesting Meshes")
    
        for i, (f, reason, filename, prop) in enumerate(interesting_mesh_files):
            first_page = i // 2 == 0
            y = 50 if not first_page else 75
            y = y if i % 2 == 0 else y + 225
            
            if i % 2 == 0 and not first_page:
                pdf.add_page()
                
            pdf.image(os.path.join(interesting_mesh_folder, f), x=28.34, y=y, w=555.3, h=222.12, type="", link="")
        
    
    pdf.output(os.path.join(output_folder, "benchmark_report.pdf"))
    

def selected_plots(folder1,
    folder2=None,
    name1='dataset 1',
    name2='dataset 2',
    produce_scatter=True,
    out_dir='.',
    comp_artist=False):
    
    #Read in the CSV files to create a namespace
    data1 = read_csv(os.path.join(folder1, "distortion_characteristics.csv"))
    data2 = None if folder2==None else read_csv(os.path.join(folder2, "distortion_characteristics.csv"))

    remeshed = any(data1.remeshed) if data2==None else (any(data1.remeshed) or any(data2.remeshed))

    if data2 != None:
        assert data1.object_id == data2.object_id
        assert remeshed or data1.nfaces == data2.nfaces
        assert remeshed or data1.nvertices == data2.nvertices
        assert data1.cut == data2.cut


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
    data1_flipped = data1.proportion_flipped_triangles
    if data2==None:
        axis = hist(data1_flipped,
            name1,
            title='Percentage of flipped triangles',
            comment='(reporting smaller of flipped and 100%-flipped, failed parametrizations are ∞)',
            percentx=True)
    else:
        data2_flipped = data2.proportion_flipped_triangles
        axis = hist(data1_flipped,
            name1,
            data2_flipped,
            name2,
            title='Percentage of flipped triangles',
            comment='(reporting smaller of flipped and 100%-flipped, failed parametrizations are ∞)',
            percentx=True)
    percentage_flipped_path = os.path.join(out_dir, 'percentage_flipped.')
    plt.savefig(percentage_flipped_path+'pdf')
    plt.savefig(percentage_flipped_path+'png', dpi=300)
    plt.close()

    def make_graphs_for_prop(prop,
        title,
        produce_scatter=True,
        plot_data2=True,
        logx=True):

        plot_2 = plot_data2 if data2 is not None else False

        if plot_2:
            axis = hist(getattr(data1, prop),
                name1,
                getattr(data2, prop),
                name2,
                title=title,
                comment='(failed parametrizations are ∞)',
                logx=logx,
                zero_bin=False,
                inf_bin=True)
        else:
            axis = hist(getattr(data1, prop),
                name1,
                title=title,
                comment='(failed parametrizations are ∞)',
                logx=logx,
                zero_bin=False,
                inf_bin=True)
        hist_path = os.path.join(out_dir, f'{prop}.')
        plt.savefig(hist_path + 'pdf')
        plt.savefig(hist_path + 'png', dpi=300)
        plt.close()

        if produce_scatter:
            if plot_2:
                axis = scatter_comparison(getattr(data1, prop),
                    name1,
                    getattr(data2, prop),
                    name2,
                    title=title,
                    logx=logx,
                    logy=logx)
                scatter_comp_path = os.path.join(out_dir, f'{prop}_scatter_comp.')
                plt.savefig(scatter_comp_path + 'pdf')
                plt.savefig(scatter_comp_path + 'png', dpi=300)
                plt.close()

    #Make plots for all properties
    make_graphs_for_prop('max_angle_distortion',
        'Maximal angle distortion',
        produce_scatter=produce_scatter)
    make_graphs_for_prop('average_angle_discrepancy',
        'Average angle discrepancy',
        produce_scatter=produce_scatter)
    make_graphs_for_prop('max_area_distortion',
        'Maximal area distortion',
        produce_scatter=produce_scatter)
    make_graphs_for_prop('average_area_discrepancy',
        'Average area discrepancy',
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
    make_graphs_for_prop('symmetric_dirichlet_energy',
        'Symmetric Dirichlet Energy',
        produce_scatter=produce_scatter)
    if not remeshed:
        make_graphs_for_prop('artist_correlation',
            'How much does per-triangle distortion correlate to the artist? (smaller is more correlated)',
            produce_scatter=produce_scatter,
            plot_data2=not comp_artist,
            logx=False)
        
    if not data1.cut:
        make_graphs_for_prop('mesh_cut_length',
                             'Cut length along mesh',
                             produce_scatter=produce_scatter)
        make_graphs_for_prop('artist_mesh_cut_length_match',
                             'How much worse is the cut length along the mesh compared to the artist? (smaller is better)',
                             produce_scatter=produce_scatter,
                             plot_data2=not comp_artist)
            
    
    return data1, data2, remeshed

def interesting_meshes(folder1,
    folder2,
    name1 = 'dataset 1',
    name2 = 'dataset 2',
    out_dir = '.',
    comp_artist=False):

    #Read in the CSV files to create a namespace
    meshes1 = read_interesting_meshes_csv(os.path.join(folder1, "triangle_singular_values.csv"))
    meshes2 = None if folder2==None else read_interesting_meshes_csv(os.path.join(folder2, "triangle_singular_values.csv"))
   
    meshes1.sort(key = lambda m: m.filename)
    
    if meshes2 is not None:
        meshes2.sort(key = lambda m: m.filename)
        new_meshes1 = []
        new_meshes2 = []
        excess_meshes1 = []
        excess_meshes2 = []
        i = 0
        j = 0
        while i < len(meshes1) and j < len(meshes2):
            if meshes1[i].filename == meshes2[j].filename:
                if meshes1[i].triangle_number == meshes2[j].triangle_number:
                    new_meshes1.append(meshes1[i])
                    new_meshes2.append(meshes2[j])
                else:
                    excess_meshes2.append(meshes2[j])
                i += 1
                j += 1
            elif meshes1[i].filename < meshes2[j].filename:
                excess_meshes1.append(meshes1[j])
                i += 1
            else:
                excess_meshes2.append(meshes2[j])
                j += 1
                
        meshes1 = new_meshes1
        meshes2 = new_meshes2
        
        assert len(meshes1) == len(meshes2)

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

    def make_interesting_graph_for_prop(prop,
        title):
        plot_2 = (meshes2 is not None)
        if plot_2:
            for mesh1,mesh2 in zip(meshes1,meshes2):
                if mesh1.reason == "Handpicked":
                    plot_title = mesh1.filename + " - " + title
                else:
                    plot_title = "Worst " + mesh1.reason + ": " + mesh1.filename +  " - " + title
                axis = scatter_comparison(getattr(mesh1, prop),
                    name1,
                    getattr(mesh2, prop),
                    name2,
                    title=plot_title)
                
                reason_e = "_".join(mesh1.reason.split(" "))
                title_e = "_".join(title.split(" " ))
                filename_e = mesh1.filename.split(".")[0]
                
                scatter_comp_path = os.path.join(out_dir, reason_e + "__" + filename_e + "__" + title_e + ".")
                plt.savefig(scatter_comp_path + 'pdf')
                plt.savefig(scatter_comp_path + 'png', dpi=300)
                plt.close()
            if comp_artist:
                for mesh in excess_meshes1:
                    if mesh.reason == "Handpicked":
                        plot_title = mesh.filename + " - " + title
                    else:
                        plot_title = "Worst " + mesh.reason + ": " + mesh.filename + " - " + title
                    
                    axis = hist(getattr(mesh, prop),
                    name1,
                    title=plot_title,
                    comment='(failed parametrizations are ∞)',
                    logx=True,
                    zero_bin=False,
                    inf_bin=True)
                    
                    reason_e = "_".join(mesh.reason.split(" "))
                    title_e = "_".join(title.split(" " ))
                    filename_e = mesh.filename.split(".")[0]
                    
                    hist_path = os.path.join(out_dir, reason_e + "__" + filename_e + "__" + title_e + ".")
                    plt.savefig(hist_path + 'pdf')
                    plt.savefig(hist_path + 'png', dpi=300)
                    plt.close()
        else:
            for mesh1 in meshes1:
                if mesh1.reason == "Handpicked":
                    plot_title = mesh1.filename + " - " + title
                else:
                    plot_title = "Worst " + mesh1.reason + ": " + mesh1.filename + " - " + title
                    
                axis = hist(getattr(mesh1, prop),
                name1,
                title=plot_title,
                comment='(failed parametrizations are ∞)',
                logx=True,
                zero_bin=False,
                inf_bin=True)
                
                reason_e = "_".join(mesh1.reason.split(" "))
                title_e = "_".join(title.split(" " ))
                filename_e = mesh1.filename.split(".")[0]
                    
                hist_path = os.path.join(out_dir, reason_e + "__" + filename_e + "__" + title_e + ".")
                plt.savefig(hist_path + 'pdf')
                plt.savefig(hist_path + 'png', dpi=300)
                plt.close()

    make_interesting_graph_for_prop('singular_value_1', 'Singular Value 1')
    make_interesting_graph_for_prop('singular_value_2', 'Singular Value 2')

    return None


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
    smallestlognum = 1e-1
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
    zeroexists = False
    def handle_logheights(bins):
        if logy:
            hts = [smallestlognum if x<smallestlognum else x for x in bins]
            if smallestlognum in hts:
                zeroexists = True
            return hts
        else:
            if 0 in bins:
                zeroexists = True
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
            # Move further to left for double bars
            zerobins1 = [zerooffset - bar_width/2]
            zeroheights1 = handle_logheights([len(zeros1)])
            # Place second bar to the right of first bar
            zerobins2 = [zerooffset + bar_width/2]
            zeroheights2 = handle_logheights([len(zeros2)])

        # Label 0 bin at appropriate x-axis coord
        labelcoords.append(zerooffset)
        labeltexts.append("0%" if percentx else "0")

        # Plot the 0 bars
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
            # Always start the logspace at 0 to show how far you are from perfect
            histbins = np.logspace(np.log10(zero_cutoff),
                np.log10(plotmax),
                num=rnbins+1,
                endpoint=True)
        else:
            # Always start the linpace at 0 to show how far you are from perfect
            histbins = np.linspace(zero_cutoff,
                plotmax,
                num=rnbins+1,
                endpoint=True)
        hist1,edges1 = np.histogram(mids1, bins=histbins)
        heights1 = handle_logheights(hist1.tolist())
    else:
        # Set up where the bar pairs are in plot x-space
        bins1 = [x - bar_width/2 for x in range(1,rnbins+1)]
        bins2 = [x + bar_width/2 for x in range(1,rnbins+1)]
        if logx:
            # Always start the logspace at 0 to show how far you are from perfect
            histbins = np.logspace(np.log10(zero_cutoff),
                np.log10(plotmax),
                num=rnbins+1,
                endpoint=True)
        else:
            # Always start the linpace at 0 to show how far you are from perfect
            histbins = np.linspace(zero_cutoff,
                plotmax,
                num=rnbins+1,
                endpoint=True)
        hist1,edges1 = np.histogram(mids1, bins=histbins)
        heights1 = handle_logheights(hist1.tolist())
        hist2,edges2 = np.histogram(mids2, bins=histbins)
        heights2 = handle_logheights(hist2.tolist())
        assert np.linalg.norm(edges1 - edges2) < 1e-8
        
    # Label left and right hand side of each bar pair
    labelcoords += [x for x in np.linspace(0.5, rnbins+0.5, num=rnbins+1)]
    if percentx:
        if logx or plotmax<0.05:
            # Use scientific notation
            labeltexts += [">0%" if zero_bin else "0%"] + ["%.1e%%"%(x*100.) for x in edges1[1:]]
        else:
            # Do not use scientific notation
            labeltexts += [">0%" if zero_bin else "0%"] + ["{0:.0%}".format(x) for x in edges1[1:]]
    else:
        # Not percentages, use scientific notation
        labeltexts += [">0" if zero_bin else "0"] + ["%.1e"%x for x in edges1[1:]]

    # Plot our middle bars
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

        # Plot the infinity bars
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
    axlo, axhi = axis.get_ylim()
    if zero_bin:
        # Draw from next to zero bin, a line from axlo to axhi (always draw some line even if axhi is low)
        axis.plot([0.5*(zerooffset+1.), 0.5*(zerooffset+1.)], [axlo,max(2.*axlo,1.2*axhi)],
            linestyle='dashed',
            dashes=(10,5),
            linewidth=0.5,
            color=(0.2,0.2,0.2))
    if inf_bin:
        # Draw from next to inf bin, a line from axlo to axhi (always draw some line even if axhi is low)
        axis.plot([0.5*(infoffset+rnbins), 0.5*(infoffset+rnbins)], [axlo,max(2.*axlo,1.2*axhi)],
            linestyle='dashed',
            dashes=(10,5),
            linewidth=0.5,
            color=(0.2,0.2,0.2))
    if zeroexists:
        axis.margins(y=0.)
    else:
        axis.margins(y=0.02)

    #Deal with labels
    axis.set(title=title,
        xscale='linear',
        yscale='log' if logy else 'linear',
        xlabel=comment,
        ylabel='count')
    axis.set_xticks(labelcoords, labels=labeltexts)

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
    smallestlognum = 1e-8
    def handle_extremes(data,log):
        cmax = non_none_max(data)
        cmin = non_none_min(data)
        if cmin==None:
            cmin = smallestlognum
        if cmax <= cmin + 1e-3:
            # Must be at least some small diff between max and min
            cmax = cmin + 1e-3
        if log:
            smallest_tick = max(cmin, smallestlognum)
            # Add inf tick one tick to the right of maximum number
            inf_tick = base ** ((math.log(cmax,base)-math.log(smallest_tick,base))/nticks + math.log(cmax,base))
        else:
            # Add inf tick one tick to the right of maximum number
            inf_tick = cmax*(1. + 1./nticks)
        lo = [x for x in data if x<math.inf]
        hi = [inf_tick for x in data if x==math.inf]
        data = lo + hi
        if log:
            if smallestlognum > cmin:
                # If smallest number is really small, label as 0 and clamp plot to start at smallestlognum
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
        # No hue data provided, normal scatter
        axis.scatter(to_plot1,
            to_plot2,
            s=3.,
            cmap=palette,
            alpha=0.25)
    else:
        max_hue = max(huedata)
        min_hue = min(huedata)
        if(loghue):
            # Clamp to minimum log number
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
    
    # Set up labels and coords for both axes
    if logx:
        loglabels = np.logspace(math.log(min1,base), math.log(max1,base), num=nticks-1, endpoint=True, base=base)
        xticks = np.concatenate((loglabels, [inf1]))
        xticklabels = ["%.1e"%minlabel1] + ["%.1e"%a for a in loglabels[1:]] + ["∞"]
    else:
        xticks = np.concatenate((np.linspace(0,  max1, num=nticks-1), [inf1]))
        xticklabels = ["%.1e"%a for a in np.linspace(0, max1, num=nticks-1, endpoint=True)] + ["∞"]
    if logy:
        loglabels = np.logspace(math.log(min2,base), math.log(max2,base), num=nticks-1, endpoint=True, base=base)
        yticks = np.concatenate((loglabels, [inf2]))
        yticklabels = ["%.1e"%minlabel2] + ["%.1e"%a for a in loglabels[1:]] + ["∞"]
    else:
        yticks = np.concatenate((np.linspace(0, max2, num=nticks-1), [inf2]))
        yticklabels = ["%.1e"%a for a in np.linspace(0, max2, num=nticks-1, endpoint=True)] + ["∞"]
    axis.set_xticks(xticks,labels=xticklabels)
    axis.set_yticks(yticks,labels=yticklabels)
    axis.minorticks_off()
    # Margins scales to the data range provided, will delete bins with no data.
    axis.margins(x=0.005, y=0.02)

    axlo = axis.get_ylim()[0]
    axhi = axis.get_ylim()[1]

    bmin = max(min1,min2) if logx else 0.
    bmax = min(max1,max2)
    if bmax > 1.2*bmin + 1e-4:
        axis.plot([bmin, bmax], [bmin, bmax],
            linestyle='dashed',
            dashes=(10,5),
            linewidth=0.5,
            alpha=0.5,
            color=(0.,0.,0.))
        label = axis.text(bmax*0.95, bmax*0.95, 'x=y', horizontalalignment='right', fontstyle='italic')
        label.set_bbox(dict(boxstyle="square,pad=0.2",facecolor=(1.,1.,1.,0.5),edgecolor=(0.,0.,0.,0.5)))

    # Undo auto scaling
    if axis.get_ylim()[0]>yticks[0]:
        axis.set_ylim(bottom=yticks[0])
    if axis.get_ylim()[1]<yticks[1]:
        axis.set_ylim(top=yticks[1])
    if axis.get_xlim()[0]>xticks[0]:
        axis.set_xlim(left=xticks[0])
    if axis.get_xlim()[1]<xticks[1]:
        axis.set_xlim(right=xticks[1])

    return axis


#Read in data from CSV
def read_csv(path):

    data = SimpleNamespace()
    data.cut = True
    data.object_id = []
    data.nfaces = []
    data.nvertices = []
    data.filename = []
    data.max_area_distortion = []
    data.average_area_discrepancy = []
    data.min_singular_value = []
    data.max_singular_value = []
    data.proportion_flipped_triangles = []
    data.max_angle_distortion = []
    data.average_angle_discrepancy = []
    data.resolution = []
    data.artist_correlation = []
    data.remeshed = []
    data.symmetric_dirichlet_energy = []
    
    current_object_id = 0

    read_first_row = False
    with open(path, newline='') as parsing:
        reader = csv.reader(parsing)
        for row in reader:
            if not read_first_row:
                assert len(row) == 14 or len(row) == 16
                read_first_row = True
                #Make sure all columns have the expected headers
                assert row[0] == 'Filename'
                assert row[1] == 'Faces'
                assert row[2] == 'Vertices'
                assert row[3] == 'Max Area Distortion'
                assert row[4] == 'Average Area Discrepancy'
                assert row[5] == 'Min Singular Value'
                assert row[6] == 'Max Singular Value'
                assert row[7] == 'Proportion Flipped Triangles'
                assert row[8] == 'Max Angle Distortion'
                assert row[9] == 'Average Angle Discrepancy'
                assert row[10] == 'Resolution'
                assert row[11] == 'Artist Correlation'
                assert row[12] == 'Remeshed'
                assert row[13] == 'Symmetric Dirichlet Energy'
                
                if len(row) == 16:
                    data.cut = False
                    assert row[14] == "Mesh Cut Length"
                    assert row[15] == "Artist Mesh Cut Length Match"
                    data.mesh_cut_length = []
                    data.artist_mesh_cut_length_match = []
            else:
                if data.cut:
                    assert len(row) == 14
                else:
                    assert len(row) == 16
                #Given the namings above, put everything into the correct array.
                data.object_id.append(current_object_id)
                data.filename.append(row[0])
                data.nfaces.append(to_int(row[1]))
                data.nvertices.append(to_int(row[2]))
                data.max_area_distortion.append(to_float(row[3]))
                data.average_area_discrepancy.append(to_float(row[4]))
                data.min_singular_value.append(to_float(row[5]))
                data.max_singular_value.append(to_float(row[6]))
                data.proportion_flipped_triangles.append(to_float(row[7]))
                data.max_angle_distortion.append(to_float(row[8]))
                data.average_angle_discrepancy.append(to_float(row[9]))
                data.resolution.append(to_float(row[10]))
                data.artist_correlation.append(to_float(row[11]))
                data.remeshed.append(to_bool(row[12]))
                data.symmetric_dirichlet_energy.append(to_float(row[13]))
                
                current_object_id += 1
                
                if not data.cut:
                    data.mesh_cut_length.append(to_float(row[14]))
                    data.artist_mesh_cut_length_match.append(to_float(row[15]))

    return data

def read_interesting_meshes_csv(path):

    datas = []
    current_filename_reason = None
    read_first_row = False
    with open(path, newline='') as parsing:
        reader = csv.reader(parsing)
        for row in reader:
            assert len(row) == 5
            if not read_first_row:
                read_first_row = True
                #Make sure all columns have the expected headers
                assert row[0] == 'Filename'
                assert row[1] == 'Triangle Number'
                assert row[2] == 'Singular Value 1'
                assert row[3] == 'Singular Value 2'
                assert row[4] == 'Reason'
            else:
                #Given the namings above, put everything into

                filename = row[0]
                reason = row[4]
                if (filename, reason) != current_filename_reason:
                    current_filename_reason = (filename, reason)
                    data = SimpleNamespace()
                    data.object_id = []
                    current_object_id = 0
                    data.filename = filename
                    data.triangle_number = []
                    data.singular_value_1 = []
                    data.singular_value_2 = []
                    data.reason = reason
                    datas.append(data)
                    
                assert datas[-1].filename == filename
                assert datas[-1].reason == row[4]
                datas[-1].object_id.append(current_object_id)
                datas[-1].triangle_number.append(to_int(row[1]))
                datas[-1].singular_value_1.append(to_float(row[2]))
                datas[-1].singular_value_2.append(to_float(row[3]))
                
                current_object_id += 1

    return datas


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
def to_bool(val):
    if val == 'True':
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate report from benchmark")
    parser.add_argument("-b", "--benchmarks", type=str, required=True, nargs="+", dest="benchmark_folders", metavar="INPUT_FOLDER")
    parser.add_argument("-n", "--names", type=str, required=True, nargs="+", dest="names", metavar="NAME")
    parser.add_argument("-o", "--output", type=str, dest="output_folder", default="report_output")
    parser.add_argument("--force-no-comp", dest="force_no_comp", action="store_true")
    
    parser.set_defaults(force_no_comp=False)
    
    args = parser.parse_args()
    benchmark_folders = args.benchmark_folders
    if len(benchmark_folders) > 2:
        raise ValueError("Can only compare two benchmark folders")
        
    benchmark_folders = [os.path.abspath(folder) for folder in benchmark_folders]
    
    if len(benchmark_folders) == 1:
        folder1 = benchmark_folders[0]
        folder2 = None
    else:
        folder1, folder2 = benchmark_folders
    
    names = args.names
    if len(names) != len(benchmark_folders):
        raise ValueError("Must provide same number of benchmark folders and names")
     
    if len(names) == 1:
        name1 = names[0]
        name2 = None
    else:
        name1, name2 = names
        
    output_folder = os.path.abspath(args.output_folder)
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    else:
        shutil.rmtree(output_folder)
        os.mkdir(output_folder)
        
    plot_folder = os.path.join(output_folder, "plots")
    interesting_meshes_folder = os.path.join(plot_folder, "interesting_meshes")
    os.mkdir(plot_folder)
    os.mkdir(interesting_meshes_folder)
    
    if folder2 is not None or args.force_no_comp:
        data1, data2, remeshed = selected_plots(folder1=folder1, folder2=folder2, name1=name1, name2=name2, \
                                      produce_scatter=True, out_dir=plot_folder)

        interesting_meshes(folder1=folder1, folder2=folder2, name1=name1, name2=name2, out_dir=interesting_meshes_folder)

        generate_report(data1=data1, data2=data2, folder1=folder1, folder2=folder2, 
                        name1=name1, name2=name2, output_folder=output_folder, produce_scatter=True,
                        remeshed=remeshed)
    else:
        folder2 = os.path.join(folder1, "artist")
        name2 = "Artist"
        data1, data2, remeshed = selected_plots(folder1=folder1, folder2=folder2, name1=name1, name2=name2, \
                                      produce_scatter=True, out_dir=plot_folder, comp_artist=True)

        interesting_meshes(folder1=folder1, folder2=folder2, name1=name1, name2=name2, out_dir=interesting_meshes_folder, comp_artist=True)

        generate_report(data1=data1, data2=data2, folder1=folder1, folder2=folder2, 
                        name1=name1, name2=name2, output_folder=output_folder, produce_scatter=True,
                        remeshed=remeshed, comp_artist=True)
