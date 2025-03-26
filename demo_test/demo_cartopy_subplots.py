#!/usr/bin/env python

import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import precip_data_processors
import sys
import utilities as utils
import xarray as xr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--regions_list", dest = "regions_list", nargs = "+",
                        help = "List of regions to set map extents; provide four (space-separated) to make nice 2x2 subplots. \
                        If exactly four are not provided, all the regions in region_extent_mapper will be plotted.")
    args = parser.parse_args()

    full_regions_list = list(precip_data_processors.region_extents.keys())
    if (args.regions_list is None) or (len(args.regions_list) > 1):
        regions_list = full_regions_list 
    else:
        regions_list = []
        for region in args.regions_list:
            if (region not in full_regions_list):
                regions_list = full_regions_list
                break
            regions_list.append(region)

    num_regions = len(regions_list)
    if (num_regions == 1):
        num_sp_columns = 1
        num_sp_rows = 1
        region = regions_list[0]
    if (num_regions == 4):
        num_sp_columns = 2
        num_sp_rows = 2
    else:
        num_sp_columns = 2
        num_sp_rows = 4
    regions_list = regions_list[:8] # Limit to at most 8 subplots for now

    if (num_regions > 1): 
        # Set up the subplots figure and plot data
        proj = ccrs.PlateCarree() 
        fig, axes = plt.subplots(num_sp_rows, num_sp_columns, figsize = (20, 20), subplot_kw = {"projection": proj})
        # print(axes, type(axes), axes.shape, axes.flatten().shape)
        for axis, region in zip(axes.flatten(), regions_list): 
            axis.coastlines()
            axis.set_extent(precip_data_processors.region_extents[region], crs = proj)
            axis.add_feature(cfeature.BORDERS)
            if ("US" in region):
                axis.add_feature(cfeature.STATES)
            gl = axis.gridlines(crs = proj, draw_labels = False, color = "gray", alpha = 0.5,
                                linewidth = 0.5, linestyle = "dashed")
            axis.set_title(region, fontsize = 20)
            #axis.label_outer()
        fig.suptitle("Region extents defined in precip_data_processors", fontsize = 20)
        fig.tight_layout(pad = 2.0)
    else:
        # Single plot
        proj = ccrs.PlateCarree()
        plt.figure(figsize = (15, 10)) 
        ax = plt.axes(projection = proj) 
        ax.coastlines()
        ax.set_extent(precip_data_processors.region_extents[region], crs = proj)
        ax.add_feature(cfeature.BORDERS)
        if ("US" in region):
            ax.add_feature(cfeature.STATES)
        gl = ax.gridlines(crs = proj, draw_labels = True, color = "gray", alpha = 0.5,
                            linewidth = 0.5, linestyle = "dashed")
        #gl.ylabels_right = False 
        #print(gl.xlabels_top)
        plt.title("Region extents defined in precip_data_processors", fontsize = 20)
        plt.tight_layout(pad = 2.0)

    # Save figure
    fig_name = "test_plot_region_extents.png"
    fig_path = os.path.join("/home/bbasarab/plots", fig_name)
    print(f"Saving {fig_path}")
    plt.savefig(fig_path)

if __name__ == "__main__":
    main()
