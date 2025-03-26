#!/usr/bin/env python

import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import precip_data_processors as pdp
import precip_plotting_utilities as pputils
import sys
import utilities as utils
import xarray as xr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_dt_str",
                        help = "Start date of time period over which to plot data (format YYYYmmdd.HH)")
    parser.add_argument("end_dt_str",
                        help = "End date of time period over which to plot data (format YYYYmmdd.HH)")
    parser.add_argument("-g", "--grid", dest = "grid", default = "native",
                        help = "Which grid the data to plot is on; default 'native', as in the native CONUS404 LC grid")
    parser.add_argument("-t", "--temporal_res", dest = "temporal_res", type = int, default = 24,
                        help = "Temporal resolution of the data to plot; default 24")
    parser.add_argument("-p", "--proj", dest = "proj", default = "PlateCarree",
                        help = "Projection to use for plotting CONUS404 data at native spatial resolution; default PlateCarree")
    args = parser.parse_args()

    # Configure input directory
    input_dir = f"CONUS404.{args.grid.title()}Grid.{args.temporal_res:02d}_hour_precipitation" 
    input_dir = os.path.join(utils.data_nc_dir, input_dir)
    if not(os.path.exists(input_dir)):
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    print(f"Input directory: {input_dir}")

    # Construct daily datetime list (which will correspond to cadence of input netCDF files)
    start_dt = dt.datetime.strptime(args.start_dt_str, "%Y%m%d.%H")
    end_dt = dt.datetime.strptime(args.end_dt_str, "%Y%m%d.%H")
    valid_daily_dt_list = pdp.construct_daily_datetime_list(start_dt, end_dt)

    # Construct list of CONUS404 netCDF files to read into xarray Dataset
    input_file_list = []
    for dtime in valid_daily_dt_list:
        fname = f"CONUS404.{args.grid.title()}Grid.{args.temporal_res:02d}_hour_precipitation.{dtime:%Y%m%d}.nc"
        fpath = os.path.join(input_dir, fname)
        if (not os.path.exists(fpath)):
            print(f"Warning: Input file path {fpath} does not exist; not including in input file list")
            continue
        input_file_list.append(fpath)
        #print(f"Adding input file {fpath} to input file list")
    
    if (len(input_file_list) == 0):
        print(f"Error: No files found in input directory {input_dir}")
        sys.exit(1)

    print("Reading data") 
    ds = xr.open_mfdataset(input_file_list)
    da = ds[f"precipitation_{args.temporal_res:02d}_hour"]

    # Select data in specific date range
    data_to_plot = da.loc[f"{start_dt:%Y-%m-%d %H:%M:%S}":f"{end_dt:%Y-%m-%d %H:%M:%S}"]

    # Plot data
    sparse_cbar_ticks = False
    if (args.grid.lower() == "native"):
        print("Plotting CONUS404 data on native grid")
        plot_conus404(data_to_plot, plot_name = "CONUS404.NativeGrid", proj = args.proj,
                      region = "CONUS", temporal_res = args.temporal_res,
                      sparse_cbar_ticks = sparse_cbar_ticks) 
    else:
        print(f"Plotting CONUS404 data on {args.grid.title()} grid")
        pputils.plot_cmap_single_panel(data_to_plot, f"CONUS404.{args.grid.title()}Grid",
                                       "CONUS", temporal_res = args.temporal_res, use_contourf = True,
                                       sparse_cbar_ticks = sparse_cbar_ticks)

    return da, data_to_plot 

def plot_conus404(data_to_plot, plot_name = "native", proj = "PlateCarree", region = "CONUS", temporal_res = 24, sparse_cbar_ticks = False):
    # Plot data on a map
    match proj:
        case "LambertConformal":
            map_proj = ccrs.LambertConformal()
            data_proj = ccrs.PlateCarree()
        case _:
            map_proj = ccrs.PlateCarree()
            data_proj = ccrs.PlateCarree()

    for i, idt in enumerate(data_to_plot["period_end_time"].values):
        valid_dt = pd.Timestamp(idt)
        print(f"Plotting {temporal_res}-hour precipitation data ending at {valid_dt:%Y%m%d.%H}")

        # Configure plot
        plt.figure(figsize = pputils.regions_info_dict[region].figsize_sp)
        axis = plt.axes(projection = map_proj)
        axis.coastlines()
        axis.set_extent(pputils.regions_info_dict[region].region_extent, crs = data_proj)
        gl = axis.gridlines(crs = data_proj, color = "gray", alpha = 0.5, draw_labels = True,
                            linewidth = 0.5, linestyle = "dashed")
        axis.add_feature(cfeature.BORDERS)
        axis.add_feature(cfeature.STATES)
        plt.title(f"CONUS404 {data_to_plot.short_name} ending at {valid_dt:%Y%m%d.%H}; {proj} projection", size = 15)

        # Plot data
        levels = pputils.variable_plot_limits(utils.accum_precip_var_name, temporal_res = temporal_res)
        p = axis.contourf(data_to_plot["lon"], data_to_plot["lat"], data_to_plot[i,:,:], transform = data_proj,
                          extend = "both", cmap = "viridis", levels = levels)
        
        # Configure colorbar 
        plt.colorbar(p, orientation = "vertical", shrink = 0.7)
        p.colorbar.ax.set_yticks(levels)
        if sparse_cbar_ticks:
            cbar_tick_labels = pputils.create_sparse_cbar_ticks(levels)
        else:
            cbar_tick_labels = levels
        p.colorbar.ax.set_yticklabels(cbar_tick_labels)
        p.colorbar.ax.tick_params(labelsize = 15)
        p.colorbar.set_label(f"{data_to_plot.short_name} [{data_to_plot.units}]", size = 15)

        # Save figure
        plt.tight_layout()
        formatted_short_name = pdp.format_short_name(data_to_plot)
        fig_name = f"cmap.{plot_name}.{formatted_short_name}.{proj}.{valid_dt:%Y%m%d.%H}.{region}.png"
        fig_fpath = os.path.join("/home/bbasarab/plots", fig_name)
        print(f"Saving {fig_fpath}")
        plt.savefig(fig_fpath)

if __name__ == "__main__":
    da, data_to_plot = main()
