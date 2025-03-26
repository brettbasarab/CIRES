#!/usr/bin/env python

import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import precip_plotting_utilities as pputils
import replay_utilities as replay_utils
import sys
import utilities as utils
import warnings
import xarray as xr

def main():
    utils.suppress_warnings()
    parser = argparse.ArgumentParser(description = "Program to plot data from the UFS GEFS13 Replay dataset")
    parser.add_argument("start_dt_str",
                        help = "Start date of time period over which to plot data (format YYYYmmdd.HH)")
    parser.add_argument("end_dt_str",
                        help = "End date of time period over which to plot data (format YYYYmmdd.HH)")
    parser.add_argument("var_name", 
                        help = "Variable name to plot") 
    parser.add_argument("--plevel", dest = "plevel", default = 500, type = int, 
                        help = "Pressure level at which to plot multi-level atmospheric fields; default 500 hPa")
    parser.add_argument("--region", dest = "region", default = "Global", 
                        help = "Region to zoom plot to; default Global")
    args = parser.parse_args()
    
    # Construct daily datetime list (which will correspond to cadence of input netCDF files)
    start_dt = parse_string_datetime_and_check_format(args.start_dt_str, dt_fmt = "%Y%m%d.%H", resolution = 3) 
    end_dt = parse_string_datetime_and_check_format(args.end_dt_str, dt_fmt = "%Y%m%d.%H", resolution = 3) 
    
    # Read in the entire Global Replay dataset (from 1994-2023) from GCP
    print("Reading Global Replay dataset from GCP")
    ds = xr.open_zarr(utils.replay_coarse_gcs_path, storage_options = {"token": "anon"})

    # Ensure variable to plot is a valid variable name, contained within the Replay xarray Dataset.
    # The windspeed variable name wspd won't be, but it's components ugrd and vgrd will.
    if (args.var_name == "wspd"):
        test_if_in_ds_var_name = "ugrd"
    else:
        test_if_in_ds_var_name = args.var_name

    if (test_if_in_ds_var_name not in ds.variables):
        print(f"Error: {args.var_name} is not a variable within the Replay dataset")
        sys.exit(1)

    # Get the correct index of the pressure level (this will only be used for vars valid on pressure levels)
    plevel_idx = -1
    plevel_idx = replay_utils.find_index_of_pressure_level(ds, args.plevel)

    da = replay_utils.convert_to_plot_units(ds, args.var_name, plevel_idx = plevel_idx) 

    # Plot data
    data_to_plot = da.loc[f"{start_dt:%Y-%m-%d %H:%M:%S}":f"{end_dt:%Y-%m-%d %H:%M:%S}"]
    plot_global_replay_native_grid(data_to_plot, args.var_name, region = args.region)

def parse_string_datetime_and_check_format(dt_str, dt_fmt = "%Y%m%d.%H", resolution = 3):
    try:
        valid_dt = dt.datetime.strptime(dt_str, dt_fmt)
    except ValueError:
        print(f"Error: Input datetime string {dt_str} does not match format {dt_fmt}")
        sys.exit(1)

    if valid_dt.hour % resolution != 0:
        print(f"Error: Dataset is {resolution}-hourly; must provide a datetime at {resolution}-hourly intervals")
        sys.exit(1)

    return valid_dt

def plot_global_replay_native_grid(data_array, var_name, region = "CONUS"):
    proj = ccrs.PlateCarree() 
    units, levels, cmap = replay_utils.map_var_name_to_plot_params(var_name)
    dtimes = [pd.Timestamp(i) for i in data_array["time"].values]

    for dtime in dtimes:
        print(f"Plotting variable {var_name} at {dtime:%Y%m%d.%H}")
        # Select data to plot (at one valid time)
        data_to_plot = data_array.sel(time = dtime)

        # Configure plot
        plt.figure(figsize = pputils.regions_info_dict[region].figsize_sp)
        axis = plt.axes(projection = proj)
        axis.coastlines()
        axis.set_extent(pputils.regions_info_dict[region].region_extent, crs = proj)
        gl = axis.gridlines(crs = proj, color = "gray", alpha = 0.5, draw_labels = True,
                            linewidth = 0.5, linestyle = "dashed")
        axis.add_feature(cfeature.BORDERS)
        axis.add_feature(cfeature.STATES)

        # Plot data
        p = data_to_plot.plot(ax = axis, levels  = levels, extend = "both", cmap = cmap,
                              cbar_kwargs = {"orientation": "vertical", "shrink": 0.7, "ticks": levels})
        
        # Configure colorbar 
        p.colorbar.ax.set_yticks(levels)
        p.colorbar.ax.set_yticklabels(levels)
        p.colorbar.ax.tick_params(labelsize = 15)
        p.colorbar.set_label(f"{var_name} [{units}]", size = 15)

        # Add title and save figure 
        plt.title(f"Global Replay {var_name} at {dtime:%Y%m%d.%H}; {region}", size = 15)
        plt.tight_layout()
        fig_name = f"cmap.GlobalReplay.{var_name}.native_grid.{dtime:%Y%m%d.%H}.{region}.png"
        fig_fpath = os.path.join(utils.plot_output_dir, fig_name)
        print(f"Saving {fig_fpath}")
        plt.savefig(fig_fpath)

if __name__ == "__main__":
    main()
