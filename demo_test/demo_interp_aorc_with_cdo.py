#!/usr/bin/env python

import argparse
import dataclasses
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import precip_data_processors as pdp
import precip_plotting_utilities as pputils
import sys
import utilities as utils
import xarray as xr

# Interpolate AORC data to global Replay grid
    
# Selected Replay data already output to netCDF to use as template file for interpolation
nc_dir = "/data/bbasarab/netcdf"
nc_testing_dir = os.path.join(nc_dir, "testing")
template_fpath = os.path.join(nc_dir, "TemplateGrids", "ReplayGrid.nc")

def main():
    utils.suppress_warnings()
    parser = argparse.ArgumentParser(description = "Interpolate AORC data using cdo command line utility; plot native and interpolated data")
    parser.add_argument("dt_str", 
                        help = "Date to process; format YYYYmm (full month of stats) or YYYYmmdd (one day of stats)")
    parser.add_argument("-t", "--temporal_res", dest = "temporal_res", type = int, default = 24,
                        help = "Temporal resolution of data to interpolate in hours; default 24")
    parser.add_argument("-i", "--interp", dest = "interp", action = "store_true", default = False,
                        help = "Set to run interpolation")
    parser.add_argument("-c", "--cdfs", dest = "cdfs", action = "store_true", default = False,
                        help = "Set to create and plot CDFs")
    parser.add_argument("-p", "--plot", dest = "plot", action = "store_true", default = False,
                        help = "Set to make contour maps to visualize differences between interpolated datasets")
    parser.add_argument("--exclude_zeros", dest = "exclude_zeros", action = "store_true", default = False,
                        help = "Set to exclude zeros from CDFs and stats")
    args = parser.parse_args()

    print(f"***** TIME PERIOD: {args.dt_str}")
    main_dir = "/data/bbasarab/netcdf/testing"

    # Native
    fpath_str = os.path.join(main_dir, f"AORC*NativeGrid*{args.dt_str}*.nc")
    print(f"Reading {fpath_str}")
    precip_native = rename_dims(utils.convert_from_dask_array(xr.open_mfdataset(fpath_str).accum_precip))
    precip_native.attrs["grid_size"] = utils.aorc_grid_cell_size

    # Bilinear
    fpath_str = os.path.join(main_dir, f"AORC*bilinear*{args.dt_str}*.nc")
    print(f"Reading {fpath_str}")
    precip_bil = utils.convert_from_dask_array(xr.open_mfdataset(fpath_str).accum_precip)
    precip_bil.attrs["grid_size"] = utils.replay_grid_cell_size

    # Conservative
    fpath_str = os.path.join(main_dir, f"AORC*conservative*{args.dt_str}*.nc")
    print(f"Reading {fpath_str}")
    precip_con = utils.convert_from_dask_array(xr.open_mfdataset(fpath_str).accum_precip)
    precip_con.attrs["grid_size"] = utils.replay_grid_cell_size
    
    # Nearest
    fpath_str = os.path.join(main_dir, f"AORC*nearest*{args.dt_str}*.nc")
    print(f"Reading {fpath_str}")
    precip_nbr = utils.convert_from_dask_array(xr.open_mfdataset(fpath_str).accum_precip)
    precip_nbr.attrs["grid_size"] = utils.replay_grid_cell_size

    # Mask data to CONUS for easier comparison to output from precip_data_processors.py
    print("Masking data to CONUS")
    conus_mask_native_grid = pputils.create_conus_mask(precip_native)
    conus_mask_replay_grid = pputils.create_conus_mask(precip_bil)
    precip_native = precip_native.where(conus_mask_native_grid)
    precip_bil = precip_bil.where(conus_mask_replay_grid)
    precip_con = precip_con.where(conus_mask_replay_grid)
    precip_nbr = precip_nbr.where(conus_mask_replay_grid)

    data_dict_tmp = {"Native": precip_native, "Bilinear": precip_bil, "Conservative": precip_con, "Nearest": precip_nbr}

    # Exclude zeros
    if args.exclude_zeros:
        data_dict_new = {} 
        print("Excluding zeros")
        plot_name = f"AORC_interp_methods exclude zeros: {args.temporal_res:02d}-hour precip, {args.dt_str}"
        fig_name = f"AORC_interp_methods.exclude_zeros.{args.temporal_res:02d}_hour_precipitation.{args.dt_str}"
        ylims = [0.0, 1.0] 
        for data_name, da in data_dict_tmp.items():
            data_dict_new[data_name] = da.where(da > 0.0) 
        data_dict = data_dict_new
    else:
        plot_name = f"AORC_interp_methods: {args.temporal_res:02d}-hour precip, {args.dt_str}"
        fig_name = f"AORC_interp_methods.{args.temporal_res:02d}_hour_precipitation.{args.dt_str}" 
        ylims = [0.75, 1.0] 
        data_dict = data_dict_tmp

    # Calculate stats
    print("Calculating stats")
    for data_name, da in data_dict.items():
        print(f"***** {data_name}:")
        print(f"Min: {da.min().item()}")
        print(f"Mean: {da.mean().item()}")
        print(f"Median: {da.median().item()}")
        print(f"Max: {da.max().item()}")
        print(f"95.0th pctl: {da.quantile(0.95).item()}")
        print(f"99.0th pctl: {da.quantile(0.99).item()}")
        print(f"99.9th pctl: {da.quantile(0.999).item()}")

    # Plot CDFs
    if args.cdfs:
        print("Plotting CDFs")
        pputils.plot_precip_cdf(data_dict,
                                plot_name, 
                                fig_name, 
                                valid_dtime = None, 
                                xticks = np.arange(0, 55, 5),
                                yticks = np.arange(0.0, 1.05, 0.05),
                                xlims = [0, 50],
                                ylims = ylims, 
                                skip_nearest = False)

    # Plot contour maps
    if args.plot:
        print("Plotting contour maps")
        plot_levels = np.arange(0, 42, 2)

        # Multi-panel plot
        pputils.plot_cmap_multi_panel(data_dict, "Native", "CONUS", plot_levels = plot_levels)
   
        # Native
        pputils.plot_cmap_single_panel(precip_native,
                                       "AORC Native Grid",
                                       "AORC.NativeGrid",
                                       "CONUS",
                                       plot_levels = plot_levels) 
        # Bilinear
        pputils.plot_cmap_single_panel(precip_bil,
                                       "AORC Replay Grid Bilinear",
                                       "AORC.ReplayGrid.bilinear",
                                       "CONUS",
                                       plot_levels =  plot_levels)

        # Conservative
        pputils.plot_cmap_single_panel(precip_con,
                                       "AORC Replay Grid Conservative",
                                       "AORC.ReplayGrid.conservative",
                                       "CONUS",
                                       plot_levels = plot_levels) 

        # Nearest neighbor
        pputils.plot_cmap_single_panel(precip_nbr,
                                       "AORC Replay Grid Nearest",
                                       "AORC.ReplayGrid.nearest",
                                       "CONUS",
                                       plot_levels = plot_levels) 


    return data_dict

def calculate_24hr_accum_precip(hourly_aorc_data):
    # Calculate 24-hour precipitation from hourly precipitation
    time_step = 24
    roller = hourly_aorc_data.rolling({utils.period_end_time_dim_str: time_step})
    precip24 = roller.sum()[(time_step - 1)::time_step,:,:]
    precip24.name = utils.accum_precip_var_name
    pdp.add_attributes_to_data_array(precip24,
                                     short_name = f"24-hour precipitation",
                                     long_name = f"Precipitation accumulated over the prior 24 hour(s)",
                                     units = "mm")

    return precip24

def run_cdo_interpolation(interp_type, dtime, input_fpath, temporal_res = 24, do_interp = True):
    output_fpath = os.path.join(nc_testing_dir, f"AORC.ReplayGrid.{temporal_res:02d}_hour_precipitation.{interp_type}.{dtime:%Y%m%d}.nc")
    if not(do_interp): # Option to return output file path only without running interpolation
        return output_fpath

    cdo_interp_flag = utils.set_cdo_interpolation_type(interp_type)

    cdo_cmd = f"cdo -P 8 {cdo_interp_flag},{template_fpath} {input_fpath} {output_fpath}"
    print(f"Running: {cdo_cmd}")
    ret = os.system(cdo_cmd)
    if (ret != 0):
        print("Error: cdo interpolation command did not work")
        sys.exit(1)

    return output_fpath

def rename_dims(data_array):
    return data_array.rename({
                             "latitude": "lat",
                             "longitude": "lon",
                             }) 

if __name__ == "__main__":
    data_dict = main() 

"""current_dt = dt.datetime.strptime(args.dt_str, "%Y%m%d")
fpath_native = os.path.join("/Projects/AORC_CONUS_4km", f"{current_dt:%Y}", f"prate.aorc.{current_dt:%Y%m%d}.nc")
if not(os.path.exists(fpath_native)):
    print(f"Error: Input AORC file to interpolate {fpath_native} does not exist")
    sys.exit(1)

current_dt_p1 = current_dt + dt.timedelta(days = 1)
fpath_native_p1 = os.path.join("/Projects/AORC_CONUS_4km", f"{current_dt_p1:%Y}", f"prate.aorc.{current_dt_p1:%Y%m%d}.nc")
if not(os.path.exists(fpath_native_p1)):
    print(f"Error: Input AORC file to interpolate {fpath_native_p1} does not exist")
    sys.exit(1)

if (args.temporal_res == 1):
    ##### Interpolate 1-hour precip to Replay grid
    # Bilinear
    fpath_bil = run_cdo_interpolation("bilinear", current_dt, fpath_native, 
                                      do_interp = args.interp, temporal_res = args.temporal_res) 

    # Conservative
    fpath_con = run_cdo_interpolation("conservative", current_dt, fpath_native,
                                      do_interp = args.interp, temporal_res = args.temporal_res) 
    
    # Nearest neighbor
    fpath_nbr = run_cdo_interpolation("nearest", current_dt, fpath_native,
                                      do_interp = args.interp, temporal_res = args.temporal_res) 

    # Read netCDFs containing hourly precip data
    precip_native = rename_dims(xr.open_dataset(fpath_native).prate)
    precip_bil = xr.open_dataset(fpath_bil).prate
    precip_con = xr.open_dataset(fpath_con).prate
    precip_nbr = xr.open_dataset(fpath_nbr).prate
    
    # Mask data to CONUS for easier comparison to output from precip_data_processors.py
    conus_mask_native = pputils.create_conus_mask(precip_native)
    conus_mask_replay = pputils.create_conus_mask(precip_bil)
    precip_native = precip_native.where(conus_mask_native)
    precip_bil = precip_bil.where(conus_mask_replay)
    precip_con = precip_con.where(conus_mask_replay)
    precip_nbr = precip_nbr.where(conus_mask_replay)

    # Create final DataArray dictionary
    data_dict = {"Native": precip_native, "Bilinear": precip_bil, "Conservative": precip_con, "Nearest": precip_nbr}
   
    ##### Plot 1-hour AORC data
    if args.plot: 
        # Native grid 
        pputils.plot_cmap_single_panel(precip_native, "AORC.native", "CONUS", plot_levels = np.arange(0, 42, 2)) 

        # Bilinear 
        pputils.plot_cmap_single_panel(precip_bil, f"AORC.ReplayGrid.bilinear", "CONUS", plot_levels = np.arange(0, 42, 2))

        # Conservative 
        pputils.plot_cmap_single_panel(precip_con, f"AORC.ReplayGrid.conservative", "CONUS", plot_levels = np.arange(0, 42, 2))
    
        # Nearest
        pputils.plot_cmap_single_panel(precip_nbr, f"AORC.ReplayGrid.nearest", "CONUS", plot_levels = np.arange(0, 42, 2))
elif (args.temporal_res == 24):
    # Calculate 24-hourly AORC data. Since data are period-ending, but each file
    # contains data from 00z to 23z, we need to get two separate files to get data
    # ending at 01z (on the current day) to data ending at 00z (on the next day).
    precip = xr.open_mfdataset([fpath_native, fpath_native_p1]).prate
    precip = precip.rename({utils.time_dim_str: utils.period_end_time_dim_str})
    precip = precip.loc[f"{current_dt:%Y-%m-%d 01:00:00}":f"{current_dt_p1:%Y-%m-%d 00:00:00}"]

    # Output 24-hour precip at native spatial resolution to netCDF
    accum_precip24 = utils.convert_from_dask_array(calculate_24hr_accum_precip(precip))
    fpath_native24 = os.path.join(nc_testing_dir, f"AORC.NativeGrid.24_hour_precipitation.{current_dt:%Y%m%d}.nc")
    print(f"Writing 24-hour precip at native resolution to {fpath_native24}")
    accum_precip24.period_end_time.encoding["units"] = utils.seconds_since_unix_epoch_str
    accum_precip24.to_netcdf(fpath_native24)
    accum_precip24 = rename_dims(accum_precip24)

    ##### Interpolate 24-hour precip to Replay grid
    # Bilinear
    fpath_bil = run_cdo_interpolation("bilinear", current_dt, fpath_native24,
                                      do_interp = args.interp, temporal_res = args.temporal_res)

    # Conservative
    fpath_con = run_cdo_interpolation("conservative", current_dt, fpath_native24,
                                      do_interp = args.interp, temporal_res = args.temporal_res)
    
    # Nearest neighbor
    fpath_nbr = run_cdo_interpolation("nearest", current_dt, fpath_native24,
                                      do_interp = args.interp, temporal_res = args.temporal_res)
   
    ##### Read netCDFs containing 24-hour precip data 
    accum_precip24_bil = xr.open_dataset(fpath_bil).accum_precip
    accum_precip24_con = xr.open_dataset(fpath_con).accum_precip
    accum_precip24_nbr = xr.open_dataset(fpath_nbr).accum_precip

    # Mask data to CONUS for easier comparison to output from precip_data_processors.py
    conus_mask_native = pputils.create_conus_mask(accum_precip24)
    conus_mask_replay = pputils.create_conus_mask(accum_precip24_bil)
    accum_precip24 = accum_precip24.where(conus_mask_native)
    accum_precip24_bil = accum_precip24_bil.where(conus_mask_replay)
    accum_precip24_con = accum_precip24_con.where(conus_mask_replay)
    accum_precip24_nbr = accum_precip24_nbr.where(conus_mask_replay)

    # Create final DataArray dictionary
    data_dict = {"Native": accum_precip24, "Bilinear": accum_precip24_bil, "Conservative": accum_precip24_con, "Nearest": accum_precip24_nbr}
    
    ##### Plot 24-hour AORC data
    if args.plot:
        plot_levels = pputils.variable_plot_limits("precip", args.temporal_res)

        # Native grid 
        pputils.plot_cmap_single_panel(accum_precip24,
                                       "AORC Native Grid",
                                       "AORC.NativeGrid",
                                       "CONUS",
                                       plot_levels = plot_levels) 
        
        # Bilinear
        pputils.plot_cmap_single_panel(accum_precip24_bil,
                                       "AORC Replay Grid Bilinear",
                                       "AORC.ReplayGrid.bilinear",
                                       "CONUS",
                                       plot_levels =  plot_levels)

        # Conservative
        pputils.plot_cmap_single_panel(accum_precip24_con,
                                       "AORC Replay Grid Conservative",
                                       "AORC.ReplayGrid.conservative",
                                       "CONUS",
                                       plot_levels = plot_levels) 

        # Nearest neighbor
        pputils.plot_cmap_single_panel(accum_precip24_nbr,
                                       "AORC Replay Grid Nearest",
                                       "AORC.ReplayGrid.nearest",
                                       "CONUS",
                                       plot_levels = plot_levels) 

    ##### Plot CDFs
    if args.cdfs:
        print("Plotting CDFs")
        pputils.plot_precip_cdf(data_dict,
                                f"AORC_interp_methods: {args.temporal_res:02d}-hour precip",
                                f"AORC_interp_methods.{args.temporal_res:02d}_hour_precipitation",
                                valid_dtime = current_dt,
                                xticks = np.arange(0, 55, 5),
                                yticks = np.arange(0.75, 1.05, 0.05),
                                xlims = [0, 50],
                                ylims = [0.75, 1.0],
                                skip_nearest = True)

##### Calculate stats
for data_name, da in data_dict.items():
    print(f"***** {data_name}:")
    print(f"Min: {da.min().item()}")
    print(f"Mean: {da.mean().item()}")
    print(f"Median: {da.median().item()}")
    print(f"Max: {da.max().item()}")
    print(f"95.0th pctl: {da.quantile(0.95).item()}")
    print(f"99.0th pctl: {da.quantile(0.99).item()}")
    print(f"99.9th pctl: {da.quantile(0.999).item()}")
"""
