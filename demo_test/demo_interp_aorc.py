#!/usr/bin/env python

import argparse
import datetime as dt
import numpy as np
import os
import pandas as pd
import precip_data_processors as pdp
import precip_plotting_utilities as pputils
import sys
import utilities as utils
import xarray as xr
    
# Selected Replay data already output to netCDF to use as template file for interpolation
nc_dir = "/data/bbasarab/netcdf"
nc_testing_dir = os.path.join(nc_dir, "testing")
template_fpath = os.path.join(nc_dir, "TemplateGrids", "ReplayGrid.nc")

def main():
    utils.suppress_warnings()
    parser = argparse.ArgumentParser(description = "Interpolate AORC data using cdo command line utility; plot native and interpolated data")
    parser.add_argument("dt_str", 
                        help = "Date to process; format YYYYmmdd")
    parser.add_argument("-t", "--temporal_res", dest = "temporal_res", type = int, default = 24,
                        help = "Temporal resolution of data to interpolate in hours; default 24")
    parser.add_argument("-i", "--interp", dest = "interp", action = "store_true", default = False,
                        help = "Set to run interpolation")
    parser.add_argument("-p", "--plot", dest = "plot", action = "store_true", default = False,
                        help = "Set to make plots")
    args = parser.parse_args()

    current_dt = dt.datetime.strptime(args.dt_str, "%Y%m%d")

    aorc_native_fpath = os.path.join(nc_testing_dir, f"prate.aorc.{current_dt:%Y%m%d}.nc")
    if not(os.path.exists(aorc_native_fpath)):
        print(f"Error: Input AORC file to interpolate {aorc_native_fpath} does not exist")
        sys.exit(1)

    if (args.temporal_res == 1):
        # Set interpolation type to be used by cdo
        cdo_interp_type = set_cdo_interpolation_type(args.interp_type) 

        # Regrid AORC to Replay grid, using Replay data itself as the template
        aorc_output_fpath = os.path.join(nc_testing_dir, f"AORC.ReplayGrid.{args.interp_type}.{current_dt:%Y%m%d}.nc")
        cdo_cmd = f"cdo -P 8 {cdo_interp_type},{template_fpath} {aorc_native_fpath} {aorc_output_fpath}"

        # Run cdo command
        print(f"Running: {cdo_cmd}")
        ret = os.system(cdo_cmd)
        if (ret != 0):
            print("Error: cdo interpolation command did not work")
            sys.exit(1)
       
        if args.plot: 
            # Read and plot AORC data on native grid
            print(f"Reading {aorc_native_fpath} (AORC data on native grid)")
            aorc_native = xr.open_dataset(aorc_native_fpath).prate
            pputils.plot_cmap_single_panel(aorc_native, "AORC.native", "CONUS") 

            # Read and plot AORC interpolated by cdo command above
            print(f"Reading {aorc_output_fpath} (AORC data on Replay grid)")
            aorc_interp = xr.open_dataset(aorc_output_fpath).prate
            pputils.plot_cmap_single_panel(aorc_interp, f"AORC.ReplayGrid.{args.interp_type}", "CONUS")
    elif (args.temporal_res == 24):
        # First, we need to calculate 24-hourly AORC data. Since data are period-ending, but each file
        # contains data from 00z to 23z, we need to get two separate files to get data ending at 01z
        # to data ending at 00z (the next day).
        current_dt_p1 = current_dt + dt.timedelta(days = 1)
        aorc_native_fpath_p1 = os.path.join(nc_testing_dir, f"prate.aorc.{current_dt_p1:%Y%m%d}.nc")
        if not(os.path.exists(aorc_native_fpath_p1)):
            print(f"Error: Input AORC file to interpolate {aorc_native_fpath_p1} does not exist")
            sys.exit(1)

        precip = xr.open_mfdataset([aorc_native_fpath, aorc_native_fpath_p1]).prate
        precip = precip.rename({utils.time_dim_str: utils.period_end_time_dim_str})
        precip = precip.loc[f"{current_dt:%Y-%m-%d 01:00:00}":f"{current_dt_p1:%Y-%m-%d 00:00:00}"]

        # Output 24-hour precip at native spatial resolution to netCDF
        accum_precip24 = calculate_24hr_accum_precip(precip)
        fpath_native = os.path.join(nc_testing_dir, f"AORC.NativeGrid.24_hour_precipitation.{current_dt:%Y%m%d}.nc")
        print(f"Writing 24-hour precip at native resolution to {fpath_native}")
        accum_precip24.period_end_time.encoding["units"] = utils.seconds_since_unix_epoch_str
        accum_precip24.to_netcdf(fpath_native)

        ##### Interpolate 24-hour precip to Replay grid
        # Bilinear
        fpath_bil = run_cdo_interpolation("bilinear", current_dt, fpath_native) 

        # Conservative
        fpath_con = run_cdo_interpolation("conservative", current_dt, fpath_native)
        
        # Nearest neighbor
        fpath_nbr = run_cdo_interpolation("nearest", current_dt, fpath_native)
       
        ##### Read netCDFs containing interpolated data 
        accum_precip24_bil = xr.open_dataset(fpath_bil).accum_precip
        accum_precip24_con = xr.open_dataset(fpath_con).accum_precip
        accum_precip24_nbr = xr.open_dataset(fpath_nbr).accum_precip
        
        ##### Plot 24-hour AORC data
        if args.plot:
            # Native grid 
            pputils.plot_cmap_single_panel(accum_precip24, "AORC.NativeGrid", "CONUS", plot_levels = np.arange(0, 125, 5)) 
            
            # Bilinear
            pputils.plot_cmap_single_panel(accum_precip24_bil, f"AORC.ReplayGrid.bilinear", "CONUS", plot_levels = np.arange(0, 125, 5))

            # Conservative
            pputils.plot_cmap_single_panel(accum_precip24_con, f"AORC.ReplayGrid.conservative", "CONUS", plot_levels = np.arange(0, 125, 5))

            # Nearest neighbor
            pputils.plot_cmap_single_panel(accum_precip24_nbr, f"AORC.ReplayGrid.nearest", "CONUS", plot_levels = np.arange(0, 125, 5))

        ##### Calculate stats
        data_dict = {"Native": accum_precip24, "Bilinear": accum_precip24_bil, "Conservative": accum_precip24_con, "Nearest": accum_precip24_nbr}

        for data_name, da in data_dict.items():
            print(f"{data_name}:")
            print(f"Min: {np.nanmin(da.values)}")
            print(f"Max: {np.nanmax(da.values)}")
            print(f"Mean: {np.nanmean(da.values)}")
            print(f"Median: {np.median(da.values)}")
            #print(f"Min: {da.min().item()}")
            #print(f"Max: {da.max().item()}")
            #print(f"Mean: {da.mean().item()}")
            #print(f"Median: {da.median().item()}")
        
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

def set_cdo_interpolation_type(args_flag):
    match args_flag:
        case "bilinear": # Bilinear interpolation
            return "remapbil"
        case "conservative": # First-order conservative interpolation
            return "remapcon"
        case "conservative2": # Second-order conservative interpolation
            return "remapcon2"
        case "nearest_neighbor": # Nearest-neighbor interpolation
            return "remapnn"
        case _:
            print(f"Unrecognized interpolation type {args_flag}; will perform bilinear interpolation")
            return "remapbil"

def run_cdo_interpolation(interp_type, dtime, input_fpath):
    output_fpath = os.path.join(nc_testing_dir, f"AORC.ReplayGrid.24_hour_precipitation.{interp_type}.{dtime:%Y%m%d}.nc")

    cdo_interp_flag = set_cdo_interpolation_type(interp_type)

    cdo_cmd = f"cdo -P 8 {cdo_interp_flag},{template_fpath} {input_fpath} {output_fpath}"
    print(f"Running: {cdo_cmd}")
    ret = os.system(cdo_cmd)
    if (ret != 0):
        print("Error: cdo interpolation command did not work")
        sys.exit(1)

    return output_fpath

if __name__ == "__main__":
    data_dict = main() 
