#!/usr/bin/env python

import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import planetary_computer
import precip_plotting_utilities as pputils
import pystac_client
import sys
import utilities as utils
import xarray as xr

def main():
    parser = argparse.ArgumentParser(description = "Interpolate CONUS404 data using cdo command line utility; plot native and interpolated data")
    parser.add_argument("-a", "--read_azure", dest = "read_azure", default = False, action = "store_true",
                        help = "Read full CONUS404 dataset from Azure, otherwise read and interpolate existing sample netCDF files")
    parser.add_argument("-i", "--interp_type", dest = "interp_type", default = "bilinear",
                        help = "CDO interpolation type to use; default bilinear")
    parser.add_argument("-d", "--dest_grid", dest = "dest_grid", default = "Replay", choices = ["Replay", "0.1deg", "0.25deg"],
                        help = "Destination grid type; default Global Replay grid")
    args = parser.parse_args()

    # Selected Replay data already output to netCDF to use as template file for interpolation
    nc_dir = "/data/bbasarab/netcdf"
    nc_testing_dir = os.path.join(nc_dir, "testing")
    template_fpath = os.path.join(nc_dir, "TemplateGrids", "ReplayGrid.nc")

    # Set interpolation type to be used by cdo
    cdo_interp_type = set_cdo_interpolation_type(args.interp_type) 

    if not(args.read_azure):
        conus404_native_fpath = os.path.join(nc_testing_dir, "conus404_native.nc")
        if not(os.path.exists(conus404_native_fpath)):
            print(f"Error: Input CONUS404 file to interpolate {conus404_native_fpath} does not exist")
            sys.exit(1)

        # Regrid CONUS404 to 0.1 degree rectilinear grid
        if (args.dest_grid == "0.1deg"):
            conus404_output_fpath = os.path.join(nc_testing_dir, f"conus404_0p1deg_grid.{args.interp_type}.nc")
            cdo_cmd = f"cdo -P 8 {cdo_interp_type},global_0.1 {conus404_native_fpath} {conus404_output_fpath}"
            plot_name = "0p1deg_grid"
            print("**** Interpolating to 0.1 degree grid")
        # Regrid CONUS404 to 0.25 degree rectilinear grid
        elif (args.dest_grid == "0.25deg"):
            conus404_output_fpath = os.path.join(nc_testing_dir, f"conus404_0p25deg_grid.{args.interp_type}.nc")
            cdo_cmd = f"cdo -P 8 {cdo_interp_type},global_0.25 {conus404_native_fpath} {conus404_output_fpath}"
            plot_name = "0p25deg_grid" 
            print("**** Interpolating to 0.25 degree grid")
        # Regrid CONUS404 to Replay grid, using Replay data itself as the template
        else:
            conus404_output_fpath = os.path.join(nc_testing_dir, f"conus404_replay_grid.{args.interp_type}.nc")
            cdo_cmd = f"cdo -P 8 {cdo_interp_type},{template_fpath} {conus404_native_fpath} {conus404_output_fpath}"
            plot_name = "ReplayGrid"
            print("**** Interpolating to Replay grid")

        # Run cdo command
        print(f"Running: {cdo_cmd}")
        ret = os.system(cdo_cmd)
        if (ret != 0):
            print("Error: cdo interpolation command did not work")
            sys.exit(1)
        
        # Read and plot CONUS404 data on native grid
        print(f"Reading {conus404_native_fpath} (CONUS404 data on native grid)")
        conus404_native = xr.open_dataset(conus404_native_fpath).accum_precip
        pputils.plot_cmap_single_panel(conus404_native, "CONUS404.native", "CONUS") 

        # Read and plot CONUS404 interpolated by cdo command above
        print(f"Reading {conus404_output_fpath}")
        conus404_interp = xr.open_dataset(conus404_output_fpath).accum_precip # Replay grid
        pputils.plot_cmap_single_panel(conus404_interp, f"CONUS404.{plot_name}.{args.interp_type}", "CONUS")

    if args.read_azure:
        # Read data from zarr archive
        conus404_ds = read_conus404_dataset_from_azure()
        conus404_da = conus404_ds.PREC_ACC_NC
        start_dt = dt.datetime(2018,9,13,1)
        end_dt = dt.datetime(2018,9,20,0)
        accum_precip_data_array = conus404_da.loc[f"{start_dt:%Y-%m-%d %H:%M:%S}":f"{end_dt:%Y-%m-%d %H:%M:%S}"]
        accum_precip_data_array = accum_precip_data_array.rename({utils.time_dim_str: utils.period_end_time_dim_str})

        # Calculate 24-hour precipitation from hourly precipitation
        time_step = 24
        roller = accum_precip_data_array.rolling({utils.period_end_time_dim_str: time_step})
        precip24 = roller.sum()[(time_step - 1)::time_step,:,:]
        precip24.name = utils.accum_precip_var_name

        # Output 24-hour precipitation to netCDF
        native_fname = "conus404_native.20180914-20180920.nc"
        native_fpath = os.path.join(nc_testing_dir, native_fname) 
        precip24.to_netcdf(native_fpath)

        # Regrid 24-hour precipitation on native CONUS404 grid to Replay grid
        output_fname = f"conus404_replay_grid.20180914-20180920.{args.interp_type}.nc"
        output_fpath = os.path.join(nc_testing_dir, output_fname)
        cmd = f"cdo -P 8 {cdo_interp_type},{template_fpath} {native_fpath} {output_fpath}"
        os.system(cmd)

        # Plotting
        conus404_native = xr.open_dataset(native_fpath).accum_precip
        conus404_replay_grid = xr.open_dataset(output_fpath).accum_precip 
        pputils.plot_cmap_single_panel(conus404_native, f"CONUS404.native.{args.interp_type}", "CONUS") 
        pputils.plot_cmap_single_panel(conus404_replay_grid, f"CONUS404.replay_grid.{args.interp_type}", "CONUS")

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

def read_conus404_dataset_from_azure():
    print(f"Reading CONUS404 data from Azure into xarray Dataset")
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                        modifier = planetary_computer.sign_inplace)

    c = catalog.get_collection("conus404")
    asset = c.assets["zarr-abfs"]

    conus404_ds = xr.open_zarr(asset.href,
                               storage_options = asset.extra_fields["xarray:storage_options"],
                               **asset.extra_fields["xarray:open_kwargs"],
                               )

    return conus404_ds

if __name__ == "__main__":
    main() 
