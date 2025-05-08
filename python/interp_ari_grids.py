#!/usr/bin/env python

import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import precip_plotting_utilities as pputils
import sys
import utilities as utils
import xarray as xr

def main():
    parser = argparse.ArgumentParser(description = "Interpolate ARI data from Stage IV grid using cdo command line utility")
    parser.add_argument("--ari", dest = "ari", type = int, default = 2,
                        help = "ARI; default 2 years")
    parser.add_argument("--duration", dest = "duration", type = int, default = 24,
                        help = "Duration; default 24 hours")
    parser.add_argument("-i", "--interp_type", dest = "interp_type", default = "bilinear",
                        help = "CDO interpolation type to use; default bilinear")
    parser.add_argument("-d", "--dest_grid", dest = "dest_grid", default = "Replay", choices = ["Replay", "0.1deg", "0.25deg"],
                        help = "Destination grid type; default Global Replay grid")
    args = parser.parse_args()

    # Selected Replay data already output to netCDF to use as template file for interpolation
    nc_dir = "/data/bbasarab/netcdf"
    template_fpath = os.path.join(nc_dir, "TemplateGrids", "ReplayGrid.nc")

    # Set interpolation type to be used by cdo
    cdo_interp_type = set_cdo_interpolation_type(args.interp_type) 

    # Configure input file path
    input_fpath = determine_ari_input_fpath(args.ari, args.duration) 

    # Configure output file path
    output_dir = os.path.join(nc_dir, f"ARIs.{args.dest_grid}Grid")
    output_file = f"ARI.{args.dest_grid}Grid.{args.ari:02d}_year.{args.duration:02d}_hour_precipitation.nc"
    output_fpath = os.path.join(output_dir, output_file)

    # Configure cdo interpolation command
    # Regrid CONUS404 to 0.1 degree rectilinear grid
    if (args.dest_grid == "0.1deg"):
        cdo_cmd = f"cdo -P 8 {cdo_interp_type},global_0.1 {input_fpath} {output_fpath}"
        plot_name = "0p1deg_grid"
        print("**** Interpolating to 0.1 degree grid")
    # Regrid CONUS404 to 0.25 degree rectilinear grid
    elif (args.dest_grid == "0.25deg"):
        cdo_cmd = f"cdo -P 8 {cdo_interp_type},global_0.25 {input_fpath} {output_fpath}"
        plot_name = "0p25deg_grid" 
        print("**** Interpolating to 0.25 degree grid")
    # Regrid CONUS404 to Replay grid, using Replay data itself as the template
    else:
        cdo_cmd = f"cdo -P 8 {cdo_interp_type},{template_fpath} {input_fpath} {output_fpath}"
        plot_name = "ReplayGrid"
        print("**** Interpolating to Replay grid")

    # Run cdo command
    print(f"Running: {cdo_cmd}")
    ret = os.system(cdo_cmd)
    if (ret != 0):
        print("Error: cdo interpolation command did not work")
        sys.exit(1)
        
def set_cdo_interpolation_type(args_flag):
    match args_flag:
        case "bilinear": # Bilinear interpolation
            return "remapbil"
        case "conservative": # First-order conservative interpolation
            return "remapcon"
        case "conservative2": # Second-order conservative interpolation
            return "remapcon2"
        case "nearest": # Nearest-neighbor interpolation
            return "remapnn"
        case "nearest_neighbor": # Nearest-neighbor interpolation
            return "remapnn"
        case _:
            print(f"Unrecognized interpolation type {args_flag}; will perform bilinear interpolation")
            return "remapbil"

def determine_ari_input_fpath(ari, duration):
    input_dir = "/Projects/WPC_ExtremeQPF/newARIs/"
    input_file = f"allusa_ari_{ari}yr_{duration}hr_xarray_st4grid.nc"

    return os.path.join(input_dir, input_file)

if __name__ == "__main__":
    main() 
