#!/usr/bin/env python

import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import precip_data_processors as pdp
import precip_plotting_utilities as pputils
import sys
import utilities as utils
import xarray as xr

# TODO: Add ability to interpolate using xesmf (in addition to current cdo capability)

def main():
    parser = argparse.ArgumentParser(description = "Interpolate ARI data from Stage IV grid using cdo command line utility")
    parser.add_argument("--ari", dest = "ari", type = int, default = 2,
                        help = "ARI; default 2 years")
    parser.add_argument("--duration", dest = "duration", type = int, default = 24,
                        help = "Duration; default 24 hours")
    parser.add_argument("--interp_engine", dest = "interp_engine", default = "cdo", choices = ["cdo", "xesmf"],
                        help = "Interpolation engine to use; choices are cdo or xesmf")
    parser.add_argument("-i", "--interp_type", dest = "interp_type", default = "bilinear",
                        help = "CDO interpolation type to use; default bilinear")
    parser.add_argument("-d", "--dest_grid", dest = "dest_grid", default = "Replay", choices = ["Replay", "AORC", "0.1deg", "0.25deg"],
                        help = "Destination grid type; default Global Replay grid")
    parser.add_argument("-p", "--plot", dest = "plot", default = False, action = "store_true",
                        help = "Set to plot original and interpolated ARI grid")
    args = parser.parse_args()

    # Selected Replay data already output to netCDF to use as template file for interpolation
    nc_dir = "/data/bbasarab/netcdf"
    output_grid_string = pdp.set_grid_name_for_file_names(args.dest_grid)
    template_fpath = os.path.join(nc_dir, "TemplateGrids", f"{output_grid_string}.nc")

    if (args.interp_engine == "cdo"):
        # Set interpolation type to be used by cdo
        cdo_interp_type = utils.set_cdo_interpolation_type(args.interp_type) 

        # Configure input file path
        input_fpath = determine_ari_input_fpath(args.ari, args.duration) 

        # Configure output file path
        output_dir = os.path.join(nc_dir, f"ARIs.{output_grid_string}")
        output_file = f"ARI.{output_grid_string}.{args.ari:02d}_year.{args.duration:02d}_hour_precipitation.nc"
        output_fpath = os.path.join(output_dir, output_file)

        # Configure CDO interpolation command
        cdo_cmd = configure_cdo_interp_command(output_grid_string, cdo_interp_type, template_fpath,
                                               input_fpath, output_fpath)
        # Run cdo command
        print(f"Running: {cdo_cmd}")
        ret = os.system(cdo_cmd)
        if (ret != 0):
            print("Error: cdo interpolation command did not work")
            sys.exit(1)
    else:
        print(f"Error: ARI grid interpolation with xesmf not yet implemented")
        sys.exit(0)

    if args.plot:
        print("Plotting")

        # ARIs on original (native) grid
        input_da = xr.open_dataset(input_fpath).precip
        pputils.plot_cmap_single_panel(input_da,
                                       f"ARI.{args.ari:02d}yr.{args.duration:02d}hr.NativeGrid",
                                       f"ARI.{args.ari:02d}yr.{args.duration:02d}hr.NativeGrid",
                                       "CONUS",
                                       plot_levels = np.arange(0, 210, 10))

        # ARIs on output grid
        output_da = xr.open_dataset(output_fpath).precip
        pputils.plot_cmap_single_panel(output_da,
                                       f"ARI.{args.ari:02d}yr.{args.duration:02d}hr.{output_grid_string}",
                                       f"ARI.{args.ari:02d}yr.{args.duration:02d}hr.{output_grid_string}",
                                       "CONUS",
                                       plot_levels = np.arange(0, 210, 10))

def configure_cdo_interp_command(dest_grid_string, cdo_interp_type, template_fpath, input_fpath, output_fpath):
    match dest_grid_string:
        # Regrid to 0.1 degree rectilinear grid
        case "0.1deg":
            print("**** Interpolating to 0.1 degree grid")
            cdo_cmd = f"cdo -P 8 {cdo_interp_type},global_0.1 {input_fpath} {output_fpath}"
        # Regrid to 0.25 degree rectilinear grid
        case "0.25deg":
            print("**** Interpolating to 0.25 degree grid")
            cdo_cmd = f"cdo -P 8 {cdo_interp_type},global_0.25 {input_fpath} {output_fpath}"
        # Regrid to Replay grid, using Replay data itself as the template
        case "ReplayGrid":
            print("**** Interpolating to Replay grid")
            cdo_cmd = f"cdo -P 8 {cdo_interp_type},{template_fpath} {input_fpath} {output_fpath}"
        # Regrid to AORC grid, using AORC data itself as the template
        case "AorcGrid":
            print("**** Interpolating to AORC grid")
            cdo_cmd = f"cdo -P 8 {cdo_interp_type},{template_fpath} {input_fpath} {output_fpath}"
        case _:
            print(f"Error: Interpolation to {dest_grid_string} grid not implemented")
            sys.exit(1)

    return cdo_cmd
 
def determine_ari_input_fpath(ari, duration):
    input_dir = "/Projects/WPC_ExtremeQPF/newARIs/"
    input_file = f"allusa_ari_{ari}yr_{duration}hr_xarray_st4grid.nc"

    return os.path.join(input_dir, input_file)

if __name__ == "__main__":
    main() 
