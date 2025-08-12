#!/usr/bin/env python

import argparse
import cf_xarray as cfxr
import matplotlib.pyplot as plt
import numpy as np
import precip_data_processors as pdp
import precip_plotting_utilities as pputils
import os
import sys
import utilities as utils
import xarray as xr
import xesmf

utils.suppress_warnings()
template_grids_dir = os.path.join(utils.data_nc_dir, "TemplateGrids")
regridders_dir = os.path.join(utils.data_nc_dir, "Regridders")
native_grid_str = pdp.set_grid_name_for_file_names("native")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dt_str",
                        help = "Input date/time str; format YYYYmmdd")
    parser.add_argument("--ds_name", dest = "ds_name", default = "CONUS404",
                        choices = utils.dataset_list, 
                        help = "Dataset name; default CONUS404")
    parser.add_argument("--output_grid", dest = "output_grid", default = "Replay",
                        help = "Output grid name; default Replay")
    parser.add_argument("--interp_method", dest = "interp_method", default = "conservative",
                        choices = ["conservative", "bilinear"],
                        help = "Interplation method; default conservative")
    parser.add_argument("--read_regridder", dest = "read_regridder", action = "store_true", default = False,
                        help = "Set to read existing regridder info from netCDF file; otherwise build regridder (calculate weights, etc.)")
    parser.add_argument("--temporal_res", dest = "temporal_res", type = int, default = 24,
                        help = "Temporal resolution of output (interpolated) data in hours; default 24")
    parser.add_argument("--plot", dest = "plot", default = False, action = "store_true",
                        help = "Set to make plots of native grid and interpolated data")
    args = parser.parse_args()

    output_grid_string = pdp.set_grid_name_for_file_names(args.output_grid)
    print(f"Regridding {args.ds_name} to {output_grid_string}")
  
    if (args.ds_name != "CONUS404") or (args.output_grid != "Replay"):
        print(f"Error: Interpolation with xESMF for dataset {args.ds_name} to {args.output_grid} grid not yet implemented")
        sys.exit(1) 
 
    # Determine whether interpolating from a curvilinear (2-D lat/lon) grid
    is_curvilinear = False
    if (args.ds_name == utils.CONUS404_data_name) or (args.ds_name == utils.HRRR_data_name):
        is_curvilinear = True
        
    # Read output template grid
    output_ds_template = read_ds_template(args.output_grid)

    # Read input template grid
    input_ds_template = read_ds_template(args.ds_name) 

    # Read input dataset to interpolate
    input_ds = read_input_data(args.dt_str, args.ds_name, args.temporal_res)

    # Run regridding
    output_ds = run_xesmf_interp(input_ds_template,
                                 output_ds_template,
                                 input_ds,
                                 interp_method = args.interp_method, 
                                 is_curvilinear = is_curvilinear, 
                                 read_regridder = args.read_regridder)

    # Plotting
    if args.plot:
        print("Plotting")

        # Native grid
        pputils.plot_cmap_single_panel(input_ds_template.precipitation_24_hour,
                                       f"{args.ds_name}.{native_grid_str}",
                                       f"{args.ds_name}.{native_grid_str}",
                                       "CONUS",
                                       plot_levels = np.arange(0, 42, 2))

        # Output grid 
        pputils.plot_cmap_single_panel(output_ds.precipitation_24_hour,
                                       f"xesmf.{args.interp_method}.{args.ds_name}.{output_grid_string}",
                                       f"xesmf.{args.interp_method}.{args.ds_name}.{output_grid_string}",
                                       "CONUS",
                                       plot_levels = np.arange(0, 42, 2))

def add_bounds(ds):
    ds = ds.cf.add_bounds(["lat", "lon"])
    for key in ["lat", "lon"]:
      corners = cfxr.bounds_to_vertices(bounds = ds[f"{key}_bounds"], bounds_dim = "bounds", order = None)
      ds = ds.assign_coords({f"{key}_b": corners})
      ds = ds.drop_vars(f"{key}_bounds")

    # In Tim Smith's code but doesn't work here (because we don't have the 'x_vertices' and
    # 'y_vertices' dimensions, so this step isn't necessary.
    # ds = ds.rename({"x_vertices": "x_b", "y_vertices": "y_b"})

    return ds

def read_ds_template(ds_name):
    print(f"Reading template dataset for {ds_name}")

    ds_template_fpath = os.path.join(template_grids_dir, f"{ds_name}Grid.nc")
    if not(os.path.exists(ds_template_fpath)):
        print(f"Error: Template file path {ds_template_fpath} does not exist")
        sys.exit(1)
    
    return xr.open_dataset(ds_template_fpath)

# Read input dataset to interpolate
def read_input_data(dt_str, ds_name, temporal_res):
    print("Reading input data") 
    fname_prefix = f"{ds_name}.{native_grid_str}.{temporal_res}_hour_precipitation"
    fname = f"{fname_prefix}.{dt_str}.nc"

    input_dir = os.path.join(utils.data_nc_dir, fname_prefix)
    fpath = os.path.join(input_dir, fname)
    if not(os.path.exists(fpath)):
        print(f"Error: File path {fpath} does not exist")
        sys.exit(1)

    return xr.open_dataset(fpath) 

def run_xesmf_interp(in_ds_template, # Dataset on input grid 
                     out_ds_template, # Dataset on output grid
                     input_ds, # The actual data to regrid 
                     interp_method = "conservative",
                     is_curvilinear = True,
                     read_regridder = False):

    # Add bounds to facilitate regridding from an input curvilinear grid
    if is_curvilinear:
        in_ds_template = add_bounds(in_ds_template)

    # TODO: Generalize Regridder file name
    regridder_fpath = os.path.join(regridders_dir,
                                   f"Regridder.CONUS404_to_Replay.{interp_method}.nc")

    if read_regridder: # Retrieve regridder from previously-written file
        print(f"Reading {interp_method} regridder {regridder_fpath}")
        if not(os.path.exists(regridder_fpath)):
            print(f"Error: Regridder file path {regridder_fpath} does not exist")
            sys.exit(1) 

        regridder = xesmf.Regridder(ds_in = in_ds_template,
                                    ds_out = out_ds_template,
                                    method = interp_method,
                                    weights = regridder_fpath)
    else: # Build regridder and write it to netCDF
        print(f"Building {interp_method} regridder")
        regridder = xesmf.Regridder(ds_in = in_ds_template,
                                    ds_out = out_ds_template,
                                    method = interp_method)

        print(f"Writing regridder to {regridder_fpath}")
        regridder.to_netcdf(regridder_fpath)
   
    # Run regridding 
    return regridder(input_ds, keep_attrs = True)

if __name__ == "__main__":
    main()




