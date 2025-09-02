#!/usr/bin/env python

import argparse
import cf_xarray as cfxr
import datetime as dt
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
    program_description = ("Program to interpolate precip data using the xesmf package, write\n"
                           "write interpolated data to netCDF, and make plots of native and interpolated data.\n"
                           "Note that each dataset requires different input date formats, listed below:\n"
                           "\tAORC: YYYYmmdd.HH and PERIOD-ENDING\n"
                           "\tCONUS404: YYYYmmdd.HH and PERIOD-ENDING\n"
                           "\tERA5: YYYYmmdd.HH and PERIOD-BEGINNING\n"
                           "\tIMERG: YYYYmmdd and PERIOD-BEGINNING (read daily netCDF files on IMERG native grid)\n"
                           "\tReplay: YYYYmmdd.HH and PERIOD-ENDING\n")
    print(program_description)
    parser = argparse.ArgumentParser()
    parser.add_argument("start_dt_str",
                        help = "Start date of time period over which to process data")
    parser.add_argument("end_dt_str",
                        help = "End date of time period over which to process data")
    parser.add_argument("--ds_name", dest = "ds_name", default = "CONUS404",
                        choices = utils.dataset_list, 
                        help = "Dataset name to regrid; default CONUS404")
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
    parser.add_argument("--region", dest = "region", default = "CONUS",
                        help = "For plotting only: region to zoom plots to (default CONUS)")
    parser.add_argument("--test_nc", dest = "test_nc", default = False, action = "store_true",
                        help = "Set to test writing output to nc (correct file output names will be printed, but data will not actually be written)") 
    args = parser.parse_args()

    output_grid_string = pdp.set_grid_name_for_file_names(args.output_grid)
    print(f"Regridding {args.ds_name} to {output_grid_string}")
  
    if (args.output_grid != "Replay"):
        print(f"Error: Interpolation with xESMF to {args.output_grid} grid not yet implemented")
        sys.exit(1) 
 
    # Determine whether interpolating from a curvilinear (2-D lat/lon) grid
    is_curvilinear = False
    if (args.ds_name == utils.CONUS404_data_name) or (args.ds_name == utils.HRRR_data_name):
        is_curvilinear = True
        
    # Read output template grid
    output_ds_template = read_ds_template(args.output_grid)

    # Read input template grid
    input_ds_template = read_ds_template(args.ds_name) 

    # Instantiate correct data processor class and then
    # get precip data at the dataset's native spatial resolution and the
    # desired temporal resolution based on args.temporal_res. This data is what
    # we'll regrid to args.output_grid using xESMF.
    match args.ds_name:
        case "AORC":
            processor = pdp.AorcDataProcessor(args.start_dt_str,
                                              args.end_dt_str,
                                              region = args.region)
            input_da = processor.get_precip_data(spatial_res = "native", temporal_res = args.temporal_res, load = True)
        case "CONUS404":
            processor = pdp.CONUS404DataProcessor(args.start_dt_str,
                                                  args.end_dt_str,
                                                  region = args.region)
            input_da = processor.get_precip_data(spatial_res = "native", temporal_res = args.temporal_res, load = True)
        case "ERA5":
            processor = pdp.ERA5DataProcessor(args.start_dt_str,
                                              args.end_dt_str,
                                              region = args.region)
            input_da = processor.get_precip_data(spatial_res = "native", temporal_res = args.temporal_res, load = True)
        case "IMERG":
            # IMERG native data won't load (within a reasonable time) in xesmf environment. So, 
            # interpolate IMERG data at desired temporal resolution on native grid previously created in default bbasarab_env environment.
            #processor = pdp.ImergDataProcessor(args.start_dt_str,
            #                                   args.end_dt_str,
            #                                   region = args.region)
            #input_da = processor.get_precip_data(spatial_res = "native", temporal_res = args.temporal_res, load = True)

            start_dt = dt.datetime.strptime(args.start_dt_str, "%Y%m%d")
            end_dt = dt.datetime.strptime(args.end_dt_str, "%Y%m%d")
            current_dt = start_dt
            valid_dt_list = [current_dt]
            while (current_dt != end_dt):
                current_dt += dt.timedelta(days = 1)
                valid_dt_list.append(current_dt)
            #valid_daily_dt_list_period_begin = [dtime - dt.timedelta(days = 1) for dtime in valid_daily_dt_list[1:]]

            # Collect netCDF file list
            fname_prefix = f"{args.ds_name}.{native_grid_str}.{args.temporal_res:02d}_hour_precipitation"
            file_list = []
            for dtime in valid_dt_list:
                fname = f"{fname_prefix}.{dtime:%Y%m%d}.nc"
                fpath = os.path.join(utils.data_nc_dir, fname_prefix, fname)
                if (not os.path.exists(fpath)):
                    print(f"Warning: Input file path {fpath} does not exist; not including in input file list")
                    continue
                file_list.append(fpath)

            if (len(file_list) == 0):
                print(f"Error: No input files found in directory {fname_prefix}; can't proceed with verification")
                sys.exit(1)

            # Read multi-file dataset
            dataset = xr.open_mfdataset(file_list)
            input_da = dataset[f"precipitation_{args.temporal_res:02d}_hour"]
            input_da.attrs["data_name"] = args.ds_name

            # Index obs data array to correct datetime range
            input_da = input_da.loc[start_dt.strftime(utils.full_date_format_str):end_dt.strftime(utils.full_date_format_str)]
        case "NestedReplay":
            processor = pdp.NestedReplayDataProcessor(args.start_dt_str,
                                                      args.end_dt_str,
                                                      region = args.region)
            input_da = processor.get_precip_data(spatial_res = "native", temporal_res = args.temporal_res, load = True)
        case "Replay":
            processor = pdp.ReplayDataProcessor(args.start_dt_str,
                                                args.end_dt_str,
                                                region = args.region)
            input_da = processor.get_replay_precip_data(time_period_hours = args.temporal_res, spatial_res = "native", load = True) 
        case _:
            print(f"Sorry, data grid processing for dataset {args.data_name} is not currently supported")
            sys.exit(0)

    # Run regridding
    output_da = run_xesmf_interp(input_ds_template,
                                 output_ds_template,
                                 input_da,
                                 interp_method = args.interp_method, 
                                 is_curvilinear = is_curvilinear, 
                                 read_regridder = args.read_regridder)

    # Write output regridded data to netCDF
    write_to_netcdf(output_da, args.ds_name, output_grid_string, args.interp_method,
                    temporal_res = args.temporal_res, testing = args.test_nc)

    # Plotting
    if args.plot:
        print("Plotting")
        short_name = f"{args.temporal_res:02d}_hour_precipitation"

        # Native grid
        pputils.plot_cmap_single_panel(input_da,
                                       f"xesmf.{args.ds_name}.{native_grid_str}",
                                       args.region,
                                       short_name = short_name, 
                                       plot_levels = np.arange(0, 42, 2))

        # Output grid 
        pputils.plot_cmap_single_panel(output_da,
                                       f"xesmf.{args.interp_method}.{args.ds_name}.{output_grid_string}",
                                       args.region,
                                       short_name = short_name, 
                                       plot_levels = np.arange(0, 42, 2))

    return input_da, output_da

def add_bounds(ds):
    ds = ds.cf.add_bounds(["lat", "lon"])
    for key in ["lat", "lon"]:
      corners = cfxr.bounds_to_vertices(bounds = ds[f"{key}_bounds"], bounds_dim = "bounds", order = None)
      ds = ds.assign_coords({f"{key}_b": corners})
      ds = ds.drop_vars(f"{key}_bounds")

    # In Tim Smith's code but doesn't work here (because we don't have the 'x_vertices' and
    # 'y_vertices' dimensions, so this step isn't necessary).
    # ds = ds.rename({"x_vertices": "x_b", "y_vertices": "y_b"})

    return ds

def read_ds_template(ds_name):
    print(f"Reading template dataset for {ds_name}")

    ds_template_fpath = os.path.join(template_grids_dir, f"{ds_name}Grid.nc")
    if not(os.path.exists(ds_template_fpath)):
        print(f"Error: Template file path {ds_template_fpath} does not exist")
        sys.exit(1)
   
    ds = xr.open_dataset(ds_template_fpath)
    ds.attrs["ds_name"] = ds_name
    
    return ds

# Read input dataset to interpolate
# NOTE: No longer used but keeping for now for testing/debuggin purposes
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

    regridder_fpath = os.path.join(regridders_dir,
                                   f"Regridder.{in_ds_template.ds_name}_to_{out_ds_template.ds_name}.{interp_method}.nc")

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

def write_to_netcdf(output_da, ds_name, output_grid, interp_method,
                    temporal_res = 24, testing = False):
    # Construct timestamp format
    if (temporal_res < 24):
        timestamp_format = "%Y%m%d.%H"
    else:
        timestamp_format = "%Y%m%d"

    base_dir_name = f"xesmf.{interp_method}.{ds_name}.{output_grid}.{temporal_res:02d}_hour_precipitation"
    dir_name = os.path.join(utils.data_nc_dir, base_dir_name)
    fname_prefix = f"{ds_name}.{output_grid}.{temporal_res:02d}_hour_precipitation"
    output_var_name = f"precipitation_{temporal_res:02d}_hour"
    output_da.name = output_var_name

    pdp.write_data_array_to_netcdf(output_da,
                                   output_var_name,
                                   dir_name,
                                   fname_prefix,
                                   timestamp_format,
                                   temporal_res = temporal_res,
                                   file_cadence = "day",
                                   testing = testing)    

if __name__ == "__main__":
    input_da, output_da = main()




