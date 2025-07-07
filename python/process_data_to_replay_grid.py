#!/usr/bin/env python

import argparse
import precip_data_processors
import precip_plotting_utilities as pputils
import sys

def main():
    program_description = ("Run script to interpolate precip data to global Replay grid,\n"
                           "write interpolated data to netCDF, and make plots of native and interpolated data.\n"
                           "Note that each dataset requires different input date formats, listed below:\n"
                           "\tAORC: YYYYmmdd.HH and PERIOD-ENDING\n"
                           "\tCONUS404: YYYYmmdd.HH and PERIOD-ENDING\n"
                           "\tERA5: YYYYmmdd.HH and PERIOD-BEGINNING\n"
                           "\tIMERG: YYYYmmdd.HH[00,30] and PERIOD-BEGINNING\n"
                           "\tReplay: YYYYmmdd.HH and PERIOD-ENDING\n")
    print(program_description)
    parser = argparse.ArgumentParser()
    parser.add_argument("data_name",
                        help = "Name to use for the data to be processed (AORC, CONUS404, ERA5, IMERG, Replay, etc.)")
    parser.add_argument("start_dt_str",
                        help = "Start date of time period over which to process data")
    parser.add_argument("end_dt_str",
                        help = "End date of time period over which to process data")
    parser.add_argument("--temporal_res", dest = "temporal_res", type = int, default = 24,
                        help = "Temporal resolution in hours of the processed obs data to write to netCDF (default 24)")
    parser.add_argument("--spatial_res", dest = "spatial_res", choices = ["native", "model"], default = "native",
                        help = "Spatial resolution type of the data to write to netCDF (default 'native')")
    parser.add_argument("--region", dest = "region", default = "Global",
                        help = "For plotting only: region to zoom plots to")
    parser.add_argument("--write_to_nc", dest = "write_to_nc", default = False, action = "store_true",
                        help = "Set to write data to netCDF")
    parser.add_argument("--test_nc", dest = "test_nc", default = False, action = "store_true",
                        help = "Set to test writing output to nc (correct file output names will be printed, but data will not actually be written)") 
    parser.add_argument("--nc_file_cadence", dest = "nc_file_cadence", default = "day",
                        help = "Cadence of output netCDF files; default 'day'")
    parser.add_argument("--plot_maps", dest = "plot_maps", default = False, action = "store_true",
                        help = "Set to plot maps of data at each time step over the time period (which will depend on temporal_res)")
    args = parser.parse_args()

    model_grid_flag = False
    model_temporal_res = 3
    if (args.spatial_res == "model"):
        model_grid_flag = True

    # Instantiate correct data processor class
    match args.data_name:
        case "AORC":
            processor = precip_data_processors.AorcDataProcessor(args.start_dt_str,
                                                                 args.end_dt_str,
                                                                 MODEL_GRID_FLAG = model_grid_flag,
                                                                 model_temporal_res = model_temporal_res, 
                                                                 region = args.region)
        case "CONUS404":
            processor = precip_data_processors.CONUS404DataProcessor(args.start_dt_str,
                                                                     args.end_dt_str,
                                                                     DEST_GRID_FLAG = model_grid_flag,
                                                                     dest_temporal_res = args.temporal_res,
                                                                     region = args.region)
        case "ERA5":
            processor = precip_data_processors.ERA5DataProcessor(args.start_dt_str,
                                                                 args.end_dt_str,
                                                                 MODEL_GRID_FLAG = model_grid_flag,
                                                                 model_temporal_res = model_temporal_res, 
                                                                 region = args.region)
        case "IMERG":
            processor = precip_data_processors.ImergDataProcessor(args.start_dt_str,
                                                                  args.end_dt_str,
                                                                  MODEL_GRID_FLAG = model_grid_flag,
                                                                  model_temporal_res = model_temporal_res, 
                                                                  region = args.region)
        case "Replay":
            print("Processing data for Replay only; not for any other datasets")
            processor = precip_data_processors.ReplayDataProcessor(args.start_dt_str,
                                                                   args.end_dt_str,
                                                                   region = args.region)
        case _:
            print(f"Sorry, data grid processing for dataset {args.data_name} is not currently supported")
            sys.exit(0)

    # Write observed precipitation to netCDF 
    if (args.write_to_nc):
        print("**** Writing data to netCDF") 
        if (args.data_name == "Replay"):
            processor.write_replay_precip_data_to_netcdf(temporal_res = args.temporal_res,
                                                        file_cadence = args.nc_file_cadence,
                                                        testing = args.test_nc)
        else:
            processor.write_precip_data_to_netcdf(temporal_res = args.temporal_res,
                                                  spatial_res = args.spatial_res,
                                                  file_cadence = args.nc_file_cadence,
                                                  testing = args.test_nc) 
 
    # Plot contour maps for visual inspection
    if (args.plot_maps):
        if (args.data_name == "Replay"):
            # Plot Replay data
            print(f"*** Plotting {processor.model_name} data")
            model_precip = processor.get_model_precip_data(time_period_hours = args.temporal_res, load = True)
            pputils.plot_cmap_single_panel(model_precip,
                                           f"{processor.model_name}.NativeGrid",
                                           f"{processor.model_name}.NativeGrid",
                                           processor.region,
                                           plot_levels = pputils.variable_plot_limits("accum_precip", temporal_res = args.temporal_res))
        else:
            # Plot QPE data at native spatial resolution
            print(f"*** Plotting {args.data_name} {args.temporal_res}-hourly data at native {args.data_name} spatial resolution")
            obs_precip = processor.get_precip_data(temporal_res = args.temporal_res, spatial_res = "native", load = True)
            pputils.plot_cmap_single_panel(obs_precip,
                                           f"{processor.obs_name}.NativeGrid",
                                           f"{processor.obs_name}.NativeGrid",
                                           processor.region,
                                           plot_levels = pputils.variable_plot_limits("accum_precip", temporal_res = args.temporal_res))
       
            if model_grid_flag: 
                # Plot QPE data at Replay resolution
                print(f"*** Plotting {args.data_name} {args.temporal_res}-hourly data at {processor.model_name} spatial resolution")
                obs_precip_model_grid = processor.get_precip_data(temporal_res = args.temporal_res, spatial_res = "model", load = True)
                pputils.plot_cmap_single_panel(obs_precip_model_grid,
                                               f"{processor.obs_name}.{processor.model_name}Grid",
                                               f"{processor.obs_name}.{processor.model_name}Grid",
                                               processor.region,
                                               plot_levels = pputils.variable_plot_limits("accum_precip", temporal_res = args.temporal_res))

                # Plot Replay data
                print(f"*** Plotting {processor.model_name} data")
                model_precip = processor.get_model_precip_data(time_period_hours = args.temporal_res, load = True)
                pputils.plot_cmap_single_panel(model_precip,
                                               f"{processor.model_name}.NativeGrid",
                                               f"{processor.model_name}.NativeGrid",
                                               processor.region,
                                               plot_levels = pputils.variable_plot_limits("accum_precip", temporal_res = args.temporal_res))

    return processor

if __name__ == "__main__":
    processor = main() 
