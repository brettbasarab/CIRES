#!/usr/bin/env python

import argparse
import precip_data_processors
import precip_plotting_utilities as pputils
import sys

def main():
    program_description = ("Run script to interpolate precip data to AORC CONUS grid,\n"
                           "write interpolated data to netCDF, and make plots of native and interpolated data.\n"
                           "Date formats for <start_dt_str> and <end_dt_str> are YYYYmmdd.HH, PERIOD-ENDING\n")
    parser = argparse.ArgumentParser(description = program_description)
    parser.add_argument("data_name",
                        help = "Name to use for the data to be processed. Currently supported options are AORC, CONUS404, NestedReplay, and HRRR.")
    parser.add_argument("start_dt_str",
                        help = "Start date of time period over which to process data")
    parser.add_argument("end_dt_str",
                        help = "End date of time period over which to process data")
    parser.add_argument("--output_temporal_res", dest = "output_temporal_res", type = int, default = 24,
                        help = "Temporal resolution (hours) of output data to write to netCDF (default 24)")
    parser.add_argument("--output_grid", dest = "output_grid", choices = ["native", "AORC"], default = "native",
                        help = "Spatial resolution of output data to write to netCDF (default 'native')")
    parser.add_argument("--replay_segment", dest = "replay_segment", choices = ["corrector", "predictor"], default = "corrector",
                        help = "Replay segment to process data from (applicable to Nested Replay only; default 'corrector')")
    parser.add_argument("--region", dest = "region", default = "CONUS",
                        help = "For plotting only: region to zoom plots to (default CONUS)")
    parser.add_argument("--write_to_nc", dest = "write_to_nc", default = False, action = "store_true",
                        help = "Set to write data to netCDF")
    parser.add_argument("--test_nc", dest = "test_nc", default = False, action = "store_true",
                        help = "Set to test writing output to nc (correct file output names will be printed, but data will not actually be written)") 
    parser.add_argument("--nc_file_cadence", dest = "nc_file_cadence", default = "day",
                        help = "Cadence of output netCDF files; default 'day'")
    parser.add_argument("--plot_maps", dest = "plot_maps", default = False, action = "store_true",
                        help = "Set to plot maps of data at each time step over the time period (which will depend on output_temporal_res)")
    args = parser.parse_args()

    dest_grid_flag = False
    output_spatial_res = "native"
    if (args.data_name != "AORC") and (args.output_grid == "AORC"):
        dest_grid_flag = True
        output_spatial_res = "dest_grid"

    # Instantiate correct data processor class
    match args.data_name:
        case "AORC":
            processor = precip_data_processors.AorcDataProcessor(args.start_dt_str,
                                                                 args.end_dt_str,
                                                                 MODEL_GRID_FLAG = False,
                                                                 region = args.region)
        case "CONUS404":
            processor = precip_data_processors.CONUS404DataProcessor(args.start_dt_str,
                                                                     args.end_dt_str,
                                                                     DEST_GRID_FLAG = dest_grid_flag,
                                                                     dest_grid_name = args.output_grid, 
                                                                     dest_temporal_res = args.output_temporal_res, 
                                                                     region = args.region)
        case "NestedReplay":
            processor = precip_data_processors.NestedReplayDataProcessor(args.start_dt_str,
                                                                         args.end_dt_str,
                                                                         DEST_GRID_FLAG = dest_grid_flag,
                                                                         dest_grid_name = args.output_grid,
                                                                         dest_temporal_res = args.output_temporal_res,
                                                                         replay_segment = args.replay_segment, 
                                                                         region = args.region)
        case _:
            print(f"Sorry, data grid processing for dataset {args.data_name} is not currently supported")
            sys.exit(0)

    # Write precipitation data to netCDF 
    if (args.write_to_nc):
        print("**** Writing data to netCDF") 
        processor.write_precip_data_to_netcdf(temporal_res = args.output_temporal_res,
                                              spatial_res = output_spatial_res,
                                              file_cadence = args.nc_file_cadence,
                                              testing = args.test_nc) 
 
    # Plot contour maps for visual inspection
    if (args.plot_maps):
        # Plot precip data at native spatial resolution
        print(f"*** Plotting {args.data_name} {args.output_temporal_res}-hourly data at native {args.data_name} spatial resolution")
        precip_native_grid = processor.get_precip_data(temporal_res = args.output_temporal_res, spatial_res = "native", load = True)
        output_grid_string = precip_data_processors.set_grid_name_for_file_names("native")
        plot_name = set_plot_output_name(output_grid_string, args.data_name, replay_segment = args.replay_segment)
        pputils.plot_cmap_single_panel(precip_native_grid,
                                       plot_name,
                                       plot_name, 
                                       processor.region,
                                       plot_levels = pputils.variable_plot_limits("accum_precip", temporal_res = args.output_temporal_res))
   
        if dest_grid_flag:
            # Plot precip data at obs grid resolution
            print(f"*** Plotting {args.data_name} {args.output_temporal_res}-hourly data at {processor.dest_grid_name} spatial resolution")
            precip_dest_grid = processor.get_precip_data(temporal_res = args.output_temporal_res, spatial_res = "dest_grid", load = True)
            output_grid_string = precip_data_processors.set_grid_name_for_file_names(processor.dest_grid_name) 
            plot_name = set_plot_output_name(output_grid_string, args.data_name, replay_segment = args.replay_segment)
            pputils.plot_cmap_single_panel(precip_dest_grid,
                                           plot_name,
                                           plot_name, 
                                           processor.region,
                                           plot_levels = pputils.variable_plot_limits("accum_precip", temporal_res = args.output_temporal_res))

    return processor

def set_plot_output_name(output_grid_string, data_name, replay_segment = "corrector"):
    plot_name = f"{data_name}.{output_grid_string}"
    if (data_name == "NestedReplay"):
        plot_name += f".{replay_segment}"
   
    return plot_name 

if __name__ == "__main__":
    processor = main() 
