#!/usr/bin/env python

import argparse
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import precip_plotting_utilities as pputils
import precip_verification_processor
import sys
import utilities as utils
import warnings
import xarray as xr

def main():
    utils.suppress_warnings()
    program_description = ("Program to create an instance of PrecipVerificationProcessor and use its functionality to create\n"
                          "many different types of plots, summarized in the flag descriptions below.")
    parser = argparse.ArgumentParser(description = program_description)
    # Fundamental positional arguments and flags to configure verification properly
    parser.add_argument("start_dt_str",
                        help = "Start date/time of verification; format consistent with temporal_res flag, e.g., 'YYYYmmdd' for 24-hourly data")
    parser.add_argument("end_dt_str",
                        help = "End date/time of verification; format consistent with temporal_res flag, e.g., 'YYYYmmdd' for 24-hourly data")
    parser.add_argument("--regions", dest = "regions", nargs = "+", default = ["CONUS"],
                        help = "Regions for which to perform verification; default CONUS")
    parser.add_argument("--region_ext", dest = "region_ext", nargs = "+", type = int,
                        help = "Lon and lat boundaries for a non-standard region; order llon, ulon, llat, ulat; region name must also be provided as --region flag") 
    parser.add_argument("--temporal_res", default = 24, type = int, choices = [1, 3, 24],
                        help = "Temporal resolution of precip data used in verification (default 24)")
    parser.add_argument("--data_names_list", dest = "data_names_list", default = None, nargs = "+",
                        help = "List of data names to verify; pass space-separated with the truth data name listed first; if None, all possible datasets for the given region are verified")
    parser.add_argument("--grid_cell_size", dest = "grid_cell_size", type = float, default = None, 
                        help = f"Assumed grid cell size for FSS and other calculations; default None to use AORC or global Replay grid cell size depending on args.verif_grid.")
    parser.add_argument("--time_period_types", dest = "time_period_types", default = ["full_period", "common_seasonal", "common_monthly"], nargs = "+", 
                        help = "Time period types over which to perform analysis; default ['full_period', 'common_seasonal', 'common_monthly']")
    parser.add_argument("--verif_grid", dest = "verif_grid", default = "Replay", choices = ["AORC", "Replay"],
                        help = "Set a standard verification grid for CONUS (or subregions) verification [AORC and (global) Replay are currently supported; default Replay]")
    parser.add_argument("--include_hrrr", dest = "include_hrrr", action = "store_true", default = False,
                        help = "Set to include HRRR in verification")
    # Statistics flags: set which statistic(s) to calculated
    parser.add_argument("--cmaps", dest = "cmaps", default = None, choices = [None, "mean", "pctls", "all"],
                        help = "Set to plot contour maps (cmaps) of mean and/or percentiles of gridded data aggregated over various time periods")
    parser.add_argument("--fss", dest = "fss", action = "store_true", default = False,
                        help = "Calculate fractions skill score (FSS) using amount thresholds")
    parser.add_argument("--fss_pctl", dest = "fss_pctl", action = "store_true", default = False,
                        help = "Calculate fractions skill score (FSS) using percentile thresholds")
    parser.add_argument("--fss_ari", dest = "fss_ari", action = "store_true", default = False,
                        help = "Calculate fractions skill score (FSS) using ARI grid thresholds")
    parser.add_argument("--pdfs", dest = "pdfs", action = "store_true", default = False,
                        help = "Calculate probability density functions (PDFs) of precip") 
    parser.add_argument("--timeseries", dest = "timeseries", action = "store_true", default = False,
                        help = "Calculate time series of precip stats aggregated over individual months, seasons, years, etc. in the specified time period")
    # Write to netCDF flags: set to write calculated statistic(s) to netCDF
    parser.add_argument("--write_to_nc", dest = "write_to_nc", default = False, action = "store_true",
                        help = "Set to write calculated stats to netCDF")
    # Plotting flags
    parser.add_argument("--plot", dest = "plot", action = "store_true", default = False,
                        help = "Set to make plots of calculated stats")
    parser.add_argument("--plot_fss_timeseries", dest = "plot_fss_timeseries", action = "store_true", default = False,
                        help = "Set to plot timeseries of FSS for each individual time period within evaluation period (useful for shorter 1-3 month analyses)")
    parser.add_argument("--poster", dest = "poster", action = "store_true", default = False,
                        help = "Set if plots to be produced are for a poster; will make plot fonts bigger")
    parser.add_argument("--scatter_plot", dest = "scatter_plot", action = "store_true", default = False,
                        help = "Create scatter plots of accumulated precip over entire specified time period (most useful for case studies)")
    args = parser.parse_args()
    
    # Set grid cell size for FSS calculations
    grid_cell_size = args.grid_cell_size
    if (grid_cell_size is None):
        grid_cell_size = utils.replay_grid_cell_size
        if (args.verif_grid == "AORC"):
            grid_cell_size = utils.aorc_grid_cell_size 
    print(f"Grid cell size: {grid_cell_size} degrees")

    for r, region in enumerate(args.regions):
        LOAD_DATA = True
        loaded_non_subset_da_dict = None 
        if (r > 0):
            LOAD_DATA = False
            loaded_non_subset_da_dict = verif.loaded_non_subset_da_dict # From previous instantiaon of PrecipVerificationProcessor

        # Determine data names to use for this particular type of verification 
        if (type(args.data_names_list) is list) and (len(args.data_names_list) >= 1):
            data_names, truth_data_name = args.data_names_list, args.data_names_list[0]
            _, _, data_grid = precip_verification_processor.map_region_to_data_names(region, verif_grid = args.verif_grid)
        else:
            data_names, truth_data_name, data_grid = precip_verification_processor.map_region_to_data_names(region,
                                                                                                            verif_grid = args.verif_grid,
                                                                                                            include_hrrr = args.include_hrrr)

        # Set up verification by instantiating the PrecipVerificationProcessor class
        verif = precip_verification_processor.PrecipVerificationProcessor(args.start_dt_str, 
                                                                          args.end_dt_str,
                                                                          LOAD_DATA = LOAD_DATA,
                                                                          loaded_non_subset_da_dict = loaded_non_subset_da_dict,
                                                                          USE_EXTERNAL_DA_DICT = False,
                                                                          external_da_dict = None, 
                                                                          data_names = data_names, 
                                                                          truth_data_name = truth_data_name,
                                                                          data_grid = data_grid,
                                                                          region = region,
                                                                          region_extent = args.region_ext,
                                                                          temporal_res = args.temporal_res,
                                                                          poster = args.poster)

        # Calculate data for and plot contour maps of specified statistics valid at each grid point (so
        # contour maps can be made of the resulting data) and across various aggregation time periods.
        if (args.cmaps == "mean") or (args.cmaps == "all"):
            for time_period_type in args.time_period_types: 
                print(f"**** Calculating {time_period_type} mean data for contour maps")
                # Mean
                agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type, stat_type = "mean", agg_type = "time", write_to_nc = args.write_to_nc)
                if args.plot:
                    precip_range = pputils.regions_info_dict[verif.region].cm_mean_precip_range
                    verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = True, single_set_of_levels = True, plot_errors = False, plot_levels = precip_range)
                    verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = False, single_set_of_levels = True, plot_errors = True, plot_levels = precip_range)

        if (args.cmaps == "pctls") or (args.cmaps == "all"):
            for time_period_type in args.time_period_types:
                print(f"**** Calculating {time_period_type} percentile data for contour maps")
                # 95th percentile
                agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type, stat_type = "pctl", agg_type = "time", pctl = 95, write_to_nc = args.write_to_nc)
                if args.plot:
                    verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = True, single_set_of_levels = False, plot_errors = False)
                    verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = False, single_set_of_levels = False, plot_errors = True)

                # 99th percentile
                agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type, stat_type = "pctl", agg_type = "time", pctl = 99, write_to_nc = args.write_to_nc)
                if args.plot:
                    verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = True, single_set_of_levels = False, plot_errors = False)
                    verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = False, single_set_of_levels = False, plot_errors = True)

                # 99.9th percentile
                agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type, stat_type = "pctl", agg_type = "time", pctl = 99.9, write_to_nc = args.write_to_nc)
                if args.plot:
                    verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = True, single_set_of_levels = False, plot_errors = False)
                    verif.plot_cmap_multi_panel(data_dict = agg_dict, single_colorbar = False, single_set_of_levels = False, plot_errors = True)
        
        eval_radius_list = utils.default_eval_radius_list_grid_cells * grid_cell_size
        # Calculate and plot fractions skill score (FSS) by radius and amount threshold
        if args.fss:
            print("******** Calculating FSS (using amount thresholds)")
            eval_threshold_list = utils.default_eval_threshold_list_mm

            # Calculate FSS by radius, using a fixed amount threshold 
            fss_dict_by_radius = verif.calculate_fss(eval_type = "by_radius",
                                                     grid_cell_size = grid_cell_size,
                                                     fixed_threshold = 10, # mm 
                                                     eval_radius_list = eval_radius_list, 
                                                     is_pctl_threshold = False, 
                                                     include_zeros = False,
                                                     write_to_nc = args.write_to_nc)

            # Calculate FSS by amount threshold, using a fixed evaluation radius
            fss_dict_by_thresh = verif.calculate_fss(eval_type = "by_threshold",
                                                     grid_cell_size = grid_cell_size,
                                                     fixed_radius = 2 * grid_cell_size, # degrees
                                                     eval_threshold_list = eval_threshold_list,
                                                     is_pctl_threshold = False,
                                                     include_zeros = False,
                                                     write_to_nc = args.write_to_nc)
     
            # Plot FSS (using amount thresholds), averaged across evaluation period 
            if args.plot:
                for time_period_type in args.time_period_types:
                    print(f"**** Calculating and plotting {time_period_type} aggregated FSS (using amount thresholds)")

                    verif.plot_aggregated_fss(eval_type = "by_radius", xaxis_explicit_values = False,
                                              time_period_type = time_period_type,
                                              is_pctl_threshold = False)

                    verif.plot_aggregated_fss(eval_type = "by_threshold", xaxis_explicit_values = False,
                                              time_period_type = time_period_type,
                                              is_pctl_threshold = False, 
                                              include_frequency_bias = True)

            # Plot FSS timeseries (using amount thresholds) for each time period across evaluation period
            if args.plot_fss_timeseries:
                print(f"Plotting FSS timeseries for eval radius {2*grid_cell_size:0.2f} deg; threshold 10mm")
                verif.plot_fss_timeseries(2 * grid_cell_size, 10)
     
        # Calculate and plot fractions skill score (FSS) by radius and percentile threshold 
        if args.fss_pctl:
            print("******** Calculating FSS (using percentile thresholds)")
            eval_threshold_list = utils.default_eval_threshold_list_pctl

            # Calculate FSS by radius, using a fixed amount threshold 
            fss_dict_by_radius = verif.calculate_fss(eval_type = "by_radius",
                                                     grid_cell_size = grid_cell_size,
                                                     fixed_threshold = 95, # percentile 
                                                     eval_radius_list = eval_radius_list, 
                                                     is_pctl_threshold = True, 
                                                     include_zeros = False, # whether to include zeros in percentile calculations
                                                     write_to_nc = args.write_to_nc)

            # Calculate FSS by amount threshold, using a fixed evaluation radius
            fss_dict_by_thresh = verif.calculate_fss(eval_type = "by_threshold",
                                                     grid_cell_size = grid_cell_size,
                                                     fixed_radius = 2 * grid_cell_size, # degrees
                                                     eval_threshold_list = eval_threshold_list,
                                                     is_pctl_threshold = True,
                                                     include_zeros = False, # whether to include zeros in percentile calculations
                                                     write_to_nc = args.write_to_nc)

            # Plot FSS (using percentile thresholds), averaged across evaluation period 
            if args.plot:
                for time_period_type in args.time_period_types:
                    print(f"**** Calculating and plotting {time_period_type} aggregated FSS (using percentile thresholds)")

                    verif.plot_aggregated_fss(eval_type = "by_radius", xaxis_explicit_values = False,
                                              time_period_type = time_period_type,
                                              is_pctl_threshold = True)

                    verif.plot_aggregated_fss(eval_type = "by_threshold", xaxis_explicit_values = False,
                                              time_period_type = time_period_type,
                                              is_pctl_threshold = True, 
                                              include_frequency_bias = True)

        # Calculate and plot fractions skill score (FSS) by radius and ARI grid threshold 
        if args.fss_ari:
            print("******** Calculating FSS (using ARI grid thresholds)")

            # Calculate FSS by radius, using an ARI grid threshold
            fss_dict_by_radius = verif.calculate_fss(eval_type = "by_radius_ari_threshold",
                                                     grid_cell_size = grid_cell_size,
                                                     fixed_ari_threshold = 2, # 2-year ARI grid 
                                                     eval_radius_list = eval_radius_list,
                                                     write_to_nc = args.write_to_nc)

            # Calculate FSS by ARI grid (using varying ARI grids thresholds) 
            fss_dict_by_ari = verif.calculate_fss(eval_type = "by_ari_grid",
                                                  grid_cell_size = grid_cell_size,
                                                  fixed_radius = 2 * grid_cell_size, # degrees
                                                  eval_ari_list = utils.default_eval_ari_list_years,
                                                  write_to_nc = args.write_to_nc)

            # Plot FSS (using ARI grid thresholds), averaged across evaluation period 
            if args.plot:
                for time_period_type in args.time_period_types:
                    print(f"**** Calculating and plotting {time_period_type} aggregated FSS (using ARI grid thresholds)")

                    verif.plot_aggregated_fss(eval_type = "by_radius_ari_threshold", xaxis_explicit_values = False,
                                              time_period_type = time_period_type)

                    verif.plot_aggregated_fss(eval_type = "by_ari_grid", xaxis_explicit_values = True,
                                              time_period_type = time_period_type)
        # Plot PDFs and CDFs
        if args.pdfs:
            for time_period_type in args.time_period_types:
                print(f"**** Calculating {time_period_type} PDFs and CDFs")
                pdf_dict = verif.calculate_pdf(time_period_type = time_period_type, write_to_nc = args.write_to_nc)
                if args.plot:
                    verif.plot_pdf(data_dict = pdf_dict, time_period_type = time_period_type) 

        # Plot monthly and seasonal mean timeseries
        if args.timeseries:
            for time_period_type in ["monthly", "seasonal"]:
                print(f"**** Calculating {time_period_type} timeseries stats")
                agg_dict = verif.calculate_aggregated_stats(time_period_type = time_period_type, stat_type = "mean", agg_type = "space_time", write_to_nc = args.write_to_nc) 
                if args.plot:
                    verif.plot_timeseries(data_dict = agg_dict, time_period_type = time_period_type, stat_type = "mean")

        # Plot scatter plot of accumulated precip over entire time period (specified by args.start_dt_str-args.end_dt_str)
        if args.scatter_plot:
            print("**** Creating scatter plots of accumulated precip summed over entire time period")
            verif.plot_scatter_plot(summed_over_time_period = True)

    return verif

if __name__ == "__main__":
    verif = main()
