#!/usr/bin/env python

import argparse
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import precip_verification_processor
import precip_data_processors as pdp
import precip_plotting_utilities as pputils
import sys
import utilities as utils
import xarray as xr

def main():
    parser = argparse.ArgumentParser(description = "Program to sum and plot all ARI exceedances over specified time period.")
    parser.add_argument("start_dt_str",
                        help = "Start date/time of verification; format consistent with temporal_res flag, e.g., 'YYYYmmdd' for 24-hourly data")
    parser.add_argument("end_dt_str",
                        help = "End date/time of verification; format consistent with temporal_res flag, e.g., 'YYYYmmdd' for 24-hourly data")
    parser.add_argument("--data_names_list", dest = "data_names_list", default = None, nargs = "+",
                        help = "List of data names to verify; pass space-separated with the truth data name listed first; if None, all possible datasets for the given region are verified")
    parser.add_argument("--verif_grid", dest = "verif_grid", default = "Replay", choices = ["AORC", "Replay"],
                        help = "Set a standard verification grid for CONUS (or subregions) verification [AORC and (global) Replay are currently supported; default Replay]")
    parser.add_argument("--include_hrrr", dest = "include_hrrr", action = "store_true", default = False,
                        help = "Set to include HRRR in verification")
    parser.add_argument("--region", dest = "region", default = "CONUS",
                        help = "Region to zoom plot to; default CONUS")
    parser.add_argument("--temporal_res", dest = "temporal_res", default = 24, type = int, choices = [1, 3, 24],
                        help = "Temporal resolution of precip data used in verification (default 24)")
    parser.add_argument("--ari", dest = "ari", type = int, default = 2,
                        help = "Average Recurrence Interval (ARI) for which to count exceedances; default 2-year") 
    args = parser.parse_args()
    
    # Read ARI grid
    output_grid_string = pdp.set_grid_name_for_file_names(args.verif_grid)
    ari_dir = os.path.join(utils.data_nc_dir, f"ARIs.{output_grid_string}")
    ari_fname = f"ARI.{output_grid_string}.{args.ari:02d}_year.{args.temporal_res}_hour_precipitation.nc"
    ari_fpath = os.path.join(ari_dir, ari_fname)
    print(f"Reading {ari_fpath}")
    if not (os.path.exists(ari_fpath)):
        print(f"Error: ARI file {ari_fpath} does not exist")
        sys.exit(1)
    aris = xr.open_dataset(ari_fpath).precip

    # Determine data names to use for this particular type of verification 
    if (type(args.data_names_list) is list) and (len(args.data_names_list) >= 1):
        data_names, truth_data_name = args.data_names_list, args.data_names_list[0]
        _, _, data_grid = precip_verification_processor.map_region_to_data_names(args.region, verif_grid = args.verif_grid)
    else:
        data_names, truth_data_name, data_grid = precip_verification_processor.map_region_to_data_names(args.region,
                                                                                                        verif_grid = args.verif_grid,
                                                                                                        include_hrrr = args.include_hrrr)

    # Instantiate PrecipVerificationProcessor class
    verif = precip_verification_processor.PrecipVerificationProcessor(args.start_dt_str, 
                                                                      args.end_dt_str,
                                                                      data_names = data_names, 
                                                                      truth_data_name = truth_data_name, 
                                                                      data_grid = data_grid,
                                                                      region = args.region,
                                                                      temporal_res = args.temporal_res)

    # Subset ARI grid to current region
    aris_region_subset = verif._subset_data_to_region(aris, data_name = "ARI_grid")

    # Calculate number of ARI exceedances
    ari_exceedance_dict = {}
    for data_name, da in verif.da_dict.items():
        # Calculate ARI exceedances, sum over full time period along time dimension
        ari_exceedance_da = xr.where(da > aris_region_subset, 1, 0, keep_attrs = True)
        ari_exceedance_da_sum = ari_exceedance_da.sum(dim = utils.period_end_time_dim_str, keepdims = True)

        # Add a period_end_time dimension to facilitate plotting
        end_dt = pd.Timestamp(da[utils.period_end_time_dim_str].values[-1])
        ari_exceedance_da_sum.coords[utils.period_end_time_dim_str] = [end_dt]

        # Add relevant attributes for plotting
        pdp.add_attributes_to_data_array(ari_exceedance_da_sum,
                                         short_name = f"{args.ari:02d}year_{args.temporal_res:02d}hour_ARIexcd",
                                         long_name = f"{args.ari:02d} year, {args.temporal_res:02d} hour ARI Exceedances",
                                         units = "")

        # Add to dictionary
        ari_exceedance_dict[data_name] = ari_exceedance_da_sum

    # Plot cumulative ARI exceedances
    cmap, bounds, norm = create_ari_colorbar()
    verif.plot_cmap_multi_panel(data_dict = ari_exceedance_dict, plot_levels = bounds, plot_cmap = cmap) 
 
def create_ari_colorbar():
    color_list = ["white", "cyan", "blue", "green", "red", "purple"]
    bounds = [0,1,2,3,4,5,6]
   
    # Other options for cmap; consider a way to select one of these options 
    #color_list = ["white", "yellow", "orange", "red", "purple"]
    #bounds = [0,1,2,3,4,5]
    
    #color_list = ["white", "red"]
    #bounds = [0, 1, 2]
    
    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, bounds, norm

if __name__ == "__main__":
    main()
