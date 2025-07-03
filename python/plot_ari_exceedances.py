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
    parser.add_argument("--region", dest = "region", default = "CONUS",
                        help = "Region to zoom plot to; default CONUS")
    parser.add_argument("--temporal_res", dest = "temporal_res", default = 24, type = int, choices = [1, 3, 24],
                        help = "Temporal resolution of precip data used in verification (default 24)")
    parser.add_argument("--ari", dest = "ari", type = int, default = 2,
                        help = "Average Recurrence Interval (ARI) for which to count exceedances; default 2-year") 
    args = parser.parse_args()

    data_names, truth_data_name, data_grid = precip_verification_processor.map_region_to_data_names(args.region)

    verif = precip_verification_processor.PrecipVerificationProcessor(args.start_dt_str, 
                                                                      args.end_dt_str,
                                                                      data_names = data_names, 
                                                                      truth_data_name = truth_data_name, 
                                                                      data_grid = data_grid,
                                                                      region = args.region,
                                                                      temporal_res = args.temporal_res)


    ari_dir = os.path.join(utils.data_nc_dir, "ARIs.ReplayGrid")
    ari_fname = f"ARI.ReplayGrid.{args.ari:02d}_year.{args.temporal_res}_hour_precipitation.nc"
    ari_fpath = os.path.join(ari_dir, ari_fname)
    if not (os.path.exists(ari_fpath)):
        print(f"Error: ARI file {ari_fpath} does not exist")
        sys.exit(1)
    aris = xr.open_dataset(ari_fpath).precip
    aris_region_subset = verif._subset_data_to_region(aris, data_name = "ARI_grid")

    ari_exceedance_dict = {}
    for data_name, da in verif.da_dict.items():
        # Calculate ARI exceedances, sum over time period
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
    verif.plot_cmap_multi_panel(data_dict = ari_exceedance_dict, plot_levels = bounds, cmap = cmap) 
 
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
