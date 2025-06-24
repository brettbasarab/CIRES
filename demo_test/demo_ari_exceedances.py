#!/usr/bin/env python

import argparse
import matplotlib as mpl
import numpy as np
import pandas as pd
import precip_verification_processor
import precip_data_processors as pdp
import precip_plotting_utilities as pputils
import utilities as utils
import xarray as xr

def create_ari_colorbar():
    #color_list = ["white", "yellow", "orange", "red", "purple"]
    
    color_list = ["white", "cyan", "blue", "green", "red", "purple"]
    bounds = [0,1,2,3,4,5,6]
    
    #color_list = ["white", "red"]
    #bounds = [0, 1, 2]
    
    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, bounds, norm

parser = argparse.ArgumentParser()
parser.add_argument("start_dt_str",
                    help = "Start date/time of verification; format consistent with temporal_res flag, e.g., 'YYYYmmdd' for 24-hourly data")
parser.add_argument("end_dt_str",
                    help = "End date/time of verification; format consistent with temporal_res flag, e.g., 'YYYYmmdd' for 24-hourly data")
parser.add_argument("--region", dest = "region", default = "CONUS",
                    help = "Region to zoom plot to; default CONUS")
parser.add_argument("--temporal_res", default = 24, type = int, choices = [1, 3, 24],
                    help = "Temporal resolution of precip data used in verification (default 24)")
args = parser.parse_args()

data_names, truth_data_name, data_grid = precip_verification_processor.map_region_to_data_names(args.region)

verif = precip_verification_processor.PrecipVerificationProcessor(args.start_dt_str, 
                                                                  args.end_dt_str,
                                                                  data_names = data_names, 
                                                                  truth_data_name = truth_data_name, 
                                                                  data_grid = data_grid,
                                                                  region = args.region,
                                                                  temporal_res = args.temporal_res)


aris = xr.open_dataset("/data/bbasarab/netcdf/ARIs.ReplayGrid/ARI.ReplayGrid.02_year.24_hour_precipitation.nc").precip
aris_region_subset = verif._subset_data_to_region(aris, data_name = "ARI_grid")

ari_ex_dict = {}
for data_name, da in verif.da_dict.items():
    # Calculate ARI exceedances, sum over time period
    ari_ex_da_raw = xr.where(da > aris_region_subset, 1, 0, keep_attrs = True)
    ari_ex_da = ari_ex_da_raw.sum(dim = utils.period_end_time_dim_str, keepdims = True)

    # Add a period_end_time dimension to facilitate plotting
    end_dt = pd.Timestamp(da[utils.period_end_time_dim_str].values[-1])
    ari_ex_da.coords[utils.period_end_time_dim_str] = [end_dt]

    # Add relevant attributes for plotting
    pdp.add_attributes_to_data_array(ari_ex_da, short_name = "02year_24hour_ARIexcd", long_name = "02 year, 24 hour ARI Exceedances", units = "")

    # Add to dictionary
    ari_ex_dict[data_name] = ari_ex_da

# Plot cumulative ARI exceedances
cmap, bounds, norm = create_ari_colorbar()
verif.plot_cmap_multi_panel(data_dict = ari_ex_dict, plot_levels = bounds, cmap = cmap) 
 
