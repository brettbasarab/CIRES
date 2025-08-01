#!/usr/bin/env python

import datetime as dt
import numpy as np
import precip_verification_processor
import utilities as utils
import xarray as xr

# Instantiate PrecipVerificationProcessor class
verif = precip_verification_processor.PrecipVerificationProcessor("20170101",
                                                                  "20180101",
                                                                  LOAD_DATA = True,
                                                                  data_names = ["AORC", "CONUS404", "ERA5", "HRRR", "IMERG", "Replay"],
                                                                  truth_data_name = "AORC",
                                                                  data_grid = "Replay",
                                                                  temporal_res = 24,
                                                                  region = "US-East")

# Calculate 99th percentiles
# TODO: I think percentiles should be calculated with respect to each time period evaluated on time series below
# For example, for each season for seasonal time series
agg_dict = verif.calculate_aggregated_stats(time_period_type = "full_period", stat_type = "pctl", agg_type = "time", pctl = 99)

# Plot percentiles for this time period
verif.plot_cmap_multi_panel(data_dict = agg_dict)

# Get observed 99th percentiles (AORC)
obs_pctl99 = agg_dict[verif.truth_data_name].squeeze()

# Get observed data grid (AORC)
obs_da = verif.da_dict[verif.truth_data_name]

# Get percentile exceedances in observed grid
print("Calculating exceedances")
#obs_pctl_excd = obs_da.where(obs_da >= obs_pctl99)

# Get points of other grids where observed percentile exceedances occur
excd_dict = {} 
for data_name, da in verif.da_dict.items():
    excd_dict[data_name] = da.where(obs_da >= obs_pctl99)

for data_name, da in excd_dict.items():
    rmse = verif.calculate_rmse(da, excd_dict[verif.truth_data_name])
    bias = verif.calculate_bias(da, excd_dict[verif.truth_data_name])
    print(f"{data_name} RMSE: {rmse:0.2f}mm, Bias: {bias:0.2f}mm")

# Contour maps of obs exceedances of obs 99th percentile and the other datasets are "masked" to just these values
#verif.plot_cmap_multi_panel(da_dict = excd_dict)

# Aggregate exceedances dict to daily means and plot time series
# TODO: stats need to be properly labelled in time series figure name to differentiate
# from standard mean of all data stats
agg_excd_dict = verif.calculate_aggregated_stats(input_da_dict = excd_dict, 
                                                 time_period_type = "monthly",
                                                 stat_type = "mean",
                                                 agg_type = "space_time")
verif.plot_timeseries(data_dict = agg_excd_dict, time_period_type = "monthly", plot_levels = np.arange(0, 64, 4), write_stats = False)

