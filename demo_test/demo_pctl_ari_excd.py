#!/usr/bin/env python

import argparse
import datetime as dt
import numpy as np
import pandas as pd
import precip_verification_processor
import sys
import utilities as utils
import xarray as xr

# TODO: Much of what's done here reproduces what's in the calculate_aggregated_stats method.
# Ultimately, that method should be able to calculate the exceedance dictionaries and/or
# the mean exceedances calculated below. (Then, also, the private methods below won't need to be used.)
# But keep as is for now, since we're still determining the validity/usefulness of the percentile exceedance statistic. 

utils.suppress_warnings()


parser = argparse.ArgumentParser()
parser.add_argument("start_dt_str",
                    help = "Start date/time string")
parser.add_argument("end_dt_str",
                    help = "End date/time string")
parser.add_argument("--pctl", dest = "pctl", type = float, default = 99,
                    help = "Percentile to use for exceedances; default  99th")
parser.add_argument("--time_period_type", dest = "time_period_type", default = "monthly",
                    help = "Time period type: monthly, seasonal, etc.")
parser.add_argument("--excd_type", dest = "excd_type", default = "at_obs_excd", choices = ["at_obs_excd", "at_own_excd"],
                    help = "Type of exceedances for which to calculate means (default at_obs_excd)")
args = parser.parse_args()

# Instantiate PrecipVerificationProcessor class
verif = precip_verification_processor.PrecipVerificationProcessor(args.start_dt_str,
                                                                  args.end_dt_str, 
                                                                  LOAD_DATA = True,
                                                                  data_names = ["AORC", "CONUS404", "ERA5", "HRRR", "IMERG", "Replay"],
                                                                  truth_data_name = "AORC",
                                                                  data_grid = "Replay",
                                                                  temporal_res = 24,
                                                                  region = "US-East")

# (Not using for now)
# Over a longer time period, if you use common_monthly or common_seasonal
# to calculate percentiles, the percentiles represent something close to a climatology (?)
# as opposed to the percentiles specific to a particular month or season.
if args.time_period_type == "monthly":
    pctl_agg_type = "common_monthly"
elif args.time_period_type == "seasonal":
    pctl_agg_type = "common_seasonal"

# Calculate 99th percentiles
agg_dict = verif.calculate_aggregated_stats(time_period_type = args.time_period_type, stat_type = "pctl", agg_type = "time", pctl = 99)

# Plot percentiles for each time period 
#verif.plot_cmap_multi_panel(data_dict = agg_dict)

# Get observed 99th percentiles (AORC)
obs_pctl = agg_dict[verif.truth_data_name]

# Get observed data grid (AORC)
obs_da = verif.truth_da 
obs_da_period_begin = utils.convert_period_end_to_period_begin(obs_da)

# Get points of all grids where observed percentile exceedances occur
print("Calculating exceedances")
dtimes, dim_name, time_period_str = verif._process_time_period_type_to_dtimes(args.time_period_type)
if (args.time_period_type == "seasonal"):
    dtimes = verif._construct_season_dt_ranges(obs_da)

excd_dict = {} 
for data_name, da in verif.da_dict.items():
    data_array = utils.convert_period_end_to_period_begin(da)
    final_da_list = []
    for dtime in dtimes:
        obs_pctl_sel_time_period = verif._determine_agg_data_from_time_period_type(obs_da_period_begin,
                                                                                   args.time_period_type,
                                                                                   dtime).quantile(args.pctl/100,
                                                                                                   dim = utils.period_begin_time_dim_str)
        obs_da_sel_time_period = verif._determine_agg_data_from_time_period_type(obs_da_period_begin,
                                                                                 args.time_period_type,
                                                                                 dtime) 
        da_sel_time_period = verif._determine_agg_data_from_time_period_type(data_array,
                                                                             args.time_period_type,
                                                                             dtime)

        # This DataArray's values where obs exceedances of obs_pctl occur for this time period 
        if (args.excd_type == "at_obs_excd"):
           excd_da = da_sel_time_period.where(obs_da_sel_time_period >= obs_pctl_sel_time_period)
        # This DataArray's own exceedances of obs_pctl for this time period
        else:
           excd_da = da_sel_time_period.where(da_sel_time_period >= obs_pctl_sel_time_period) 
        final_da_list.append(excd_da)

    final_da = xr.concat(final_da_list, dim = "period_begin_time")
    excd_dict[data_name] = utils.convert_period_begin_to_period_end(final_da)

# Aggregate exceedances dict to daily means and plot time series
agg_excd_dict = verif.calculate_aggregated_stats(input_da_dict = excd_dict, 
                                                 time_period_type = args.time_period_type,
                                                 stat_type = "mean",
                                                 agg_type = "space_time")

verif.plot_timeseries(data_dict = agg_excd_dict,
                      time_period_type = args.time_period_type,
                      stat_type = "pctl_excd_mean",
                      pctl = args.pctl, 
                      plot_levels = np.arange(0, 85, 5),
                      write_stats = False)

##### OLD/SCRATCH #####
"""
obs_pctl_excd = obs_da.where(obs_da >= obs_pctl)
excd_dict[data_name] = da.where(obs_da >= obs_pctl)

for data_name, da in excd_dict.items():
    rmse = verif.calculate_rmse(da, excd_dict[verif.truth_data_name])
    bias = verif.calculate_bias(da, excd_dict[verif.truth_data_name])
    print(f"{data_name} RMSE: {rmse:0.2f}mm, Bias: {bias:0.2f}mm")

# Contour maps of obs exceedances of obs 99th percentile and the other datasets are "masked" to just these values
verif.plot_cmap_multi_panel(da_dict = excd_dict)

for data_name, da in verif.da_dict.items():
    da = utils.convert_period_end_to_period_begin(da, temporal_res = 24)
    final_da_list = []
    for m, month in enumerate(obs_pctl.months.dt.month.values):
       obs_da_sel_time_period = obs_da_period_begin.sel(period_begin_time = da.period_begin_time.dt.month.isin([month]))
       da_sel_time_period = da.sel(period_begin_time = da.period_begin_time.dt.month.isin([month]))
       if (args.excd_type == "at_obs_excd"): # This DataArray's values where obs exceedances of obs_pctl occur for this time period 
           excd_da = da_sel_time_period.where(obs_da_sel_time_period >= obs_pctl[m,:,:])
       else: # This DataArray's own exceedances of obs_pctl for this time period
           excd_da = da_sel_time_period.where(da_sel_time_period >= obs_pctl[m,:,:]) 
       final_da_list.append(excd_da)
    final_da = xr.concat(final_da_list, dim = "period_begin_time")
    excd_dict[data_name] = utils.convert_period_begin_to_period_end(final_da)

# TESTING: September example
obs_da_september = utils.convert_period_end_to_period_begin(verif.da_dict["AORC"]).sel(period_begin_time = "2017-09")
obs_pctl_full_period = obs_da.quantile(0.99, dim = "period_end_time")
september_obs_pctl = obs_pctl.sel(months = "2017-09-01")

# "Correct" version: mean of values of each DataArray where OBS data exceeds OBS percentile
for data_name, da in verif.da_dict.items():
  da = utils.convert_period_end_to_period_begin(da)
  da_september = da.sel(period_begin_time = "2017-09")
  da_excd_full_period = da_september.where(obs_da_september >= obs_pctl_full_period)
  da_excd_september = da_september.where(obs_da_september >= september_obs_pctl)
  print(f"***** {data_name}")
  print(da_excd_full_period.mean().item())
  print(da_excd_september.mean().item())

# "Incorrect" but interesting version: mean of values of each DataArray where THAT DataArray exceeds OBS percentile
# This approach could account for spatial displacement issues: Don't worry about exceedances being in the right place
# which the previous approach demands (WHERE the obs exceedances occur). Rather, wherever there are exceedances of observed
# pctls for this dataset, how do they compare (in a quantitative but aggregated sense) to observed pctl exceedances? 
for data_name, da in verif.da_dict.items():
  da = utils.convert_period_end_to_period_begin(da)
  da_september = da.sel(period_begin_time = "2017-09")
  da_excd_full_period = da_september.where(da_september >= obs_pctl_full_period)
  da_excd_september = da_september.where(da_september >= september_obs_pctl)
  print(f"***** {data_name}")
  print(da_excd_full_period.mean().item())
  print(da_excd_september.mean().item())
"""
