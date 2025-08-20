#!/usr/bin/env python

import argparse
import datetime as dt
import numpy as np
import pandas as pd
import precip_plotting_utilities as pputils
import precip_verification_processor
import sys
import utilities as utils
import xarray as xr

# TODO: Much of what's done here reproduces what's in the calculate_aggregated_stats method.
# Ultimately, that method should be able to calculate the exceedance dictionaries and/or
# the mean exceedances calculated below. (Then, also, the private methods below won't need to be used.)
# But keep as is for now, since we're still determining the validity/usefulness of the percentile exceedance statistic. 

utils.suppress_warnings()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_dt_str",
                        help = "Start date/time string")
    parser.add_argument("end_dt_str",
                        help = "End date/time string")
    parser.add_argument("--regions", dest = "regions", nargs = "+", default = ["CONUS"],
                        help = "Regions for which to perform verification; default CONUS")
    parser.add_argument("--pctl", dest = "pctl", type = float, default = 99,
                        help = "Percentile to use for exceedances; default 99th")
    parser.add_argument("--time_period_type", dest = "time_period_type", default = "monthly",
                        help = "Time period type: monthly, seasonal, etc.; default  monthly")
    parser.add_argument("--exclude_zeros", dest = "exclude_zeros", action = "store_true", default = False,
                        help = "Set to exclude zeros from percentile calculations; default False")
    parser.add_argument("--plot_cmaps", dest = "plot_cmaps", action = "store_true", default = False,
                        help = "Set to plot contour maps of percentiles for each time period, and percentile exceedances")
    parser.add_argument("--excd_type", dest = "excd_type", default = "mdl_excd_obs_pctl",
                        choices = ["obs_excd_obs_pctl", "mdl_excd_obs_pctl", "mdl_excd_mdl_pctl"],
                        help = "Type of exceedances for which to calculate means (default mdl_excd_obs_pctl); \n"
                               "obs_excd_obs_pctl: Model values where obs exceedances of obs_pctl occur;\n"
                               "mdl_excd_obs_pctl: Model's own exceedances of obs_pctl;\n"
                               "mdl_excd_mdl_pctl: Model's own exceedances of its own pctl;\n"
                               "default mdl_excd_obs_pctl")
    args = parser.parse_args()

    for r, region in enumerate(args.regions):
        LOAD_DATA = True
        loaded_non_subset_da_dict = None 
        if (r > 0):
            LOAD_DATA = False
            loaded_non_subset_da_dict = verif.loaded_non_subset_da_dict # From previous instantiation of PrecipVerificationProcessor

        # Instantiate PrecipVerificationProcessor class
        verif = precip_verification_processor.PrecipVerificationProcessor(args.start_dt_str,
                                                                          args.end_dt_str, 
                                                                          LOAD_DATA = LOAD_DATA,
                                                                          loaded_non_subset_da_dict = loaded_non_subset_da_dict,
                                                                          data_names = ["AORC", "CONUS404", "ERA5", "HRRR", "IMERG", "Replay"],
                                                                          truth_data_name = "AORC",
                                                                          data_grid = "Replay",
                                                                          temporal_res = 24,
                                                                          region = region) 

        # (Not using for now)
        # Over a longer time period, if you use common_monthly or common_seasonal
        # to calculate percentiles, the percentiles represent something close to a climatology (?)
        # as opposed to the percentiles specific to a particular month or season.
        if args.time_period_type == "monthly":
            pctl_agg_type = "common_monthly"
        elif args.time_period_type == "seasonal":
            pctl_agg_type = "common_seasonal"

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
                # Observed data
                obs_da_sel_time_period = verif._determine_agg_data_from_time_period_type(obs_da_period_begin,
                                                                                         args.time_period_type,
                                                                                         dtime)

                # Model data
                da_sel_time_period = verif._determine_agg_data_from_time_period_type(data_array,
                                                                                     args.time_period_type,
                                                                                     dtime)

                # Retain or exclude zeros in data arrays for percentile calculations
                if args.exclude_zeros:
                    obs_da_for_pctls = obs_da_sel_time_period.where(obs_da_sel_time_period > 0.0)
                    model_da_for_pctls = da_sel_time_period.where(da_sel_time_period > 0.0)
                    exclude_zeros_str = ".exclude_zeros"
                else:
                    obs_da_for_pctls = obs_da_sel_time_period
                    model_da_for_pctls = da_sel_time_period
                    exclude_zeros_str = ""
     
                # Observed percentile
                obs_pctl_sel_time_period = obs_da_for_pctls.quantile(args.pctl/100, keep_attrs = True,
                                                                     dim = utils.period_begin_time_dim_str)

                # Model percentile
                da_pctl_sel_time_period = model_da_for_pctls.quantile(args.pctl/100, keep_attrs = True,
                                                                      dim = utils.period_begin_time_dim_str)

                # Plotting
                if (args.plot_cmaps):
                    print(f"Plotting observed and modeled percentiles for dtime {dtime}")
                    if (type(dtime) is dt.datetime) or (type(dtime) is pd.Timestamp):
                        dt_str = f"{dtime:%Y%m%d.%H}"
                    elif (type(dtime) is str):
                        dt_str = dtime
                    else:
                        dt_str = ""
     
                    if (data_name == "AORC"):
                        pputils.plot_cmap_single_panel(obs_pctl_sel_time_period, 
                                                       f"{data_name}.obs_pctl{exclude_zeros_str}.{dt_str}",
                                                       f"{data_name}.obs_pctl{exclude_zeros_str}.{dt_str}",
                                                       region,
                                                       plot_levels = np.arange(0, 62, 2))
                    else:
                        pputils.plot_cmap_single_panel(da_pctl_sel_time_period, 
                                                       f"{data_name}.da_pctl{exclude_zeros_str}.{dt_str}",
                                                       f"{data_name}.da_pctl{exclude_zeros_str}.{dt_str}",
                                                       region,
                                                       plot_levels = np.arange(0, 62, 2))

                # This DataArray's values where obs exceedances of obs_pctl occur for this time period 
                if (args.excd_type == "obs_excd_obs_pctl"):
                    excd_da = da_sel_time_period.where(obs_da_sel_time_period >= obs_pctl_sel_time_period)
                # This DataArray's own exceedances of obs_pctl for this time period
                elif (args.excd_type == "mdl_excd_obs_pctl"):
                    excd_da = da_sel_time_period.where(da_sel_time_period >= obs_pctl_sel_time_period)
                # This DataArray's own exceedances of its own pctl for this time period 
                elif (args.excd_type == "mdl_excd_mdl_pctl"):
                    excd_da = da_sel_time_period.where(da_sel_time_period >= da_pctl_sel_time_period) 
                else:
                    print(f"Error: Unknown model/obs percentile exceedance type {args.excd_type}")
                    sys.exit(1) 
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

        # Contour maps of given type of exceedances masked to just those values (all other values,
        # i.e., non-exceedances, are NaNs)
        if args.plot_cmaps:
            print(f"Plotting percentile exceedances")
            verif.plot_cmap_multi_panel(data_dict = excd_dict, plot_levels = np.arange(0, 62, 2))

    return verif

if __name__ == "__main__":
    verif = main()

##### OLD/SCRATCH #####
"""
excd_dict[data_name] = da.where(obs_da >= obs_pctl)
for data_name, da in excd_dict.items():
    rmse = verif.calculate_rmse(da, excd_dict[verif.truth_data_name])
    bias = verif.calculate_bias(da, excd_dict[verif.truth_data_name])
    print(f"{data_name} RMSE: {rmse:0.2f}mm, Bias: {bias:0.2f}mm")

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
