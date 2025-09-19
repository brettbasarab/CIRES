#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import precip_data_processors as pdp
import precip_plotting_utilities as ppu
import precip_verification_processor as pvp
import sys
import utilities as utils
import xarray as xr

utils.suppress_warnings()

truth_data_name = "AORC"
grid_cell_size = utils.nested_eagle_grid_cell_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fhr", dest = "fhr", type = int, default = 24, choices = [6, 12, 24, 48],
                        help = "Forecast hour from HRRR, GFS, and NestedEagle datasets; default 24")
    parser.add_argument("--plot", dest = "plot", action = "store_true", default = False,
                        help = "Set to make plots")
    parser.add_argument("--fss", dest = "fss", action = "store_true", default = False,
                        help = "Set to calculate FSS")
    parser.add_argument("--subregions", dest = "subregions", action = "store_true", default = False,
                        help = "Set to verify additional CONUS subregions; otherwise CONUS only")
    args = parser.parse_args()

    # Open AORC Dataset
    print("Reading datasets from zarr")
    aorc = xr.open_zarr("/Projects/ml_benchmarks/conus-precip-eval/15km/aorc.zarr", decode_timedelta=True)

    # Open GFS
    gfs = xr.open_zarr("/Projects/ml_benchmarks/conus-precip-eval/15km/gfs.forecasts.zarr", decode_timedelta=True)

    # Open HRRR
    hrrr = xr.concat(
        [
            xr.open_zarr(f"/Projects/ml_benchmarks/conus-precip-eval/15km/hrrr.forecasts.{fhr:02d}h.zarr", decode_timedelta=True)
            for fhr in [6, 12, 24, 48]
        ],
        dim = "fhr",
    )

    # Open Nested EAGLE
    nested = xr.open_zarr("/Projects/ml_benchmarks/conus-precip-eval/15km/nested-eagle.conus15km.precip.zarr/", decode_timedelta=True)

    # Open AORC mask
    mask = xr.open_dataset("/Projects/ml_benchmarks/conus-precip-eval/15km/mask.nc")["aorc_mask"]

    # Create dictionary of original datasets without any modifications
    data_names = ["AORC", "GFS", "HRRR", "NestedEagle"]
    ds_list = [aorc, gfs, hrrr, nested]
    ds_dict_orig = {}
    for d, data_name in enumerate(data_names):
        ds_dict_orig[data_name] = ds_list[d]

    # Correct valid times
    ds_dict = {}
    for data_name, xds in ds_dict_orig.items():
        if ("AORC" not in data_name):
            xds["valid_time"] = calc_valid_time(xds)

        # For NestedEagle, change valid_time from a data variable to a coordinate,
        # so that it can subsequently be swapped with t0 and become one of the DataArray's dimensions
        if ("Nested" in data_name):
            xds.coords["valid_time"] = xds.valid_time

        ds_dict[data_name] = xds
    
    # Trim the mask to the Nested Eagle domain
    trimmed_mask = mask.isel(x = slice(10, -11), y = slice(10, -11))
    trimmed_mask["x"] = np.arange(len(nested.x))
    trimmed_mask["y"] = np.arange(len(nested.y))

    # Trim the other datasets to the NestedEagle domain
    trimmed_ds_dict = {}
    for data_name, xds in ds_dict.items():
        if ("Nested" not in data_name):
            trimmed_xds = xds.isel(x = slice(10, -11), y = slice(10, -11))
            trimmed_xds["x"] = np.arange(len(nested.x))
            trimmed_xds["y"] = np.arange(len(nested.y))
            
            trimmed_ds_dict[data_name] = trimmed_xds
        else:
            trimmed_ds_dict[data_name] = xds

    # Now extract precipitation DataArrays at the given forecast hour from the Datasets. 
    # In so doing, swap dimensions 't0' (init time) and 'valid_time' so that the time
    # dimension will be correctly represented by valid times rather than init times.
    # Finally, rename 'valid_time' dimension (or 'time' for AORC) to 'period_end_time'.
    # This renaming is required for PrecipVerificationProcessor to work.
    trimmed_da_dict = {}
    for data_name, xds in trimmed_ds_dict.items(): 
        # Mask DataArray to the slightly-trimmed Nested Eagle domain
        # TODO: This step may be redundant since we've already trimmed the Datasets in trimmed_ds_dict
        da = xds.accum_tp.where(trimmed_mask)

        if "fhr" in da.dims:
            da = da.sel(fhr = args.fhr)

        if "t0" in da.dims:
            da = da.swap_dims({"t0": "valid_time"}).rename({"valid_time": "period_end_time"})
        else:
            da = da.rename({"time": "period_end_time"})

        da.attrs["data_name"] = data_name
        trimmed_da_dict[data_name] = da

    # OBSOLETE: To be deleted
    # Since Nested-Eagle has the shortest time dimension, select data for the
    # valid times from that DataArray from all the other DataArrays.
    # That is, these valid times should be common to all datasets.
    #common_period_end_times = trimmed_da_dict["NestedEagle"].period_end_time
    #for data_name, da in trimmed_da_dict.items():
        #if ("NestedEagle" not in data_name):
        #    da = da.sel(period_end_time = common_period_end_times)

    # Align arrays. Xarray.align is equivalent to numpy.intersect1d. This step
    # is necessary to align along the time ('period_end_time') dimensions.
    # NestedEagle has the shortest time dimension, AND depending on fhr the model
    # datasets may extend beyond the last time in the AORC dataset. By using
    # xarray.align, we select the set of time coordinates common to all datasets.
    aorc_da = trimmed_da_dict["AORC"]
    gfs_da = trimmed_da_dict["GFS"]
    hrrr_da = trimmed_da_dict["HRRR"]
    nested_da = trimmed_da_dict["NestedEagle"]
    aligned_das = xr.align(aorc_da, gfs_da, hrrr_da, nested_da)
    
    # Create final DataArrays that will work with PrecipVerificationProcessor
    da_dict = {}
    for da in aligned_das: 
        data_name = da.data_name

        da.coords["latitude"] = trimmed_mask.latitude
        da.coords["longitude"] = trimmed_mask.longitude
        da = da.rename({"latitude": "lat",
                        "longitude": "lon"})
        
        # Delete unneeded attributes
        for attr in list(da.attrs.keys()):
            del da.attrs[attr]
        
        # Add standard attributes expected by verification processor
        pdp.add_attributes_to_data_array(da,
                                         short_name = f"06-hour precipitation fhr{args.fhr:02d}",
                                         long_name = f"Precipitation accumulated over the prior 6 hours; forecast hour {args.fhr:02d}",
                                         units = "mm",
                                         interval_hours = 6) 

        da_dict[data_name] = da

    # Instantiate PrecipVerificationProcessor class
    start_dt = pd.to_datetime(da_dict["AORC"].period_end_time.values[0])
    end_dt = pd.to_datetime(da_dict["AORC"].period_end_time.values[-1])

    start_dt_str = start_dt.strftime("%Y%m%d.%H")
    end_dt_str = end_dt.strftime("%Y%m%d.%H")

    verif_conus = pvp.PrecipVerificationProcessor(start_dt_str,
                                                  end_dt_str, 
                                                  LOAD_DATA = False, 
                                                  loaded_non_subset_da_dict = False, 
                                                  USE_EXTERNAL_DA_DICT = True,
                                                  IS_STANDARD_INPUT_DICT = True,
                                                  external_da_dict = da_dict, 
                                                  data_names = data_names, 
                                                  truth_data_name = truth_data_name,
                                                  data_grid = "NestedEagle",
                                                  region = "CONUS", 
                                                  temporal_res = 6)

    verif_dict = {"CONUS": verif_conus}

    # REGIONAL STATS
    if args.subregions:
        # US-WestCoast
        verif_west_coast = create_subset_region_verif_processor(verif_conus, "US-WestCoast")
        verif_dict["US-WestCoast"] = verif_west_coast 

        # US-Mountain
        verif_mountain = create_subset_region_verif_processor(verif_conus, "US-Mountain")
        verif_dict["US-Mountain"] = verif_mountain

        # US-Central
        verif_central = create_subset_region_verif_processor(verif_conus, "US-Central")
        verif_dict["US-Central"] = verif_central

        # US-EastCoast
        verif_east = create_subset_region_verif_processor(verif_conus, "US-East")
        verif_dict["US-East"] = verif_east

    # PLOTTING: Monthly means 
    if args.plot:
        plot_monthly_means(verif_conus)
        if args.subregions:
            plot_monthly_means(verif_west_coast) 
            plot_monthly_means(verif_mountain) 
            plot_monthly_means(verif_central) 
            plot_monthly_means(verif_east)

    # FSS
    if args.fss:
        calculate_and_plot_fss(verif_conus, plot = args.plot)
        if args.subregions:
            calculate_and_plot_fss(verif_west_coast, plot = args.plot)
            calculate_and_plot_fss(verif_mountain, plot = args.plot)
            calculate_and_plot_fss(verif_central, plot = args.plot)
            calculate_and_plot_fss(verif_east, plot = args.plot)

    return ds_dict_orig, ds_dict, trimmed_ds_dict, trimmed_da_dict, verif_dict

def calc_valid_time(xds):
    lead_time = xr.DataArray([pd.Timedelta(hours = fhr) for fhr in xds.fhr.values],
                             coords=xds.fhr.coords)
    return xds["t0"] + lead_time

def create_subset_region_verif_processor(verif, region):
    match region:
        case "US-WestCoast":
            region_mask = ppu.create_west_coast_states_mask(verif.truth_da)
            x_slice = None
        case "US-Mountain":
            region_mask = ppu.create_mountain_states_mask(verif.truth_da)
            x_slice = None
        case "US-Central":
            region_mask = ppu.create_conus_mask(verif.truth_da)
            x_slice = (135, 250)
        case "US-East":
            region_mask = ppu.create_conus_mask(verif.truth_da)
            x_slice = (250, verif.truth_da.x[-1].item())
        case _:
            region_mask = ppu.create_conus_mask(verif.truth_da)
            x_slice = None

    da_dict = {}
    for data_name, da in verif.da_dict.items():
        # Slice
        if x_slice is not None:
            da = da.isel(x = slice(x_slice[0], x_slice[1]))

        # Mask
        da_dict[data_name] = da.where(region_mask)

    verif_subregion = pvp.PrecipVerificationProcessor(verif.start_dt_str,
                                                      verif.end_dt_str,
                                                      LOAD_DATA = False,
                                                      loaded_non_subset_da_dict = False,
                                                      USE_EXTERNAL_DA_DICT = True,
                                                      IS_STANDARD_INPUT_DICT = True,
                                                      external_da_dict = da_dict,
                                                      data_names = verif.data_names,
                                                      truth_data_name = verif.truth_data_name,
                                                      data_grid = "NestedEagle",
                                                      region = region, 
                                                      temporal_res = 6)

    return verif_subregion

def plot_monthly_means(verif, plot_cmaps = True, plot_errors = True, write_to_nc = False):
    agg_dict_ts = verif.calculate_aggregated_stats(time_period_type = "monthly", stat_type = "mean", agg_type = "space_time",
                                                   write_to_nc = write_to_nc)
    verif.plot_timeseries(data_dict = agg_dict_ts, time_period_type = "monthly", stat_type = "mean",
                          plot_levels = np.arange(0, 1.8, 0.2), full_short_name_in_fig_name = True)
    
    if plot_cmaps:
        agg_dict_cmaps = verif.calculate_aggregated_stats(time_period_type = "monthly", stat_type = "mean", agg_type = "time",
                                                          write_to_nc = write_to_nc)
        verif.plot_cmap_multi_panel(data_dict = agg_dict_cmaps, time_period_type = "monthly", stat_type = "mean",
                                    plot_levels = np.arange(0, 5.5, 0.5))
        if plot_errors:
            verif.plot_cmap_multi_panel(data_dict = agg_dict_cmaps, time_period_type = "monthly", stat_type = "mean",
                                        plot_levels = np.arange(0, 5.5, 0.5), plot_errors = True, single_colorbar = False)

    if plot_cmaps:
        return agg_dict_ts, agg_dict_cmaps
    else:
        return agg_dict_ts, {}

def calculate_and_plot_fss(verif, plot = True, write_to_nc = False, by_amount_threshold_only = False):
    # FSS by amount threshold, fixed eval radius
    fss_threshold_mm_dict = verif.calculate_fss(eval_type = "by_threshold",
                                                grid_cell_size = grid_cell_size,
                                                fixed_radius = 2 * grid_cell_size,
                                                eval_threshold_list = utils.default_eval_threshold_list_mm,
                                                is_pctl_threshold = False,
                                                write_to_nc = write_to_nc)
    if plot:
        verif.plot_aggregated_fss(eval_type = "by_threshold", is_pctl_threshold = False, include_frequency_bias = True)

    if by_amount_threshold_only:
        return fss_threshold_mm_dict
    
    # FSS by percentile threshold, fixed eval radius
    fss_threshold_pctl_dict = verif.calculate_fss(eval_type = "by_threshold",
                                                  grid_cell_size = grid_cell_size,
                                                  fixed_radius = 2 * grid_cell_size,
                                                  eval_threshold_list = utils.default_eval_threshold_list_pctl,
                                                  is_pctl_threshold = True,
                                                  write_to_nc = write_to_nc)
    if plot:
        verif.plot_aggregated_fss(eval_type = "by_threshold", is_pctl_threshold = True)

    # FSS by eval radius, fixed amount threshold
    fss_radius_thresh_mm_dict = verif.calculate_fss(eval_type = "by_radius",
                                                    grid_cell_size = grid_cell_size,
                                                    fixed_threshold = 10,
                                                    eval_radius_list = grid_cell_size * utils.default_eval_radius_list_grid_cells,
                                                    is_pctl_threshold = False,
                                                    write_to_nc = write_to_nc)
    if plot:
        verif.plot_aggregated_fss(eval_type = "by_radius", is_pctl_threshold = False)

    # FSS by eval radius, fixed percentile threshold
    fss_radius_thresh_pctl_dict = verif.calculate_fss(eval_type = "by_radius",
                                                      grid_cell_size = grid_cell_size,
                                                      fixed_threshold = 95.0,
                                                      eval_radius_list = grid_cell_size * utils.default_eval_radius_list_grid_cells,
                                                      is_pctl_threshold = True,
                                                      write_to_nc = write_to_nc)
    if plot:
        verif.plot_aggregated_fss(eval_type = "by_radius", is_pctl_threshold = True)

    return fss_threshold_mm_dict, fss_threshold_pctl_dict, fss_radius_thresh_mm_dict, fss_radius_thresh_pctl_dict
    
def create_subset_time_verif_processor(verif, sel_string):
    da_dict = {}
    for data_name, da in verif.da_dict.items():
      da_dict[data_name] = da.sel(period_end_time = sel_string)

    start_dt = pd.Timestamp(da_dict["AORC"].period_end_time.values[0])
    end_dt = pd.Timestamp(da_dict["AORC"].period_end_time.values[-1])

    return pvp.PrecipVerificationProcessor(start_dt.strftime("%Y%m%d.%H"),
                                           end_dt.strftime("%Y%m%d.%H"),
                                           LOAD_DATA = False,
                                           USE_EXTERNAL_DA_DICT = True,
                                           external_da_dict = da_dict,
                                           data_names = ["AORC", "GFS", "HRRR", "NestedEagle"],
                                           data_grid = "NestedEagle",
                                           region = verif.region, 
                                           temporal_res = 6)

def calculate_pctl_exceedances(verif, pctl_val = 95.0, exclude_zeros = True):
    period_end_times = verif.truth_da.period_end_time
    pctl_excd_dict = {}
    for data_name, da in verif.da_dict.items():
        if exclude_zeros:
            quantile_da = da.where(da > 0.0).quantile(pctl_val/100.0, dim = ("y", "x"))
        else:
            quantile_da = da.quantile(pctl_val/100.0, dim = ("y", "x"))
        excd_da = da.where(da > quantile_da)
        pctl_excd_dict[data_name] = excd_da

        print(f"****** {data_name}")
        print(quantile_da.shape)
        for i, val in enumerate(quantile_da.values):
            print(f"{pd.Timestamp(period_end_times.values[i]).strftime('%Y%m%d.%H')}: {val:0.3f}")
        
    return pctl_excd_dict 
                    
if __name__ == "__main__":
    ds_dict_orig, ds_dict, trimmed_ds_dict, trimmed_da_dict, verif_dict = main()
