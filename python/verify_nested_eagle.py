#!/usr/bin/env python

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

data_names = ["AORC", "GFS", "HRRR", "NestedEagle"]
truth_data_name = "AORC"
fhr = 24 # Which forecast hour to evaluate
# t0 = "2023-03-10T06"
# vtime = pd.Timestamp(t0) + pd.Timedelta(hours = fhr)
# stime = str(vtime)[:13]

def main():
    # Open AORC Dataset
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
    ds_list = [aorc, gfs, hrrr, nested]
    ds_dict = {}
    for d, data_name in enumerate(data_names):
        ds_dict[data_name] = ds_list[d]
    #ds_dict = {"AORC": aorc, "GFS": gfs, "HRRR": hrrr, "NestedEagle": nested}

    # Correct valid times
    gfs["valid_time"] = calc_valid_time(gfs)
    hrrr["valid_time"] = calc_valid_time(hrrr)
    nested["valid_time"] = calc_valid_time(nested)

    # For nested array, change valid_time from a data variable
    # to a coordinate, so that it can subsequently be swapped with t0 and become
    # one of the DataArray's dimensions
    nested.coords["valid_time"] = nested.valid_time

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

    # Extract DataArrays with time masked to Nested-Eagle grid and with
    # time dimensions renamed to 'period_end_time' 
    da_dict_tmp = {}
    for data_name, xds in trimmed_ds_dict.items(): 
        # Old masking (from Tim's notebook); makes other DataArrays a slightly different shape from NestedEagle
        #if "Nested" not in data_name:
        #    da = xds.accum_tp.where(mask)
        #else:
        #    da = xds.accum_tp.where(trimmed_mask)

        # Mask DataArray to the slightly-trimmed Nested Eagle domain
        da = xds.accum_tp.where(trimmed_mask)

        if "fhr" in da.dims:
            da = da.sel(fhr = fhr)

        if "t0" in da.dims:
            da = da.swap_dims({"t0": "valid_time"}).rename({"valid_time": "period_end_time"})
        else:
            da = da.rename({"time": "period_end_time"})

        da_dict_tmp[data_name] = da

    # Since Nested-Eagle has the shortest time dimension, select data for the valid
    # times from that DataArray from all the other DataArrays.
    # These valid times should be common to all datasets.
    common_period_end_times = da_dict_tmp["NestedEagle"].period_end_time

    # Create final DataArrays that will work with PrecipVerificationProcessor
    da_dict = {}
    for data_name, da in da_dict_tmp.items():
        if (data_name != "NestedEagle"):
            da = da.sel(period_end_time = common_period_end_times)

        da.coords["latitude"] = trimmed_mask.latitude
        da.coords["longitude"] = trimmed_mask.longitude
        da = da.rename({"latitude": "lat",
                        "longitude": "lon"})
        
        # Delete unneeded attributes
        for attr in list(da.attrs.keys()):
            del da.attrs[attr]
        
        # Add standard attributes expected by verification processor
        pdp.add_attributes_to_data_array(da,
                                         short_name = "06-hour precipitation",
                                         long_name = "Precipitation accumulated over the prior 6 hours",
                                         units = "mm",
                                         interval_hours = 6) 

        da_dict[data_name] = da

    # Instantiate PrecipVerificationProcessor class
    start_dt = pd.to_datetime(da_dict["AORC"].period_end_time.values[0])
    end_dt = pd.to_datetime(da_dict["AORC"].period_end_time.values[-1])

    start_dt_str = start_dt.strftime("%Y%m%d.%H")
    end_dt_str = end_dt.strftime("%Y%m%d.%H")

    verif = pvp.PrecipVerificationProcessor(start_dt_str,
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

    
    plot_monthly_means(verif)
    sys.exit(0)

    ##### Regional stats #####
    # US-WestCoast
    verif_west_coast = create_subset_region_verif_processor(verif, "US-WestCoast") 
    plot_monthly_means(verif_west_coast) 

    # US-Mountain
    verif_mountain = create_subset_region_verif_processor(verif, "US-Mountain")
    plot_monthly_means(verif_mountain) 

    # US-Central
    verif_central = create_subset_region_verif_processor(verif, "US-Central")
    plot_monthly_means(verif_central) 

    # US-EastCoast
    verif_east = create_subset_region_verif_processor(verif, "US-East") 
    plot_monthly_means(verif_east)

    return verif, ds_dict, trimmed_ds_dict

def calc_valid_time(xds):
    lead_time = xr.DataArray(
        [pd.Timedelta(hours=fhr) for fhr in xds.fhr.values],
        coords=xds.fhr.coords,
    )
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

def plot_monthly_means(verif, plot_cmaps = True, plot_errors = True):
    agg_dict_ts = verif.calculate_aggregated_stats(time_period_type = "monthly", stat_type = "mean", agg_type = "space_time")
    verif.plot_timeseries(data_dict = agg_dict_ts, time_period_type = "monthly", stat_type = "mean",
                          plot_levels = np.arange(0, 1.8, 0.2))
    
    if plot_cmaps:
        agg_dict_cmaps = verif.calculate_aggregated_stats(time_period_type = "monthly", stat_type = "mean", agg_type = "time")
        verif.plot_cmap_multi_panel(data_dict = agg_dict_cmaps, time_period_type = "monthly", stat_type = "mean",
                                    plot_levels = np.arange(0, 5.5, 0.5))
        if plot_errors:
            verif.plot_cmap_multi_panel(data_dict = agg_dict_cmaps, time_period_type = "monthly", stat_type = "mean",
                                        plot_levels = np.arange(0, 5.5, 0.5), plot_errors = True, single_colorbar = False)


if __name__ == "__main__":
    verif, ds_dict, trimmed_ds_dict = main()
