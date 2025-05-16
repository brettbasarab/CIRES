#!/usr/bin/env python

import argparse
import numpy as np
import os
import precip_plotting_utilities as pputils
import utilities as utils
import xarray as xr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("valid_dt_str", 
                        help = "Valid date/time string (format YYYYmmdd)")
    parser.add_argument("--multi_panel", dest = "multi_panel", default = False, action = "store_true",
                        help = "Set to plot multi-panel plot")
    args = parser.parse_args()

    plot_name = "HAFS_FF.NativeGrid"
    region = "CONUS"
    native_temporal_res = 1
    output_temporal_res = 24
    use_contourf = False

    ##### Single panel plot #####
    if not(args.multi_panel):
        # Read data
        fpath = f"/data/bbasarab/netcdf/HAFS_FF.NativeGrid.01_hour_precipitation/*{args.valid_dt_str}*.nc"
        print(f"Reading {fpath}")
        da = xr.open_mfdataset(fpath).accum_precip

        # Create 24-hour accumulated precip DataArray
        print("Calculating accumulated precip")
        accum_precip_da = calculate_accum_precip(da, output_temporal_res, native_temporal_res)

        # Plot data
        print("Plotting data, single-panel plot")
        pputils.plot_cmap_single_panel(accum_precip_da, plot_name, region, use_contourf = use_contourf, temporal_res = output_temporal_res)
    ##### Multi panel plot #####
    else:
        # Read data
        print("Reading data")
        data_names = ["AORC.NativeGrid", "CONUS404.AorcGrid", "HAFS_FF.NativeGrid", "NestedReplay.AorcGrid"]
        data_dict = {}
        for data_name in data_names:
            if ("HAFS" in data_name):
                fpath = os.path.join(utils.data_nc_dir, f"{data_name}.{native_temporal_res:02d}_hour_precipitation", f"*{args.valid_dt_str}*.nc")
                print(f"Reading {fpath}")
                da = xr.open_mfdataset(fpath).accum_precip
                accum_precip = calculate_accum_precip(da, output_temporal_res, native_temporal_res)
            elif ("NestedReplay" in data_name): 
                fpath = os.path.join(utils.data_nc_dir, f"{data_name}.{output_temporal_res:02d}_hour_precipitation", f"*{args.valid_dt_str}*.nc")
                print(f"Reading {fpath}")
                accum_precip = xr.open_mfdataset(fpath).precipitation_24_hour
            else:
                fpath = os.path.join(utils.data_nc_dir, f"{data_name}.{output_temporal_res:02d}_hour_precipitation", f"*{args.valid_dt_str}*.nc")
                print(f"Reading {fpath}")
                accum_precip = xr.open_mfdataset(fpath).precipitation_24_hour

            data_dict[data_name] = accum_precip

        print("Plotting data, multi-panel plot")
        pputils.plot_cmap_multi_panel(data_dict, "AORC.NativeGrid", region, np.arange(0,160,10))

def calculate_accum_precip(da, output_temporal_res, native_temporal_res):
    da = da.rename({"time":"period_end_time"})
    roller = da.rolling({"period_end_time": output_temporal_res})
    time_step = int(output_temporal_res/native_temporal_res)
    accum_precip_da = roller.sum()[(time_step - 1)::time_step,:,:]
    accum_precip_da.attrs["short_name"] = f"{output_temporal_res:02d}_hour_precipitation"

    return accum_precip_da

if __name__ == "__main__":
    main()
