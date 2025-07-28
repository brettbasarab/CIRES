#!/usr/bin/env python

import argparse
import datetime as dt
import numpy as np
import os
import precip_data_processors as pdp
import sys
import utilities as utils
import xarray as xr
 
def main():
    program_description = ("Program to convert Tim Smith's HRRR precip 'reanalysis' for zarr to netCDF.")
    parser = argparse.ArgumentParser(description = program_description) 
    parser.add_argument("start_dt_str",
                        help = "Start date/time to output; format YYYYmmdd.HH, PERIOD ENDING")
    parser.add_argument("end_dt_str",
                        help = "End date/time to output; format YYYYmmdd.HH, PERIOD ENDING")
    parser.add_argument("--input_temporal_res", dest = "input_temporal_res", type = int, default = 1, choices = [1, 6],
                        help = "Temporal resolution of HRRR precip data to read from zarr")
    parser.add_argument("--output_temporal_res", dest = "output_temporal_res", type = int, default = 24,
                        help = "Temporal resolution of summed HRRR precip data to output to netCDF.")
    parser.add_argument("-t", "--testing", dest = "testing", action = "store_true", default = False,
                        help = "Testing mode: Configure netCDF output file paths but don't write")
    args = parser.parse_args()
    utils.suppress_warnings()

    if (args.output_temporal_res < args.input_temporal_res):
        print(f"Error: Can't create {args.output_temporal_res}-hourly data from {args.input_temporal_res}-hourly data")
        sys.exit(1)

    start_dt = dt.datetime.strptime(args.start_dt_str, "%Y%m%d.%H")
    end_dt = dt.datetime.strptime(args.end_dt_str, "%Y%m%d.%H")

    if (args.input_temporal_res == 1):
        zarr_fpath = "/Projects/ml_benchmarks/hrrr-precip/hrrr-precip.replay-grid.zarr" 
        output_model_name = "HRRR"
    else:
        zarr_fpath = "/Projects/ml_benchmarks/hrrr-precip/hrrr-precip.6hr.replay-grid.zarr"
        output_model_name = "HRRRfrom06hr"
    print(f"Reading zarr archive {zarr_fpath}")
    hrrr_ds = xr.open_zarr(zarr_fpath)
    hrrr_ds = hrrr_ds.squeeze().swap_dims({"t0": "valid_time"}).rename({"valid_time": "period_end_time", "latitude": "lat", "longitude": "lon"})
    hrrr_precip = hrrr_ds.accum_tp[1:,:,:] # First (period-ending) valid time at 00z, so valid 23z-00z (18z-00z) previous day for hourly (6-hourly) data. Don't include in 24-hour precip amounts.
    hrrr_precip = hrrr_precip.loc[start_dt:end_dt] # Subset to desired date/time range

    if (args.output_temporal_res == 24):
        print (f"Calculating {args.output_temporal_res}-hourly precip")
        hrrr_output_precip = pdp.calculate_24hr_accum_precip(hrrr_precip, args.output_temporal_res, args.input_temporal_res)

    # Delete unneeded attributes
    for attr in list(hrrr_output_precip.attrs.keys()):
        if (attr != "long_name" and attr != "short_name" and attr != "units" and attr != "interval_hours"):
            del hrrr_output_precip.attrs[attr]

    # Delete unused coordinates
    hrrr_output_precip = hrrr_output_precip.drop_vars("fhr")
    hrrr_output_precip = hrrr_output_precip.drop_vars("t0")
    hrrr_output_precip = hrrr_output_precip.drop_vars("lead_time")

    # Prepare for output to netCDF
    output_var_name = f"precipitation_{args.output_temporal_res:02d}_hour"
    fname_prefix = f"{output_model_name}.ReplayGrid.{args.output_temporal_res:02d}_hour_precipitation"
    dir_name = os.path.join(utils.data_nc_dir, fname_prefix)
    timestamp_format = "%Y%m%d"
    hrrr_output_precip.name = output_var_name

    # Remove exisiting attributes from period_end_time coordinate
    # Otherwise, these cause an error upon writing to netCDF.
    for attr in list(hrrr_output_precip.period_end_time.attrs):
        del hrrr_output_precip.period_end_time.attrs[attr]

    # Write to netCDF
    print("Writing to netCDF")
    pdp.write_data_array_to_netcdf(hrrr_output_precip,
                                   output_var_name,
                                   dir_name, fname_prefix,
                                   timestamp_format,
                                   temporal_res = args.output_temporal_res,
                                   file_cadence = "day",
                                   testing = args.testing)

    return hrrr_precip, hrrr_output_precip

if __name__ == "__main__":
    hrrr_precip, hrrr_output_precip = main()
