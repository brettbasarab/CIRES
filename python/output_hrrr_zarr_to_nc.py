#!/usr/bin/env python

import argparse
import datetime as dt
import numpy as np
import os
import precip_data_processors as pdp
import utilities as utils
import xarray as xr
 
def main():
    program_description = ("Program to convert Tim Smith's HRRR precip 'reanalysis' for zarr to netCDF. "
                            "Currently configured to output 24-hour precip only to netCDF.")
    parser = argparse.ArgumentParser(description = program_description) 
    parser.add_argument("start_dt_str",
                        help = "Start date/time to output; format YYYYmmdd.HH, PERIOD ENDING")
    parser.add_argument("end_dt_str",
                        help = "End date/time to output; format YYYYmmdd.HH, PERIOD ENDING")
    parser.add_argument("-t", "--testing", dest = "testing", action = "store_true", default = False,
                        help = "Testing mode: Configure netCDF output file paths but don't write")
    args = parser.parse_args()
    utils.suppress_warnings()

    start_dt = dt.datetime.strptime(args.start_dt_str, "%Y%m%d.%H")
    end_dt = dt.datetime.strptime(args.end_dt_str, "%Y%m%d.%H")

    print("Reading from zarr")
    hrrr_ds = xr.open_zarr("/Projects/ml_benchmarks/hrrr-precip/hrrr-precip.replay-grid.zarr")
    hrrr_ds = hrrr_ds.squeeze().swap_dims({"t0": "valid_time"}).rename({"valid_time": "period_end_time", "latitude": "lat", "longitude": "lon"})
    hrrr_precip = hrrr_ds.accum_tp[1:,:,:] # First hour from this archive at 00z, so valid 23z-00z previous day. Don't include in 24-hour precip amounts.
    hrrr_precip = hrrr_precip.loc[start_dt:end_dt] # Subset to desired date/time range

    print ("Calculating 24-hour precip")
    hrrr_precip24 = pdp.calculate_24hr_accum_precip(hrrr_precip)

    # Delete unneeded attributes
    for attr in list(hrrr_precip24.attrs.keys()):
        if (attr != "long_name" and attr != "short_name" and attr != "units" and attr != "interval_hours"):
            del hrrr_precip24.attrs[attr]

    # Delete unused coordinates
    hrrr_precip24 = hrrr_precip24.drop_vars("fhr")
    hrrr_precip24 = hrrr_precip24.drop_vars("t0")
    hrrr_precip24 = hrrr_precip24.drop_vars("lead_time")

    # Prepare for output to netCDF
    output_var_name = "precipitation_24_hour"
    fname_prefix = "HRRR.ReplayGrid.24_hour_precipitation"
    dir_name = os.path.join(utils.data_nc_dir, fname_prefix)
    timestamp_format = "%Y%m%d"
    hrrr_precip24.name = output_var_name

    # Remove exisiting attributes from period_end_time coordinate
    # Otherwise, these cause an error upon writing to netCDF.
    for attr in list(hrrr_precip24.period_end_time.attrs):
        del hrrr_precip24.period_end_time.attrs[attr]

    # Write to netCDF
    print("Writing to netCDF")
    pdp.write_data_array_to_netcdf(hrrr_precip24,
                                   output_var_name,
                                   dir_name, fname_prefix,
                                   timestamp_format,
                                   temporal_res = 24,
                                   file_cadence = "day",
                                   testing = args.testing)

    return hrrr_precip, hrrr_precip24

if __name__ == "__main__":
    hrrr_precip, hrrr_precip24 = main()
