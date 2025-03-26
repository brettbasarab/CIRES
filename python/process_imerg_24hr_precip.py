#!/usr/bin/env python

import argparse
import datetime as dt
import numpy as np
import os
import sys
import utilities as utils
import xarray as xr

def main():
    parser = argparse.ArgumentParser(description = "Calculate 24-hour accumulated IMERG precip from daily files containing 3-hourly precip and output to netCDF")
    parser.add_argument("start_dt_str",
                        help = "First month to process, format YYYYmmdd")   
    parser.add_argument("end_dt_str",
                        help = "Last month to process, format YYYYmmdd")
    args = parser.parse_args()

    start_dt = dt.datetime.strptime(args.start_dt_str, "%Y%m%d")
    end_dt = dt.datetime.strptime(args.end_dt_str, "%Y%m%d")
    
    input_dir = "/data/bbasarab/netcdf/IMERGonReplayGrid.03_hour_precipitation"
    output_dir = "/data/bbasarab/netcdf/IMERGonReplayGrid.24_hour_precipitation"
    if (not os.path.exists(output_dir)):
        os.mkdir(output_dir)
   
    current_dt = start_dt
    while (current_dt != end_dt + dt.timedelta(days = 1)):
        print(f"***** Current date: {current_dt:%Y%m%d}")
    
        # Configure input file
        input_fname = f"IMERGonReplayGrid.03_hour_precipitation.{current_dt:%Y%m%d}.nc"
        input_fpath = os.path.join(input_dir, input_fname)
        print(f"Input file: {input_fpath}")
        if (not os.path.exists(input_fpath)):
            print(f"Warning: Input file path {input_fpath} does not exist; continuing to next file")
            current_dt += dt.timedelta(days = 1)
            continue

        # Read input 3-hourly data
        print("Reading input 3-hourly data")
        precip_3hr = xr.open_dataset(input_fpath).precipitation_3_hour

        # Calculate desired 24-hour accumulated precip data
        print("Calculating 24-hour precip by summing 3-hourly precip")
        precip_24hr = precip_3hr.sum(dim = utils.period_end_time_dim_str) #keepdims = True)

        # Add back in the time dimension of length one for readability (explicitly gives the
        # END TIME of the 24-hour accumulation period
        precip_24hr = precip_24hr.expand_dims(dim = {utils.period_end_time_dim_str: [current_dt + dt.timedelta(days = 1)]},
                                              axis = 0)

        # Add necessary attributes
        precip_24hr.attrs["short_name"] = f"24-hour precipitation"
        precip_24hr.attrs["long_name"] = f"Precipitation accumulated over the prior 24 hour(s)"
        precip_24hr.attrs["units"] = precip_3hr.units
        output_var_name = "precipitation_24_hour"
        precip_24hr.name = output_var_name

        # Write 24-hour data to netCDF
        output_fname = f"IMERGonReplayGrid.24_hour_precipitation.{current_dt:%Y%m%d}.nc"
        output_fpath = os.path.join(output_dir, output_fname)
        print(f"Writing output file {output_fpath}")
        precip_24hr.to_netcdf(output_fpath, encoding = {output_var_name: {"dtype": "float32"}}) 

        current_dt += dt.timedelta(days = 1)

    return precip_24hr

if __name__ == "__main__":
    precip_24hr = main()
