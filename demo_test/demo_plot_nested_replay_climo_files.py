#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import xarray as xr

nc_dir = "/home/bbasarab/nested_replay_static_files/netcdf"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nc_name",
                        help = "netCDF file name (excluding full path, just the name")
    args = parser.parse_args()

    data_array = read_data_array(args.nc_name)
    plot_data(data_array, args.nc_name.split(".nc")[0]) 

def read_data_array(fname):
    fpath = os.path.join(nc_dir, fname)
    if not(os.path.exists(fpath)):
        print(f"Error: Input file path {fname} does not exist")
        sys.exit(1)

    print(f"Reading {fpath}")
    ds = xr.open_dataset(fpath)

    if ("sst" in fname.lower()):
        return ds.t
    else:
        return ds.icec    

def plot_data(data_array, data_name):
    times = [pd.Timestamp(i) for i in data_array.time.values]
    for time in times:
        plt.figure(figsize = (10,8))
        data_to_plot = data_array.sel(time = time.strftime("%Y-%m-%d %H:00:00"))
        data_to_plot.plot()
        fname = f"{data_name}.{time:%Y%m%d.%H}.png"
        fpath = os.path.join(nc_dir, fname)
        print(f"Saving {fpath}")
        plt.savefig(fpath)

if __name__ == "__main__":
    main()
