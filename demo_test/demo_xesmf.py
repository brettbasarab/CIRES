#!/usr/bin/env python

import cf_xarray as cfxr
import matplotlib.pyplot as plt
import numpy as np
import precip_plotting_utilities as pputils
import os
import utilities as utils
import xarray as xr
import xesmf

utils.suppress_warnings()

def add_bounds(ds):
    ds = ds.cf.add_bounds(["lat", "lon"])
    for key in ["lat", "lon"]:
      corners = cfxr.bounds_to_vertices(bounds = ds[f"{key}_bounds"], bounds_dim = "bounds", order = None)
      ds = ds.assign_coords({f"{key}_b": corners})
      ds = ds.drop_vars(f"{key}_bounds")
      
    # ds = ds.rename({"x_vertices": "x_b", "y_vertices": "y_b"})

    return ds

# Read template datasets
print("Reading input template dataset")
ds = xr.open_dataset("/data/bbasarab/netcdf/CONUS404.NativeGrid.24_hour_precipitation/CONUS404.NativeGrid.24_hour_precipitation.20210601.nc")
input_ds_template = add_bounds(ds)

print("Reading output template dataset")
output_ds_template = xr.open_dataset("/data/bbasarab/netcdf/TemplateGrids/ReplayGrid.nc")


# Build regridders and do regridding
print("Building conservative regridder")
regridder_con = xesmf.Regridder(ds_in = input_ds_template,
                                ds_out = output_ds_template,
                                method = "conservative")
output_ds_con = regridder_con(input_ds_template, keep_attrs = True)

print("Building bilinear regridder")
regridder_bil = xesmf.Regridder(ds_in = input_ds_template,
                                ds_out = output_ds_template,
                                method = "bilinear")
output_ds_bil = regridder_bil(input_ds_template, keep_attrs = True)

# Plotting
print("Plotting")

# Native grid
pputils.plot_cmap_single_panel(input_ds_template.precipitation_24_hour,
                               "test_xesmf.CONUS404.NativeGrid",
                               "test_xesmf.CONUS404.NativeGrid",
                               "CONUS",
                               plot_levels = np.arange(0, 42, 2))

# Conservative regridding
pputils.plot_cmap_single_panel(output_ds_con.precipitation_24_hour,
                               "test_xesmf_con.CONUS404.ReplayGrid",
                               "test_xesmf_con.CONUS404.ReplayGrid",
                               "CONUS",
                               plot_levels = np.arange(0, 42, 2))

# Bilinear regridding
pputils.plot_cmap_single_panel(output_ds_bil.precipitation_24_hour,
                               "test_xesmf_bil.CONUS404.ReplayGrid",
                               "test_xesmf_bil.CONUS404.ReplayGrid",
                               "CONUS",
                               plot_levels = np.arange(0, 42, 2))






