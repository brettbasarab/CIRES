#!/usr/bin/env python

# Basic program to read data in zarr format.
# Briefly, zarr is a format optimized for storing multi-dimensional geospatial data in the cloud.
# See more here: wiki.earthdata.nasa.gov/display/ESO/Zarr+Format

import xarray as xr

# Coarse/classic (0.25 degree resolution) UFS Replay on Google Cloud in zarr format
replay_coarse_gcs_path = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr"

# Read in the entire Replay dataset (from 1994-2023)
atmos_dataset = xr.open_zarr(replay_coarse_gcs_path, storage_options={"token": "anon"})

# The above creates an xarray Dataset, which contains a bunch of xarray DataArrays.
# Select any of them using atmos_dataset["var_name"] or atmos_dataset.var_name
# Average precipitation rate over the prior three hours
prate = atmos_dataset.prateb_ave

# We can work with a DataArray very similarly to a numpy array, but it has
# nice extra features like dimensions with coordinates, attributes, etc.
dims = prate.dims # List of the dimension names
lats = prate["grid_yt"].values # Numpy array of the latitude coordinates

# Multiply by 3 hours (in seconds) to get accumulated precipitation in mm 
accum_precip = prate * 3600 * 3

# Select slices of data you want using this (overly?) intuitive .loc syntax.
# Nice not to have to find the numerical indices, as you would for numpy arrays.
# See docs.xarray.dev/en/stable/user-guide/indexing.html for more on xarray indexing.
# Here we get data over the globe for September 2013.
sep2013_data = accum_precip.loc["2013-09-01 03:00:00":"2013-10-01 00:00:00"]

# Get September 2013 data at a point near Boulder (i.e., a time series) by using the .sel syntax 
boulder_sep2013_data = sep2013_data.sel(grid_yt = 40, grid_xt = 255, method = "nearest")

# Finally, if you want to get the data as a numpy array:
boulder_data_array = boulder_sep2013_data.values
