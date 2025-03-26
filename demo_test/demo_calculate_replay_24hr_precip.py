#!/usr/bin/env python

import xarray as xr

# Read in the entire Replay dataset (from 1994-2023) from GCP
atmos_dataset = xr.open_zarr("gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr", storage_options={"token": "anon"})

# The above creates an xarray Dataset, which contains a bunch of xarray DataArrays.
# Select any of them using atmos_dataset["var_name"] or atmos_dataset.var_name
# prate = average precipitation rate over the prior three hours
prate = atmos_dataset.prateb_ave

# We can work with a DataArray very similarly to a numpy array, but it has
# nice extra features like dimensions with coordinates, attributes, etc.
dims = prate.dims # List of the dimension names
lats = prate["grid_yt"].values # Numpy array of the latitude coordinates

# Multiply by 3 hours (in seconds) to get accumulated precipitation in mm 
precip_3hourly = prate * 3600 * 3

# Remove a few hours of data at the beginning and end of 3-hourly data array so that 
# it nicely starts at a 03z time and ends at a 00z time. This way we only have complete (UTC)
# days for which to calculate 24-hour precip. Start at 03z because precip data is valid over the 
# prior 3 hours. 
precip_3hourly = precip_3hourly.loc["1994-01-01 03:00:00":"2023-10-11 00:00:00"]

# Set a time step to use in the xarray rolling method below
# Set it to 8 since there are eight 3-hour periods in one 24-hour period.
# Take the sum over the output of the rolling method, indexed every 24 hours.
# The result, precip_24hourly, contains precipitation accumulated over the prior 24 hours (UTC).
time_step = 8 
roller = precip_3hourly.rolling({"time": time_step})
precip_24hourly = roller.sum()[(time_step - 1)::time_step,:,:]

# You might want to change the time dimension name to 'period_end_time' to
# match what I have in the other datasets (ERA5, etc.) This is done below (commented for now):
#precip_24hourly = precip_24hourly.rename({"time": "period_end_time"}) 
