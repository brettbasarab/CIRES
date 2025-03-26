#!/usr/bin/env python

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import regionmask
import xarray as xr

# Read input Replay data (converted a small bit of Replay data to netCDF,
# available on the Linux servers)
input_fpath = "/data/bbasarab/netcdf/testing/replay_native.nc"
model_precip = xr.open_dataset(input_fpath).accum_precip

# Extract the data valid over the globe that we want to plot; take the sum over the time period
data_to_plot = model_precip.sum(dim = "period_end_time")

# Create the CONUS mask, excluding AK and HI, to only include data over the lower 48 states (i.e., the CONUS)
# Each state+DC is represented by an integer between 0 and 50 (clunky, but that's the way it is!)
# So there are 51 "states" total (regionmask refers to each states as regions)
states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
mask = states.mask(data_to_plot) # This step creates a DatArray with coordinates the same as data_to_plot
AK_index = states.map_keys("Alaska")
HI_index = states.map_keys("Hawaii")
mask_CONUS = (mask != AK_index) & (mask != HI_index) & (mask >= 0) # Define the portion of the mask above that is in any of the states except AK and HI
data_to_plot = data_to_plot.where(mask_CONUS) # Use the xarray.DataArray.where method to drop any values that don't fall into the mask_CONUS region

# Plot data on a map
plt.figure(figsize = (15, 10))
axis = plt.axes(projection = ccrs.PlateCarree())
axis.coastlines()
axis.set_extent([-125, -66, 24, 51], crs=ccrs.PlateCarree())
axis.add_feature(cfeature.BORDERS)
axis.add_feature(cfeature.STATES)
levels = np.arange(0, 680, 20)
plot_handle = data_to_plot.plot(ax = axis, levels = levels, extend = "both", cbar_kwargs = {"orientation": "vertical", "shrink": 0.7, "ticks": levels})
plt.tight_layout()
plt.savefig("./test_regionmask.png")
