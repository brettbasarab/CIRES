#!/usr/bin/env python

import numpy as np
import precip_plotting_utilities as pputils
import xarray as xr

da = xr.open_dataset("/data/bbasarab/netcdf/CONUS404.NativeGrid.24_hour_precipitation/CONUS404.NativeGrid.24_hour_precipitation.20210601.nc").precipitation_24_hour
pputils.plot_cmap_single_panel(da, "CONUS404", "CONUS", np.arange(0, 85, 5))
