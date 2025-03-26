#!/usr/bin/env python

import numpy as np
import utilities as utils
import xarray as xr

da = xr.DataArray(np.arange(5), coords = [[-50., -25., 0., 25., 50.]], dims = "lon")
lons0_to_360 = utils.longitude_to_0to360(da.lon.values)
da["lon"] = lons0_to_360
da = da.sortby("lon")
