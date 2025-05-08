#!/usr/bin/env python

import numpy as np
import precip_plotting_utilities as pputils
import xarray as xr

precip_bilin = xr.open_dataset("/data/bbasarab/netcdf/ARIs.ReplayGrid/bilinear.ARI.ReplayGrid.02_year.24_hour_precipitation.nc").precip
precip_nearest = xr.open_dataset("/data/bbasarab/netcdf/ARIs.ReplayGrid/nearest.ARI.ReplayGrid.02_year.24_hour_precipitation.nc").precip
precip_native = xr.open_dataset("/Projects/WPC_ExtremeQPF/newARIs/allusa_ari_2yr_24hr_xarray_st4grid.nc").precip

da_dict = {"Native": precip_native, "Bilinear": precip_bilin, "Nearest": precip_nearest}

for data_name, da in da_dict.items():
    da_dict[data_name] = pputils.format_data_array_for_plotting(da) 

plot_levels = np.arange(0, 210, 10)
#pputils.plot_cmap_multi_panel(da_dict, "Native", "CONUS", plot_levels, use_contourf = True)
#pputils.plot_cmap_single_panel(da_dict["Bilinear"], "Bilinear", "CONUS", plot_levels, use_contourf = False)    
#pputils.plot_cmap_single_panel(da_dict["Nearest"], "Nearest", "CONUS", plot_levels, use_contourf = False)    
pputils.plot_cmap_single_panel(da_dict["Native"], "Native", "CONUS", plot_levels, use_contourf = False)    
