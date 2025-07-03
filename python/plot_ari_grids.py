#!/usr/bin/env python

import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import os
import precip_plotting_utilities as pputils
import utilities as utils
import xarray as xr

def main():
    # Read in ARI data
    print("Reading data")
    precip_native = xr.open_dataset("/Projects/WPC_ExtremeQPF/newARIs/allusa_ari_2yr_24hr_xarray_st4grid.nc").precip
    precip_bilin = xr.open_dataset("/data/bbasarab/netcdf/ARIs.ReplayGrid/bilinear.ARI.ReplayGrid.02_year.24_hour_precipitation.nc").precip
    precip_nearest = xr.open_dataset("/data/bbasarab/netcdf/ARIs.ReplayGrid/nearest.ARI.ReplayGrid.02_year.24_hour_precipitation.nc").precip

    da_dict = {"Native": precip_native, "Bilinear": precip_bilin, "Nearest": precip_nearest}

    # Plot contour maps for visual comparison
    print("Plotting contour maps for visual comparison")
    pputils.plot_cmap_multi_panel(da_dict, "Native", "CONUS",
                                  plot_levels = np.arange(0, 210, 10),
                                  short_name = "02yearARIs")

    # Plot CDFs
    print("Plotting CDFs")
    pputils.plot_precip_cdf(da_dict,
                            "02-year, 24-hour ARI interp grids",
                            "02year_24hour_ARI_interp_grids",
                            xticks = np.arange(0, 260, 10),
                            xlims = [0, 250],
                            xticks_rotation = 45,
                            #skip_nearest = True,
                            valid_dtime = None)

    return da_dict 

if __name__ == "__main__":
    da_dict = main() 
