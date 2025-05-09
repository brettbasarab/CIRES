#!/usr/bin/env python

import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import os
import precip_plotting_utilities as pputils
import utilities as utils
import xarray as xr

@dataclasses.dataclass
class CDF:
    quantiles: float
    values: float

def main():
    precip_native = xr.open_dataset("/Projects/WPC_ExtremeQPF/newARIs/allusa_ari_2yr_24hr_xarray_st4grid.nc").precip
    precip_bilin = xr.open_dataset("/data/bbasarab/netcdf/ARIs.ReplayGrid/bilinear.ARI.ReplayGrid.02_year.24_hour_precipitation.nc").precip
    precip_nearest = xr.open_dataset("/data/bbasarab/netcdf/ARIs.ReplayGrid/nearest.ARI.ReplayGrid.02_year.24_hour_precipitation.nc").precip

    da_dict = {"Native": precip_native, "Bilinear": precip_bilin, "Nearest": precip_nearest}

    for data_name, da in da_dict.items():
        da_dict[data_name] = pputils.format_data_array_for_plotting(da) 

    plot_levels = np.arange(0, 210, 10)
    #pputils.plot_cmap_multi_panel(da_dict, "Native", "CONUS", plot_levels, use_contourf = True)
    #pputils.plot_cmap_single_panel(da_dict["Bilinear"], "Bilinear", "CONUS", plot_levels, use_contourf = False)    
    #pputils.plot_cmap_single_panel(da_dict["Nearest"], "Nearest", "CONUS", plot_levels, use_contourf = False)    
    #pputils.plot_cmap_single_panel(da_dict["Native"], "Native", "CONUS", plot_levels, use_contourf = False)

    cdf_native = create_cdf(precip_native)
    cdf_bilin = create_cdf(precip_bilin)
    cdf_nearest = create_cdf(precip_nearest)

    cdf_dict = {"Native": cdf_native, "Bilinear": cdf_bilin, "Nearest": cdf_nearest}

    # Plot CDFs
    plt.figure(figsize = (10, 10))
    plt.grid(True, linewidth = 0.5)
    plt.xlabel("Precip amount (mm)", size = 15)
    plt.ylabel("Probability", size = 15)
    plt.title("CDFs of native and upscaled ARI grids", size = 15)
    for cdf_name, cdf in cdf_dict.items():
        plt.plot(cdf_dict[cdf_name].values, cdf_dict[cdf_name].quantiles, linewidth = 2, label = cdf_name)
    plt.legend(loc = "best", prop = {"size": 15})
    fig_fpath = os.path.join(utils.plot_output_dir, "test_ari_grids_cdf.png")
    print(f"Saving {fig_fpath}") 
    plt.savefig(fig_fpath)

    return precip_native, precip_bilin, precip_nearest

def create_cdf(data_array):
    quantiles = np.concatenate(( np.arange(0.0, 1.0, 0.01), np.arange(0.991, 1.0, 0.001) ))
    values = np.array([data_array.quantile(i).item() for i in quantiles])

    return CDF(quantiles = quantiles, values = values) 

if __name__ == "__main__":
    precip_native, precip_bilin, precip_nearest = main() 
