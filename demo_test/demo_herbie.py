#!/usr/bin/env python

import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import herbie
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import precip_data_processors as pdp
import precip_plotting_utilities as pputils
import xarray as xr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dt_str",
                        help = "Model run datetime; format YYYYmmdd.HH")
    parser.add_argument("--model_name", dest = "model_name", default = "hrrr",
                        help = "Model name; default 'hrrr'")
    parser.add_argument("--fhour", dest = "fhour", default = 1, type = int,
                        help = "Forecast hour; default 1")
    args = parser.parse_args()

    # Configure datetimes and date/time strings
    run_datetime = dt.datetime.strptime(args.run_dt_str, "%Y%m%d.%H")

    # Access the data using the Herbie class
    H = herbie.Herbie(f"{run_datetime:%Y-%m-%d %H:%M}", # Model run date/time
                      model = args.model_name, # Model name
                      product = "sfc", # Model-dependent product name
                      fxx = args.fhour, # Forecast lead time
                     )

    # Inventory the data; creates a pandas dataframe
    # NOTE: Not currently used in code below; keeping for now as a demo
    inventory = H.inventory()

    # Produce xarray dataset of specific variables, in this case accumulated precipitation
    da = get_hrrr_accum_precip_prior_hour(H, args.fhour)

    # Add attributes to latitude/longitude variables which may be necessary for cdo interpolation
    # Since it appears this renaming isn't necessary, this function currently does nothing.
    da = fix_da_for_output(da)

    # Write to netcdf
    fname = f"test_hrrr.{generate_run_dt_fhr_str(run_datetime, args.fhour)}.nc"
    fpath = os.path.join("/data/bbasarab/netcdf/testing", fname)
    print(f"Writing data to {fpath}")
    da.to_netcdf(fpath)

    # Plot data
    print("Plotting data")
    plot_hrrr(da, run_datetime, args.fhour, plot_name = "HRRR.NativeGrid", use_contourf = True)

    return da 

# Return as xarray dataset
# Clunky; see if can get the regex commands from the documentation working
def get_hrrr_accum_precip_prior_hour(herbie_object, fhour):
    if (fhour >= 1):
        search_str = f":APCP:surface:{fhour - 1}-{fhour}"
    else:
        search_str = f":APCP:surface:{fhour}-{fhour}"
    ds = herbie_object.xarray(search_str)
    
    return ds.tp 

def generate_run_dt_fhr_str(run_datetime, fhour):
    return f"{run_datetime:%Y%m%d.%H}.f{fhour:03d}" 

def plot_hrrr(data_to_plot, run_dt, fhour, plot_name = "native", proj = "PlateCarree", region = "CONUS",
              sparse_cbar_ticks = False, use_contourf = False):
    # Plot data on a map
    match proj:
        case "LambertConformal":
            map_proj = ccrs.LambertConformal()
            data_proj = ccrs.PlateCarree()
        case _:
            map_proj = ccrs.PlateCarree()
            data_proj = ccrs.PlateCarree()

    # Configure plot
    plt.figure(figsize = pputils.regions_info_dict[region].figsize_sp)
    axis = plt.axes(projection = map_proj)
    axis.coastlines()
    axis.set_extent(pputils.regions_info_dict[region].region_extent, crs = data_proj)
    gl = axis.gridlines(crs = data_proj, color = "gray", alpha = 0.5, draw_labels = True,
                        linewidth = 0.5, linestyle = "dashed")
    axis.add_feature(cfeature.BORDERS)
    axis.add_feature(cfeature.STATES)

    # Plot data
    levels = np.arange(0, 85, 5)
    if use_contourf:
        p = axis.contourf(data_to_plot["longitude"], data_to_plot["latitude"], data_to_plot, transform = data_proj,
                          extend = "both", cmap = "viridis", levels = levels)
        plt.colorbar(p, orientation = "vertical", shrink = 0.7)
        p.colorbar.ax.set_yticks(levels)
    else:
        # FIXME: Only contourf works for HRRR data, not the xarray plot wrapper, which just plots all zeros. WTF?
        p = data_to_plot.plot(ax = axis, levels = levels, extend = "both", cmap = "viridis",
                              cbar_kwargs = {"orientation": "vertical", "shrink": 0.7, "ticks": levels})
    
    # Configure title and colorbar 
    valid_dt = run_dt + dt.timedelta(hours = fhour)
    plt.title(f"HRRR fcst {run_dt:%Y%m%d.%H} f{fhour:03d} (valid {valid_dt:%Y%m%d.%H}); {proj} projection", size = 15)
    p.colorbar.set_label(f"{data_to_plot.name} [{data_to_plot.units}]", size = 15)
    if sparse_cbar_ticks:
        cbar_tick_labels = pputils.create_sparse_cbar_ticks(levels)
    else:
        cbar_tick_labels = levels
    p.colorbar.ax.set_yticklabels(cbar_tick_labels)
    p.colorbar.ax.tick_params(labelsize = 15)

    # Save figure
    plt.tight_layout()
    fig_name = f"cmap.{plot_name}.{data_to_plot.name}.{proj}.{generate_run_dt_fhr_str(run_dt, fhour)}.{region}.png"
    fig_fpath = os.path.join("/home/bbasarab/plots", fig_name)
    print(f"Saving {fig_fpath}")
    plt.savefig(fig_fpath)

# Var/dim renaming as necessary
# Plus deleting of some attributes
def fix_da_for_output(da):
    #da = da.rename({"x": "west_east", "y": "south_north", "longitude": "lon", "latitude": "lat"})
    #da = da.rename({"longitude": "lon", "latitude": "lat"}) 

    #da.longitude.attrs["units"] = "degree_east"
    #da.latitude.attrs["units"] = "degree_north"
    
    #da.longitude.attrs["coordinates"] = "XLONG XLAT"
    #da.latitude.attrs["coordinates"] = "XLONG XLAT"

    #del da.coords["time"]
    del da.coords["step"]
    del da.coords["surface"]
    #del da.coords["valid_time"]
    del da.coords["gribfile_projection"]

    return da

if __name__ == "__main__":
    da = main()


