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
import sys
import utilities as utils
import xarray as xr

def main():
    utils.suppress_warnings()
    parser = argparse.ArgumentParser()
    parser.add_argument("start_dt_str",
                        help = "Start datetime; format YYYYmddd")
    parser.add_argument("end_dt_str",
                        help = "End datetime; format YYYYmmdd")
    parser.add_argument("--fhour", dest = "fhour", default = 1, type = int,
                        help = "Forecast hour; default 1")
    parser.add_argument("--lookback_hours", dest = "lookback_hours", type = int, default = 6,
                        help = "Number of previous hourly HRRR forecasts to look at if current forecast is missing; default 6")
    parser.add_argument("--read_from_nc", dest = "read_from_nc", action = "store_true",
                        help = "Read from existing netCDF files (for fast plotting), rather than downloading from NCEP AWS archive")
    parser.add_argument("--plot_hourly", dest = "plot_hourly", action = "store_true",
                        help = "Set to make plots of hourly precip")
    parser.add_argument("--plot_24hr", dest = "plot_24hr", action = "store_true",
                        help = "Set to make plots of final 24-hour precip")
    args = parser.parse_args()

    # Configure datetimes and date/time strings
    start_dt = dt.datetime.strptime(args.start_dt_str, "%Y%m%d")
    end_dt = dt.datetime.strptime(args.end_dt_str, "%Y%m%d")
    
    current_dt = start_dt
    while (current_dt != end_dt + dt.timedelta(days = 1)):
        print(f"**** Current datetime: {current_dt:%Y%m%d}")

        if (args.read_from_nc): # Read existing 24-hourly precip data in netCDF format
            input_fpath = configure_nc_fpath(current_dt, read_or_write = "read")
            if (input_fpath == ""):
                current_dt = increment_dtime_by_one_day(current_dt) 
                continue
            print(f"Reading {input_fpath}")
            ds = xr.open_dataset(input_fpath)
            da_24hr = ds.precipitation_24_hour
        else: # Download hourly HRRR data and use it to create 24-hourly precip data
            for model_run_hour in range(0, 24): 
                model_run_dt = current_dt + dt.timedelta(hours = model_run_hour)
                valid_dt = model_run_dt + dt.timedelta(hours = args.fhour)
                print(f"Model run datetime: {model_run_dt:%Y%m%d.%H}")
                print(f"Valid  datetime: {valid_dt:%Y%m%d.%H}")

                # Access the data using the Herbie class
                H = read_hrrr_sfc_data_with_herbie(model_run_dt, args.fhour)

                # Produce xarray dataset of specific variables, in this case accumulated precipitation
                da, model_run_dt, fhour = get_hrrr_accum_precip_hour_ending(H, args.fhour, lookback_hours = args.lookback_hours)

                # If couldn't find data for a specific valid datetime, including looking back at earlier model
                # runs, use persistence, i.e., use the previous valid datetime's data at  the current valid datetime.
                # FIXME: This won't work if we reached the persistence case on the very first HRRR model run read in,
                # in which case there won't be any previous DataArray to persist! 
                if (type(da) is int) and (da == -1):
                    print(f"Persisting precip data from previous valid time {previous_valid_dt:%Y%m%d.%H}")
                    da = previous_da 

                # Set up the daily (24-hour) precip DataArray.
                # If it's the first model run of the day (model_run_hour = 00z),
                # initialize this array. For all other hours, add to it to arrive at
                # 24-hour accumulated precip.
                if (model_run_hour == 0):
                    da_24hr = da
                else:
                    da_24hr += da
                previous_da = da.copy()
                previous_valid_dt = valid_dt

                # Plot hourly precipitation data
                if (args.plot_hourly):
                    pdp.add_attributes_to_data_array(da, short_name = "1-hour precipitation", units = "mm") 
                    plot_hrrr_hourly_precip(da, model_run_dt, fhour, plot_name = "HRRR.NativeGrid", use_contourf = True)
                print("")
           
            # If couldn't find data for this day, continue in while loop 
            if (type(da) is int) and (da == -1):
                current_dt = increment_dtime_by_one_day(current_dt) 
                continue

            # Write to netcdf
            da_24hr = prepare_da_for_output(da_24hr, current_dt)
            write_hrrr_24hr_precip_to_netcdf(da_24hr, current_dt)

        # Plot 24-hour precipitation data
        if args.plot_24hr:
            plot_hrrr_24hr_precip(da_24hr, plot_name = "HRRR.NativeGrid", use_contourf = True)

        current_dt = increment_dtime_by_one_day(current_dt) 

    return da_24hr 

def increment_dtime_by_one_day(dtime):
    return dtime + dt.timedelta(days = 1)

def read_hrrr_sfc_data_with_herbie(model_run_dt, fhour):
    # Access the data using the Herbie class
    return herbie.Herbie(f"{model_run_dt:%Y-%m-%d %H:%M}", # Model run date/time
                         model = "hrrr", 
                         product = "sfc", # Model-dependent product name
                         fxx = fhour, # Forecast lead time
                        )

# Return as xarray dataset
# Clunky; see if can get the regex commands from the documentation working
def get_hrrr_accum_precip_hour_ending(herbie_object, fhour, lookback_hours = 6):
    valid_dt = herbie_object.date + dt.timedelta(hours = fhour)
    current_lookback_hour = 0

    while (current_lookback_hour <= lookback_hours):
        if (fhour >= 1):
            search_str = f":APCP:surface:{fhour - 1}-{fhour}"
        else:
            search_str = f":APCP:surface:{fhour}-{fhour}"

        try:
            ds = herbie_object.xarray(search_str)
            return ds.tp, herbie_object.date, fhour
        except: # Try the previous model run datetime, so fhour must increase by one for the same valid time
        #except (ValueError, FileNotFoundError): # Ideally should except on specific errors, but tired of Herbie being a shithead and finding new ways to fail. 
            model_run_dt = herbie_object.date - dt.timedelta(hours = 1)
            fhour += 1
            print(f"Warning: No data found for {herbie_object.date:%Y%m%d.%H}; trying previous model run time {model_run_dt:%Y%m%d.%H}, forecast hour {fhour}")
            herbie_object = read_hrrr_sfc_data_with_herbie(model_run_dt, fhour)
            current_lookback_hour += 1

    print(f"Error: Could not find any HRRR data for {herbie_object.date:%Y%m%d.%H}, forecast hour {fhour:03d}; can't proceed with precip accumulation calculation for this date")
    return -1, model_run_dt, fhour 
    
def generate_run_dt_fhr_str(run_datetime, fhour):
    return f"{run_datetime:%Y%m%d.%H}.f{fhour:03d}" 

def configure_nc_fpath(dtime, read_or_write = "read"):
    # HRRR netCDF directory name and file name prefix (for reading/writing from/to netCDF)
    nc_prefix = "HRRR.NativeGrid.24_hour_precipitation"
    nc_dir = os.path.join(utils.data_nc_dir, nc_prefix)
    fname = f"{nc_prefix}.{dtime:%Y%m%d}.nc"
    fpath = os.path.join(nc_dir, fname)

    if (read_or_write == "write"):
        if not(os.path.exists(nc_dir)):
            os.mkdir(nc_dir)
    else:
        if not(os.path.exists(nc_dir)):
            print(f"Error: Input directory {nc_dir} does not exist")
            sys.exit(1)
        if not(os.path.exists(fpath)):
            print(f"Error: Input file {fpath} does not exist")
            return "" 

    return fpath

def write_hrrr_24hr_precip_to_netcdf(da, dtime):
    fpath = configure_nc_fpath(dtime, read_or_write = "write")
    print(f"Writing data to {fpath}")
    da.to_netcdf(fpath)

# TODO: Ensure period_end_time is a dimension (as well as a variable)
# and expand the dimensions of the tp array accordingly. Also change name
# from tp to precipitation_24_hour
def prepare_da_for_output(da, dtime):
    # Delete as-of-right-now unnecessary coordinates; may change this in the future. 
    del da.coords["gribfile_projection"]
    del da.coords["step"]
    del da.coords["surface"]
    del da.coords["time"]

    # Add period_end_time coordinate and dimension
    # Rename latitude and longitude coordinates to lat and lon
    period_end_time = dtime + dt.timedelta(days = 1)
    da = da.rename({"valid_time": utils.period_end_time_dim_str,
                    "longitude": "lon",
                    "latitude": "lat"})
    da[utils.period_end_time_dim_str] = period_end_time

    # Add time dimension of length one for readability and standardization with other datasets.
    # This period_end_time dimension explicitly gives the END TIME of the 24-hour accumulation period.
    da = da.expand_dims(dim = {utils.period_end_time_dim_str: [period_end_time]}, axis = 0)

    # Change name of DataArray to be standarized with other datasets.
    da.name = "precipitation_24_hour"

    # Add additional attributes to DataArray for further standardization
    # Change the DataArray units to mm (equivalent to the original units kg/m^2)
    pdp.add_attributes_to_data_array(da, 
                                     short_name = "24-hour precipitation",
                                     long_name = "Precipitation accumulated over the prior 24 hour(s)",
                                     units = "mm") 

    # Ensure period_end_time will be encoded in the output netCDF in 
    # Unix time (i.e., seconds since 1970-01-01 00:00:00)
    da.period_end_time.encoding["units"] = utils.seconds_since_unix_epoch_str 

    return da

def plot_hrrr_hourly_precip(data_to_plot, run_dt, fhour, plot_name = "native", proj = "PlateCarree", region = "CONUS",
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
    levels = pputils.variable_plot_limits("accum_precip", temporal_res = 1) 
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
    p.colorbar.set_label(f"{data_to_plot.short_name} [{data_to_plot.units}]", size = 15)
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

def plot_hrrr_24hr_precip(data_to_plot, plot_name = "native", proj = "PlateCarree", region = "CONUS",
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
    levels = pputils.variable_plot_limits("accum_precip", temporal_res = 24)
    if use_contourf:
        p = axis.contourf(data_to_plot["lon"], data_to_plot["lat"], data_to_plot[0,:,:], transform = data_proj,
                          extend = "both", cmap = "viridis", levels = levels)
        plt.colorbar(p, orientation = "vertical", shrink = 0.7)
        p.colorbar.ax.set_yticks(levels)
    else:
        # FIXME: Only contourf works for HRRR data, not the xarray plot wrapper, which just plots all zeros. WTF?
        p = data_to_plot.plot(ax = axis, levels = levels, extend = "both", cmap = "viridis",
                              cbar_kwargs = {"orientation": "vertical", "shrink": 0.7, "ticks": levels})
    
    # Configure title and colorbar 
    valid_dt = pd.Timestamp(data_to_plot.period_end_time.values[0])
    plt.title(f"HRRR {data_to_plot.short_name} ending at {valid_dt:%Y%m%d.%H}; {proj} projection", size = 15)
    p.colorbar.set_label(f"{data_to_plot.short_name} [{data_to_plot.units}]", size = 15)
    if sparse_cbar_ticks:
        cbar_tick_labels = pputils.create_sparse_cbar_ticks(levels)
    else:
        cbar_tick_labels = levels
    p.colorbar.ax.set_yticklabels(cbar_tick_labels)
    p.colorbar.ax.tick_params(labelsize = 15)

    # Save figure
    plt.tight_layout()
    formatted_short_name = pdp.format_short_name(data_to_plot)
    fig_name = f"cmap.{plot_name}.{formatted_short_name}.{proj}.{valid_dt:%Y%m%d.%H}.{region}.png"
    fig_fpath = os.path.join("/home/bbasarab/plots", fig_name)
    print(f"Saving {fig_fpath}")
    plt.savefig(fig_fpath)

if __name__ == "__main__":
    da = main()


