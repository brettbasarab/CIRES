#!/usr/bin/env python

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import precip_plotting_utilities as pputils
import precip_verification_processor
import sys
import utilities as utils
import xarray as xr

# Pick a threshold (.sel from DataArrays) then aggregate to common months and/or seasons.
# Plot each FSS value on map, with entire region colored by that value

regions = ["US-East", "US-Central", "US-Mountain", "US-WestCoast"]    
data_names = ["AORC", "CONUS404", "ERA5", "IMERG", "Replay"]
truth_data_name = "AORC"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_dt_str",
                        help = "Start date/time of verification; format YYYYmmdd for 24-hourly data; otherwise YYYYmmdd.HH")
    parser.add_argument("end_dt_str",
                        help = "End date/time of verification; format YYYYmmdd for 24-hourly data; otherwise YYYYmmdd.HH")
    parser.add_argument("--temporal_res", default = 24, type = int, choices = [1, 3, 24],
                        help = "Temporal resolution of precip data used in verification (default 24)")
    parser.add_argument("--time_period_type", default = "common_seasonal", choices = ["common_seasonal", "common_monthly"],
                        help = "Time period type for which to calculated aggregated FSS")
    parser.add_argument("--radius", dest = "radius", default = utils.replay_grid_cell_size * 2, type = float,
                        help = f"Fixed radius at which to aggregated FSS data for given threshold (default {utils.replay_grid_cell_size * 2} deg)")
    parser.add_argument("--threshold", dest = "threshold", default = 10.0, type = float,
                        help = "FSS evaluation threshold (mm); default 10mm")
    parser.add_argument("--cmap", dest = "cmap", default = "RdYlGn", 
                        help = "Color map for contour plots; default RdYlGn")
    args = parser.parse_args()

    regions_dict = {}
    for region in regions:
        print(f"********** COLLECTING DATA FOR REGION {region} **********") 
        fss_dict = {}
        for data_name in data_names: 
            if (data_name == truth_data_name):
                continue

            if (data_name == "Replay"):
                grid_name = "NativeGrid"
            else:
                grid_name = "ReplayGrid"
        
            nc_dir = f"{data_name}.{grid_name}.24_hour_precipitation.stats"
            nc_file = f"{data_name}.{grid_name}.24_hour_precipitation.full_period.fss.by_threshold.radius{args.radius:0.1f}deg.200201-202112.{region}.nc"
            nc_fpath = os.path.join(utils.data_nc_dir, nc_dir, nc_file)
            if (not os.path.exists(nc_fpath)):
                print(f"Error: {nc_fpath} does not exist")
                sys.exit(1)            

            # Gives a DataArray with dims 7302 (length of full 2002-2021 data)
            print(f"Reading {nc_fpath}")
            fss = xr.open_dataset(nc_fpath).fss
            print(f"FSS array shape: {fss.shape}")
            fss_dict[data_name] = fss
           
        verif = precip_verification_processor.PrecipVerificationProcessor(args.start_dt_str,
                                                                          args.end_dt_str,
                                                                          USE_EXTERNAL_DA_DICT = True, 
                                                                          IS_STANDARD_INPUT_DICT = False,
                                                                          external_da_dict = fss_dict, 
                                                                          data_names = data_names, 
                                                                          truth_data_name = truth_data_name,
                                                                          region = region,
                                                                          temporal_res = 24)
        agg_fss_dict, _, _ = verif.calculate_aggregated_fss(external_fss_dict = fss_dict, time_period_type = args.time_period_type, eval_type = "threshold")
        regions_dict[region] = agg_fss_dict

    template_da = read_template_data()
    conus_mask = pputils.create_conus_mask(template_da)
    mountain_mask = pputils.create_mountain_states_mask(template_da)
    west_coast_mask = pputils.create_west_coast_states_mask(template_da)

    if (args.time_period_type == "common_seasonal"):
        time_period_string_list = pputils.construct_seasonal_string_list()
        dim_str = utils.common_season_dim_str 
    else: 
        time_period_string_list = pputils.construct_monthly_string_list()
        dim_str = utils.common_month_dim_str 

    for time_period in time_period_string_list: 
        print(f"**** Creating FSS grid with all regions for time period {time_period}")
        data_names_dict = extract_values_to_plot(regions_dict, time_period, threshold = args.threshold)
        da_dict = {}
        da_region_ann_dict = {}
        for data_name, fss_by_region_dict in data_names_dict.items():
            # Populate US-WestCoast value; populate everything else with US-Mountain value
            fss_da = xr.where(west_coast_mask, fss_by_region_dict["US-WestCoast"], fss_by_region_dict["US-Mountain"])

            # Populate US-Central values
            lon1, lon2, lat1, lat2 = pputils.regions_info_dict["US-Central"].region_extent
            fss_da.loc[{"lon": slice(lon1 + 360, lon2 + 360), "lat": slice(lat2, lat1)}] = fss_by_region_dict["US-Central"]

            # Populate US-East values 
            lon1, lon2, lat1, lat2 = pputils.regions_info_dict["US-East"].region_extent
            fss_da.loc[{"lon": slice(lon1 + 360, lon2 + 360), "lat": slice(lat2, lat1)}] = fss_by_region_dict["US-East"]

            # Mask to CONUS only (may no longer need this with initial setting to np.nan)
            fss_da = fss_da.where(conus_mask)

            # Expand dimensions to work better with plotting tools
            #fss_da = fss_da.expand_dims(dim = {dim_str: [time_period]}, axis = 0)
            da_dict[data_name] = fss_da

            # Collect annotations to plot
            region_ann_dict = {}
            for region in regions:
                region_ann_dict[region] =  pputils.regions_info_dict[region].central_point + [fss_by_region_dict[region]] 
            da_region_ann_dict[data_name] = region_ann_dict

        print(f"**** Plotting FSS grid with all regions for time period {time_period}")
        plot_cmap_multi_panel_fss_by_region(da_dict, time_period, args.radius, args.threshold, verif.monthly_time_period_str,
                                            time_period_type = args.time_period_type, temporal_res = args.temporal_res, cmap = args.cmap,
                                            region_ann_dict = da_region_ann_dict) 

    return regions_dict, verif, da_region_ann_dict

# This function rearranges regions_dict (created in main() as regions_dict[regions][time_period][data_name])
# to extract the FSS values we want to plot. It returns another nested dictionary data_names_dict[data_name][region]
# for the given time_period.
def extract_values_to_plot(regions_dict, time_period, threshold):
    # All values within data_names_dict[data_name] go on one panel of the plots with
    # each region colored by FSS for the given time period (common_seasonal, etc.). 
    data_names_dict = {}
    for data_name in data_names:
        if (data_name == truth_data_name):
            continue

        data_name_regions_dict = {}
        for region in regions: 
            data = regions_dict[region][time_period][data_name].sel(threshold = threshold).item()
            data_name_regions_dict[region] = data
        data_names_dict[data_name] = data_name_regions_dict 
            
    return data_names_dict

# Create a template on the Replay grid populated with all NaNs 
def read_template_data(grid_name = "Replay"):
    fname = f"{grid_name}Grid.nc"
    fpath = os.path.join(utils.data_nc_dir, "TemplateGrids", fname)
    template_da = xr.open_dataset(fpath)[utils.accum_precip_var_name][0,:,:]
    template_da[:,:] = np.nan 

    return template_da 

def set_plot_levels_based_on_threshold(threshold):
    if (threshold <= 35.0):
        return np.arange(0.2, 0.85, 0.05)
    elif (threshold <= 60.0):
        return np.arange(0.1, 0.65, 0.05)
    else:
        return np.arange(0.0, 0.32, 0.02)

def plot_cmap_multi_panel_fss_by_region(data_dict, time_period, radius, threshold, eval_period_str, time_period_type = "common_seasonal",
                                        master_region = "CONUS", temporal_res = 24, cmap = "RdYlGn", radius_units = "deg", region_ann_dict = None):
    plot_info = pputils.regions_info_dict[master_region] # We're plotting the entire CONUS

    # Configure basic info about the data
    num_da = len(data_dict.items())
    data_names_str = "".join(f"{key}." for key in data_dict.keys())

    # Set map projection to be used for all subplots in the figure 
    proj = ccrs.PlateCarree()

    # Loop through these datetimes, making figures with subplots corresponding to each of the data arrays in data_dict
    if (num_da >= 5):
        figsize = plot_info.figsize_mp_5plus
    else:
        figsize = plot_info.figsize_mp

    fig = plt.figure(figsize = figsize)
    axes_list, cbar_ax = pputils.create_gridded_subplots(num_da, proj) 

    # Loop through each of the subplot axes defined above (one axis for each DataArray) and plot the data 
    for axis, (data_name, da) in zip(axes_list, data_dict.items()):
        axis.coastlines()
        axis.set_extent(plot_info.region_extent, crs = proj)
        axis.add_feature(cfeature.BORDERS)
        axis.add_feature(cfeature.STATES)

        # One colorbar for entire figure; add as its own separate axis defined using subplot2grid 
        #plot_handle = da.plot(ax = axis, levels = plot_levels, transform = proj, extend = "both", cmap = cmap, add_colorbar = False)
        plot_levels = set_plot_levels_based_on_threshold(threshold)
        plot_handle = axis.contourf(da["lon"], da["lat"], da, transform = proj, extend = "both", cmap = cmap, levels = plot_levels)
        cbar = fig.colorbar(plot_handle, cax = cbar_ax, ticks = plot_levels, shrink = 0.5, orientation = "horizontal")
        cbar.set_label("FSS", size = 15)
        cbar_tick_labels_rotation, cbar_tick_labels_fontsize = pputils.set_cbar_labels_rotation_and_fontsize(plot_levels, master_region, num_da, for_single_cbar = True)
        cbar.ax.set_xticklabels([f"{level:0.2f}" for level in plot_levels], rotation = cbar_tick_labels_rotation) 
        cbar.ax.tick_params(labelsize = cbar_tick_labels_fontsize)

        gl = axis.gridlines(crs = proj, color = "gray", alpha = 0.5, draw_labels = False,
                            linewidth = 0.5, linestyle = "dashed")
        axis.set_title(data_name, fontsize = 16) 
        
        # Annotate each region with the exact FSS value
        if (region_ann_dict is not None):
            for val in region_ann_dict[data_name].values():
                axis.text(val[1] - 1.2, val[0], f"{val[-1]:0.2f}", color = "purple", size = 18, fontweight = "bold") 

        # Plot lines to visually separate regions
        central_region_extent = pputils.regions_info_dict["US-Central"].region_extent
        axis.plot([central_region_extent[0], central_region_extent[0]], [29, 49], color = "black", linewidth = 3)
        axis.plot([central_region_extent[1], central_region_extent[1]], [29.55, 47], color = "black", linewidth = 3)

    # Create the plot title. If we're looping through individual Timestamp objects, they represent
    # the period end time of the data. Indicate this explicitly in the title.
    title_string = f"FSS by region, CONUS {temporal_res}-hour precipitation (r = {radius:0.2f} {radius_units}, t = {threshold:0.1f} mm): {eval_period_str} {time_period}"
    fig.suptitle(title_string, fontsize = 16, fontweight = "bold")
    fig.tight_layout()

    # Save figure
    if (time_period in pputils.construct_seasonal_string_list()):
        time_period_number = pputils.season_string_to_season_number(time_period)
    elif (time_period in pputils.construct_monthly_string_list()): 
        time_period_number = pputils.month_string_to_month_number(time_period)
    dt_str = f"{time_period_number:02d}{time_period}.{eval_period_str}"

    fig_name = f"cmap.FSSbyRegion.{data_names_str}radius{radius:0.2f}{radius_units}.threshold{threshold:0.1f}mm.{time_period_type}.{temporal_res}_hour_precipitation.{dt_str}.{master_region}.png"
    fig_path = os.path.join(utils.plot_output_dir, fig_name)
    print(f"Saving {fig_path}")
    plt.savefig(fig_path)

if __name__ == "__main__":
    regions_dict, verif, da_region_ann_dict = main()
