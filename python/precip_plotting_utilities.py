import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dataclasses
import precip_data_processors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import regionmask
import sys
import utilities as utils

@dataclasses.dataclass
class RegionPlottingConfiguration:
    region_extent: [int, int, int, int]
    figsize_sp: (int, int) # Figure size for single-panel plots
    figsize_mp: (int, int) # Figure size for multi-panel plots
    figsize_mp_5plus: (int, int) # Figure size for multi-panel plots with five or more panels
    cm_mean_precip_range: np.ndarray # Contour map precip color map range
    ts_mean_precip_range: np.ndarray # Time series precip plot limits range
    central_point: [int, int] # Qualitative central point (for annotating text, etc.)

def print_region_config_info(region):
    region_config = regions_info_dict[region]
    print(f"Region is: {region}\n"
          f"Region extent: {region_config.region_extent}\n"
          f"Single-panel plot figure size: {region_config.figsize_sp}\n"
          f"Multi-panel plot figure size: {region_config.figsize_mp}\n"
          f"Multi-panel plot figure size for 5+ panel plots: {region_config.figsize_mp_5plus}\n"
          f"Contour map mean precip range: {region_config.cm_mean_precip_range}\n"
          f"Time series mean precip range {region_config.ts_mean_precip_range}")

regions_info_dict = \
    {
    ##### CONTINENTAL UNITES STATES #####
    "CONUS": RegionPlottingConfiguration(
                region_extent = [-125, -66, 24, 51],
                figsize_sp = (15, 10), 
                figsize_mp = (15, 10),
                figsize_mp_5plus = (15, 7.5),
                cm_mean_precip_range = np.arange(0, 10.5, 0.5),
                ts_mean_precip_range = np.arange(0, 4.5, 0.5),
                central_point = [39.8, -98.6],
                ),
                
    ##### MAIN CONUS SUB-REGIONS #####
    "US-East": RegionPlottingConfiguration(
                region_extent = [-85, -66, 24, 51],
                figsize_sp = (10, 15), 
                figsize_mp = (10, 15), 
                figsize_mp_5plus = (10, 12.5),
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 7.0, 0.5),
                central_point = [37.3, -80.0],
                ),

    "US-Central": RegionPlottingConfiguration(
                region_extent = [-103, -85, 24, 51],
                figsize_sp = (11, 18), 
                figsize_mp = (11, 18), 
                figsize_mp_5plus = (11, 14.5),
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 6.5, 0.5),
                central_point = [38.5, -94.5],
                ),

    "US-Mountain": RegionPlottingConfiguration(
                #region_extent = [-117, -103, 28, 51], # Old extent for mountain states not explicitly included
                region_extent = [-120, -103, 28, 51],
                figsize_sp = (10, 14),
                figsize_mp = (11, 14), # (10, 14) 
                figsize_mp_5plus = (11, 14), # (10, 14)
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 3.0, 0.5),
                central_point = [39.1, -111.8],
                ),

    "US-WestCoast": RegionPlottingConfiguration( # Same domain as US-West, but verification will be confined to WA, OR, and CA
                region_extent = [-129, -113, 28, 51],
                figsize_sp = (10, 15), 
                figsize_mp = (10, 15), 
                figsize_mp_5plus = (10, 12.5),
                cm_mean_precip_range = np.arange(0, 15, 1),
                ts_mean_precip_range = np.arange(0, 7.5, 0.5),
                central_point = [43.0, -121.0],
                ),

    ##### OTHER CONUS SUB-REGIONS #####
    "US-Colorado": RegionPlottingConfiguration(
                region_extent = [-110, -100, 35, 43],
                figsize_sp = (13, 12), 
                figsize_mp = (13, 12), 
                figsize_mp_5plus = (13, 12),
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 3.5, 0.5), 
                central_point = [-999.0, -999.0],
                ),

    "US-GulfCoast": RegionPlottingConfiguration(
                region_extent = [-95, -80, 24, 32], 
                figsize_sp = (15, 11), 
                figsize_mp = (15, 11), 
                figsize_mp_5plus = (15, 8),
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 9.5, 0.5), 
                central_point = [-999.0, -999.0],
                ),

    "US-SouthEast": RegionPlottingConfiguration(
                region_extent = [-95, -75, 24, 37], 
                figsize_sp = (15, 11), 
                figsize_mp = (15, 11), 
                figsize_mp_5plus = (15, 9.5),
                cm_mean_precip_range = np.arange(0, 6.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 8.5, 0.5),
                central_point = [-999.0, -999.0],
                ),


    ##### OTHER PARTS OF THE WORLD #####
    # TODO: Additional region for equatorial+subtropical Africa only?
    "Africa": RegionPlottingConfiguration( 
                region_extent = [-20, 55, -38, 40],
                figsize_sp = (10, 12), 
                figsize_mp = (13, 17.5), 
                figsize_mp_5plus = (13, 15.5),
                cm_mean_precip_range = np.arange(0, 17, 1),
                ts_mean_precip_range = np.arange(0, 4.5, 0.5),
                central_point = [-999.0, -999.0],
                ),

    "Australia": RegionPlottingConfiguration(
                region_extent = [112, 156, -45, -9],
                figsize_sp = (13, 11), 
                figsize_mp = (13, 11), 
                figsize_mp_5plus = (13, 10),
                cm_mean_precip_range = np.arange(0, 10.5, 0.5),
                ts_mean_precip_range = np.arange(0, 6.5, 0.5),
                central_point = [-999.0, -999.0],
                ),

    "EastPacAR": RegionPlottingConfiguration( # Extends into Pacific to capture ARs
                region_extent = [-150, -113, 28, 51],
                figsize_sp = (15, 10), 
                figsize_mp = (15, 10), 
                figsize_mp_5plus = (15, 9),
                cm_mean_precip_range = np.arange(0, 10.5, 0.5), 
                ts_mean_precip_range = np.arange(0, 6.5, 0.5), 
                central_point = [-999.0, -999.0],
                ),

    "Europe": RegionPlottingConfiguration(
                region_extent = [-12, 50, 35, 72],
                figsize_sp = (15, 12), 
                figsize_mp = (15, 12), 
                figsize_mp_5plus = (15, 10),
                cm_mean_precip_range = np.arange(0, 6.5, 0.5),
                ts_mean_precip_range = np.arange(0, 4.5, 0.5),
                central_point = [-999.0, -999.0],
                ),

    "Global": RegionPlottingConfiguration(
                region_extent = [-180, 180, -90, 90],
                figsize_sp = (15, 10), 
                figsize_mp = (15, 10.5),
                figsize_mp_5plus = (15, 9),
                cm_mean_precip_range = np.arange(0, 21, 1),
                ts_mean_precip_range = np.arange(0, 4.0, 0.5), 
                central_point = [-999.0, -999.0],
                ),

    "MaritimeContinent": RegionPlottingConfiguration(
                region_extent = [90, 160, -18, 18],
                figsize_sp = (15, 11), 
                figsize_mp = (15, 11),
                figsize_mp_5plus = (15, 9),
                cm_mean_precip_range = np.arange(0, 21, 1),
                ts_mean_precip_range = np.arange(0, 13, 1),
                central_point = [-999.0, -999.0],
                ),

    "MJO": RegionPlottingConfiguration(
                region_extent = [70, 170, -18, 18],
                figsize_sp = (16, 10), 
                figsize_mp = (16, 11), 
                figsize_mp_5plus = (16, 9),
                cm_mean_precip_range = np.arange(0, 21, 1),
                ts_mean_precip_range = np.arange(0, 13, 1),
                central_point = [-999.0, -999.0],
                ),

    "SouthAmerica": RegionPlottingConfiguration(
                region_extent = [-83, -33, -57, 13],
                figsize_sp = (9, 14), 
                figsize_mp = (9, 14), 
                figsize_mp_5plus = (9, 12),
                cm_mean_precip_range = np.arange(0, 21, 1),
                ts_mean_precip_range = np.arange(0, 11, 1),
                central_point = [-999.0, -999.0],
                ),

    "WestPacJapan": RegionPlottingConfiguration(
                region_extent = [118, 178, 20, 50],
                figsize_sp = (14, 10), 
                figsize_mp = (14, 10), 
                figsize_mp_5plus = (14, 8),
                cm_mean_precip_range = np.arange(0, 15.5, 0.5),
                ts_mean_precip_range = np.arange(0, 8, 1),
                central_point = [-999.0, -999.0],
                ),
    }

# Map three-letter month strings to the correponding numerical month of the year.
def month_string_to_month_number(month_string):
    month_to_number_dict = {
                           "JAN" :  1,
                           "FEB" :  2,
                           "MAR" :  3,
                           "APR" :  4,
                           "MAY" :  5,
                           "JUN" :  6,
                           "JUL" :  7,
                           "AUG" :  8,
                           "SEP" :  9,
                           "OCT" : 10,
                           "NOV" : 11,
                           "DEC" : 12,
                           }
    return month_to_number_dict[month_string]

# Map three-letter season strings to the corresponding numerical season (chose
# to start counting with DJF = 1).
def season_string_to_season_number(season_string):
    season_to_number_dict = {
                           "DJF" :  1,
                           "MAM" :  2,
                           "JJA" :  3,
                           "SON" :  4,
                           }
    return season_to_number_dict[season_string]

# Returns a 12-element list of three-letter months strings.
def construct_monthly_string_list():
    return ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Returns a 4-element list of three-letter season strings.
def construct_seasonal_string_list():
    return ["DJF", "MAM", "JJA", "SON"]

# Dictionary mapping dataset names to colors to use for the respective
# lines on timeseries plots
time_series_color_dict = {"AORC":       "blue",
                          "CONUS404": "purple",
                          "ERA5":        "red",
                          "IMERG":     "green",
                          "Replay":   "orange",
                         }

# Create the CONUS mask, excluding AK and HI, to only include data over the lower 48 states (i.e., the CONUS)
# Each state+DC is represented by an integer between 0 and 50 (clunky, but that's the way it is)
# So there are 51 "states" total (regionmask refers to each stats as a region)
def create_conus_mask(data_array):
    # Define a regionmask object representing all 50 US states
    states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50

    # Create a mask valid within all 50 US states, in the form of an xarray DataArray
    # with coordinates the same as <data_array>. The mask method determines which grid
    # points of the array lie within the region. Grid points outside of the region are set to NaN.
    # Gridpoints within the region are encoded with an integer representing the region.
    # In this case, each of the 50 states is defined with a different integer. 
    mask = states.mask(data_array)

    # Get the integers representing the non-CONUS states Alaska and Hawaii (don't want to include data over these states in CONUS stats) 
    AK_index = states.map_keys("Alaska")
    HI_index = states.map_keys("Hawaii")

    # Convert the mask to a boolean DataArray that is True over all the states except AK and HI
    # We'll use this boolean DataArray to pull out only data from the precip DataArrays within the lower-48 (CONUS) states 
    mask_conus = (mask != AK_index) & (mask != HI_index) & (mask >= 0) 

    return mask_conus

# Follow the logic above for the CONUS mask, but for only the states within the US-Mountain region. 
def create_mountain_states_mask(data_array):
    states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
    mask = states.mask(data_array)
    ID_index = states.map_keys("Idaho")
    NV_index = states.map_keys("Nevada")
    UT_index = states.map_keys("Utah")
    AZ_index = states.map_keys("Arizona")
    NM_index = states.map_keys("New Mexico")
    CO_index = states.map_keys("Colorado")
    WY_index = states.map_keys("Wyoming")
    MT_index = states.map_keys("Montana")
    ND_index = states.map_keys("North Dakota")
    SD_index = states.map_keys("South Dakota")
    NE_index = states.map_keys("Nebraska")
    TX_index = states.map_keys("Texas")
    mask_west_coast_states = (mask == ID_index) | (mask == NV_index) | (mask == UT_index) | \
                             (mask == AZ_index) | (mask == NM_index) | (mask == CO_index) | \
                             (mask == WY_index) | (mask == MT_index) | (mask == ND_index) | \
                             (mask == SD_index) | (mask == NE_index) | (mask == TX_index)   

    return mask_west_coast_states

# Follow the logic above for the CONUS mask, but for only the states of WA, OR, and CA
def create_west_coast_states_mask(data_array):
    states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
    mask = states.mask(data_array)
    WA_index = states.map_keys("Washington")
    OR_index = states.map_keys("Oregon")
    CA_index = states.map_keys("California")
    mask_west_coast_states = (mask == WA_index) | (mask == OR_index) | (mask == CA_index)

    return mask_west_coast_states

# Follow the logic above for the CONUS mask, but for all countries that are part of additional
# continental regions. NOTE: France is included in the South America list to capture French Guiana. 
def create_continent_mask(data_array, region):
    countries = regionmask.defined_regions.natural_earth_v5_1_2.countries_50
    mask = countries.mask(data_array)

    country_list_file = os.path.join(utils.home_dir, "regionmask_country_lists", f"{region}.csv")
    country_list = [i.strip() for i in open(country_list_file).readlines()]
    country_indices = [countries.map_keys(i) for i in country_list]

    for i, country_index in enumerate(country_indices):
        if (i > 0):
            mask_continent = (mask == country_index) | (mask_continent == True) 
        else:
            mask_continent = (mask == country_index)

    return mask_continent

def set_cbar_labels_rotation_and_fontsize(plot_levels, region, num_da, for_single_cbar = False):
    if (type(plot_levels) is list):
        plot_levels = np.array(plot_levels)

    rotation = 0
    fontsize = 15

    # If for a smaller colorbar on a subplot, set small rotation 
    if not(for_single_cbar):
        rotation = 15
        
    # If the number of levels is large, set a larger rotation and reduce the fontsize
    if (len(plot_levels) >= 10): #or (np.max(np.abs(plot_levels)) >= 100):
        rotation = 60
        fontsize = 14 # was 13

        # Finally, if the figure is taller than it is wide, reduce the fontsize
        # and increase the rotation further. Even with the changes above,
        # cbar labels will to be too squished on "tall" plots (e.g., US-WestCoast)
        if not(for_single_cbar):
            #fontsize = 12

            if (num_da >= 5): 
                region_figsize = regions_info_dict[region].figsize_mp_5plus
            else:
                region_figsize = regions_info_dict[region].figsize_mp
            figsize_ratio = region_figsize[0]/region_figsize[1]

            if (figsize_ratio < 1.0):
                fontsize = 13 # was 10
                rotation = 75

    return rotation, fontsize

# Plot limits to use for contour maps for different variables
# and for each accumulation period for precipitation.
def variable_plot_limits(var_name, temporal_res = "native"):
    match var_name:
        case utils.accum_precip_var_name:
            match temporal_res:
                case "native":
                    return np.arange(0, 32, 2) # mm 
                case 1:
                    return np.arange(0, 42, 2) 
                case 3:
                    return np.arange(0, 75, 5)
                case 6:
                    return np.arange(0, 90, 10)
                case 12:
                    return np.arange(0, 130, 10) 
                case 24:
                    return np.arange(0, 210, 10)
                case 48:
                    return np.arange(0, 310, 10)
                case 72:
                    return np.arange(0, 380, 20)
                case 120:
                    return np.arange(0, 440, 20)
                case 168:
                    return np.arange(0, 750, 50)
                case _:
                    print(f"No accumulated precip data set with resolution {temporal_res}") 
                    sys.exit(1)
        case "tmp2m":
            return np.arange(220, 325, 5) # Kelvin
        case _:
            print(f"Variable plot limits not defined for var {var_name}")
            sys.exit(1)

# Plot limits to use for contour maps of percentiles for different variables
# and for each accumulation period for precipitation.
# TODO: Use different plot limits for 95th and 99th percentiles (and additional percentiles)?
def variable_pctl_plot_limits(var_name, temporal_res = "native"):
    match var_name:
        case utils.accum_precip_var_name:
            match temporal_res:
                case "native":
                    return np.arange(0, 11, 1) # mm 
                case 1:
                    return np.arange(0, 11, 1) 
                case 3:
                    return np.arange(0, 22, 2)
                case 6:
                    return np.arange(0, 36, 5)
                case 12:
                    return np.arange(0, 55, 5) 
                case 24:
                    return np.arange(0, 110, 10)
                case _:
                    print(f"No accumulated precip data set with resolution {temporal_res}") 
                    sys.exit(1)
        case "tmp2m":
            return np.arange(280, 330, 5) # Kelvin
        case _:
            print(f"Variable percentile plot limits not defined for var {var_name}")
            sys.exit(1)

# Label only every-other tick of a colorbar and use "" (blank) for the other ticks
# This logic is necessary since the lists of colorbar ticks and tick labels must have the same length
# Useful for small colorbars on multi-panel subplots
def create_sparse_cbar_ticks(plot_levels):
    cbar_tick_labels = []
    for level in plot_levels[::2]:
        cbar_tick_labels.append(level)
        cbar_tick_labels.append("")

    if (len(cbar_tick_labels) == len(plot_levels) + 1):
        cbar_tick_labels = cbar_tick_labels[:-1]

    return cbar_tick_labels

# Create subplots using subplot2grid for use in multi-panel contour map.
def create_gridded_subplots(num_da, proj, single_colorbar = True):
    cbar_ax = None
    match num_da:
        case 1:
            single_colorbar = False # Keyword is irrelevant if we only have a single plot, so set it to False
            axes_list = [
                        plt.subplot2grid((1, 1), (0, 0), projection = proj),
                        ]
        case 2:
            if single_colorbar:
                axes_list = [
                            plt.subplot2grid((5, 4), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((5, 4), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            ]
                cbar_ax = plt.subplot2grid((5, 4), (4, 1), colspan = 2, rowspan = 1)
            else:
                axes_list = [
                            plt.subplot2grid((4, 4), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((4, 4), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            ]
        case 3:
            if single_colorbar:
                axes_list = [
                            plt.subplot2grid((9, 4), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((9, 4), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((9, 4), (4, 1), colspan = 2, rowspan = 4, projection = proj),
                            ]
                cbar_ax = plt.subplot2grid((9, 4), (8, 1), colspan = 2, rowspan = 1)
            else:
                axes_list = [
                            plt.subplot2grid((8, 4), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 4), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 4), (4, 1), colspan = 2, rowspan = 4, projection = proj),
                            ]
        case 4:
            if single_colorbar:
                axes_list = [
                            plt.subplot2grid((9, 4), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((9, 4), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((9, 4), (4, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((9, 4), (4, 2), colspan = 2, rowspan = 4, projection = proj),
                            ]
                cbar_ax = plt.subplot2grid((9, 4), (8, 1), colspan = 2, rowspan = 1)
            else:
                axes_list = [
                            plt.subplot2grid((8, 4), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 4), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 4), (4, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 4), (4, 2), colspan = 2, rowspan = 4, projection = proj),
                            ]
        case 5:
            if single_colorbar:
                axes_list = [
                            plt.subplot2grid((9, 6), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((9, 6), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((9, 6), (0, 4), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((9, 6), (4, 1), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((9, 6), (4, 3), colspan = 2, rowspan = 4, projection = proj),
                            ]
                cbar_ax = plt.subplot2grid((9, 6), (8, 1), colspan = 4, rowspan = 1)
            else:
                axes_list = [
                            plt.subplot2grid((8, 6), (0, 0), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (0, 2), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (0, 4), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (4, 1), colspan = 2, rowspan = 4, projection = proj),
                            plt.subplot2grid((8, 6), (4, 3), colspan = 2, rowspan = 4, projection = proj),
                            ]
        case _:
            print("Error: Can't handle {num_da}-paneled subplot")
            return

    return axes_list, cbar_ax

# For each time in the period_end_time dimension of the data array, create a single-paneled contour plot of precipitation
def plot_cmap_single_panel(data_array, data_name, region, temporal_res = "native", use_contourf = False, sparse_cbar_ticks = False):
    dtimes = [pd.Timestamp(i) for i in data_array[utils.period_end_time_dim_str].values]
    for dtime in dtimes:
        # Select data to plot (at one valid time)
        data_to_plot = data_array.sel(period_end_time = dtime)

        # Set figure title and name
        formatted_short_name = precip_data_processors.format_short_name(data_to_plot)
        valid_dt_str = dtime.strftime("%Y%m%d.%H")
        print(f"Plotting {data_name} {data_array.short_name} data at {valid_dt_str}")

        # Set up the figure
        proj = ccrs.PlateCarree() 
        plt.figure(figsize = regions_info_dict[region].figsize_sp) 
        axis = plt.axes(projection = proj)
        axis.coastlines()
        axis.set_extent(regions_info_dict[region].region_extent, crs = proj)
        axis.add_feature(cfeature.BORDERS)
        if ("US" in region):
            axis.add_feature(cfeature.STATES)
        gl = axis.gridlines(crs = proj, color = "gray", alpha = 0.5, draw_labels = True,
                            linewidth = 0.5, linestyle = "dashed")
        
        # Plot the data
        levels = variable_plot_limits(utils.accum_precip_var_name, temporal_res = temporal_res)
        if use_contourf:
            lon_mesh, lat_mesh = np.meshgrid(data_to_plot["lon"], data_to_plot["lat"])
            plot_handle = axis.contourf(lon_mesh, lat_mesh, data_to_plot, levels = levels, extend = "both", cmap = "viridis")
            plt.colorbar(plot_handle, orientation = "vertical", shrink = 0.7)
            plot_handle.colorbar.ax.set_yticks(levels)
        else:
            plot_handle = data_to_plot.plot(ax = axis, levels  = levels, extend = "both", cmap = "viridis",
                                            cbar_kwargs = {"orientation": "vertical", "shrink": 0.7, "ticks": levels})
        
        # Configure axis labels, colorbar, and figure name
        plot_handle.colorbar.set_label(f"{data_array.short_name} [{data_array.units}]", size = 15)
        if sparse_cbar_ticks:
            cbar_tick_labels = create_sparse_cbar_ticks(levels) 
        else:
            cbar_tick_labels = levels    
        plot_handle.colorbar.ax.set_yticklabels(cbar_tick_labels)
        plot_handle.colorbar.ax.tick_params(labelsize = 15)
        plt.title(f"{data_name} {data_array.short_name} ending at {valid_dt_str}", fontsize = 15)
        plt.xlabel("Latitude", fontsize = 15)
        plt.ylabel("Longitude", fontsize = 15)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.tight_layout()

        # Save figure
        fig_name = f"cmap.{data_name}.{formatted_short_name}.{valid_dt_str}.{region}.png"
        fig_path = os.path.join(utils.plot_output_dir, fig_name)
        print(f"Saving {fig_path}")
        plt.savefig(fig_path)

# Plot a blank map for each region defined in regions_info_dict 
# This can be useful to assess whether the bounds for each region need to be adjusted, for example. 
def plot_blank_map_of_each_region():
    for region, region_config in regions_info_dict.items():
        plt.figure(figsize = region_config.figsize_sp)
        proj = ccrs.PlateCarree() 
        axis = plt.axes(projection = proj)
        axis.coastlines()
        axis.set_extent(region_config.region_extent, crs = proj)
        axis.add_feature(cfeature.BORDERS)
        if ("US" in region):
            axis.add_feature(cfeature.STATES)
        gl = axis.gridlines(crs = proj, color = "gray", alpha = 0.5, draw_labels = False,
                            linewidth = 0.5, linestyle = "dashed")
        plt.title(region, fontsize = 20)
        plt.tight_layout()

        fig_name = f"region.{region}.blank_map.png"
        fig_path = os.path.join(utils.plot_output_dir, fig_name)
        print(f"Saving {fig_path}") 
        plt.savefig(fig_path)

