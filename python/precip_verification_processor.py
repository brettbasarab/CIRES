import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dataclasses
import datetime as dt
import dateutil
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import precip_data_processors as pdp
import precip_plotting_utilities as pputils
import pylab
import scipy
import sys
import utilities as utils
import xarray as xr

# TODO (both low priority):
    # Figure out the source of the "All-NaN slice encountered" warning in percentile calculations
    # Update plot names to use '.' notation

@dataclasses.dataclass
class StatsDataClass:
    threshold: float
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int
    total_events: int
    frequency_bias: float
    CSI: float
    ETS: float

# Keyword to evaluate FSS for varying radius, using a fixed amount threshold
evaluate_by_radius_kw_str = "by_radius"

# Keyword to evaluate FSS for varying radius, using a fixed ARI grid as threshold
evaluate_by_radius_ari_threshold_kw_str = "by_radius_ari_threshold"

# Keyword to evaluate FSS for varying amount thresholds 
evaluate_by_threshold_kw_str = "by_threshold"

# Keyword to evaluate FSS using varying ARI grids as thresholds
evaluate_by_ari_kw_str = "by_ari_grid"

# List of variables that can be used as keyword to evaluate FSS for varying radius
#evaluate_by_radius_kw_options = [evaluate_by_radius_kw_str, "radius", "r"]

# For a given region, return a list of the names of the datasets currently used
# in verification, as well as the dataset that will be considered truth (truth_data_name)
def map_region_to_data_names(region, verif_grid = utils.Replay_data_name, include_hrrr = False):
    if ("US" in region): 
        truth_data_name = utils.AORC_data_name 
        data_grid = utils.Replay_data_name 
        data_names = [utils.AORC_data_name, utils.CONUS404_data_name, utils.ERA5_data_name,
                      utils.IMERG_data_name, utils.Replay_data_name] # ["AORC", "CONUS404", "ERA5", "IMERG", "Replay"]
        if (verif_grid == utils.AORC_data_name): 
            data_grid = utils.AORC_data_name
            data_names = [utils.AORC_data_name, utils.CONUS404_data_name, utils.NestedReplay_data_name, utils.Replay_data_name] # ["AORC", "CONUS404", "NestedReplay", "Replay"]
        if include_hrrr:
            data_names.append(utils.HRRR_data_name)
    else: 
        truth_data_name = utils.IMERG_data_name
        data_grid = utils.Replay_data_name
        data_names = [utils.IMERG_data_name, utils.Replay_data_name, utils.ERA5_data_name] # ["IMERG", "Replay", "ERA5"]

    return sorted(data_names), truth_data_name, data_grid

class PrecipVerificationProcessor(object):
    def __init__(self, start_dt_str, end_dt_str,
                 LOAD_DATA_FLAG = True, # Set to load data directly that we'll need, overrides external_da_dict
                 IS_STANDARD_INPUT_DICT = True, # Set to assume that external_da_dict is a dictionary of the form {data_name_1: da_1, ...,data_name_N: da_N}  
                 external_da_dict = None, # Dictionary of precalculated data arrays for which to make plots (only used if LOAD_DATA_FLAG = False)
                 data_names = ["AORC", "IMERG", "Replay", "ERA5"],
                 truth_data_name = "AORC", # Dataset that is considered truth (i.e., observations) for this particular verification
                 data_grid = "Replay", # Dataset whose grid we're performing verification on
                 model_var_list = ["prateb_ave"],
                 region = "Global",
                 region_extent = None,
                 temporal_res = 3,
                 thresholds = [0.25, 1.0, 10.0], #, 25.0, 50.0], # Thresholds in mm
                 percentiles = [95, 99], # Percentiles to calculate 
                 user_dir = "bbasarab",
                 poster = False): 

        self.data_grid = data_grid
        self.data_grid_name = pdp.set_grid_name_for_file_names(self.data_grid)
        self.native_grid_name = pdp.set_grid_name_for_file_names("Native")
        self.model_var_list = model_var_list
        self.temporal_res = temporal_res
        self.thresholds = thresholds
        self.percentiles = percentiles
        self.num_thresholds = len(self.thresholds)
        self.variable_plot_limits = pputils.variable_plot_limits("accum_precip", temporal_res = self.temporal_res)
        self.variable_pctl_plot_limits = pputils.variable_pctl_plot_limits("accum_precip", temporal_res = self.temporal_res)
        self.replay_grid_cell_size = 0.234375

        self.user_dir = user_dir
        self.home_dir = os.path.join("/home", self.user_dir)
        self.data_dir = os.path.join("/data", self.user_dir)
        self.netcdf_dir = os.path.join(self.data_dir, "netcdf") 
        self.plot_output_dir = os.path.join(self.home_dir, "plots") 
        self.stats_output_dir = os.path.join(self.home_dir, "stats")

        # If data are on the Replay grid, longitudes go from 0 to 360,
        # rather than -180  to 180, and the latitude coordinates are flipped,
        # i.e., they go from +90 to -90 (north to south).
        if (self.data_grid == "Replay"):
            self.LONS_360_FLAG = True 
            self.LATS_FLIP_FLAG = True 
        else:
            self.LONS_360_FLAG = False 
            self.LATS_FLIP_FLAG = False
        self._set_region_info(region, region_extent)

        # If plots are for a poster, increase the fontsize of plot titles, axes labels, etc. 
        self.poster_font_increase = 0
        if poster:
            self.poster_font_increase = 2

        # Create lists of valid datetimes at the desired temporal resolution and at a daily cadence
        self.start_dt_str = start_dt_str
        self.end_dt_str = end_dt_str
        print(f"Start datetime: {self.start_dt_str}")
        print(f"End datetime: {self.end_dt_str}")
        self._construct_valid_dt_list()

        # Load data for verification 
        self.data_names = data_names
        self.truth_data_name = truth_data_name
        self._reorder_data_names_with_truth_data_first()
        self.data_names_str = "".join(f"{data_name}." for data_name in self.data_names)
        self.num_datasets = len(self.data_names)

        print(f"Data name list: {self.data_names}")
        print(f"Truth data name: {self.truth_data_name}")

        self.LOAD_DATA_FLAG = LOAD_DATA_FLAG
        self.IS_STANDARD_INPUT_DICT = IS_STANDARD_INPUT_DICT
        if self.LOAD_DATA_FLAG:
            # Read model and obs data, check that the data array shapes match, then load the data
            self._read_region_subset_and_load_data()
            self._sum_data_over_full_time_period()
            self.truth_da_summed_time = self.da_dict_summed_time[self.truth_data_name]
        else:
            print("**** LOAD_DATA_FLAG not set; will perform any verification using external data arrays provided")
            if (external_da_dict is None):
                print("Error: LOAD_DATA_FLAG not set but no external data arrays provided; can't perform verification")
                sys.exit(1)
            else:
                if self.IS_STANDARD_INPUT_DICT:
                    self.da_dict = {}
                    for data_name in self.data_names: # This loop ensures self.da_dict will maintain desired order of datasets (with truth data first)
                        if (data_name not in external_da_dict.keys()):
                            print(f"Error: Dataset {data_name} not in provided dictionary of external data arrays ({external_da_dict.keys()})")
                            sys.exit(1)
                        self.da_dict[data_name] = external_da_dict[data_name]
                else:
                    self.da_dict = self._create_dummy_data_dict_for_non_standard_input() 
                self.truth_da = self.da_dict[self.truth_data_name]

    ##### PRIVATE METHODS #####
    ############################################################################
    def _create_dummy_data_dict_for_non_standard_input(self):
        da_dict = {}
        da = xr.DataArray(np.arange(10))
        pdp.add_attributes_to_data_array(da, short_name = f"{self.temporal_res}-hour precipitation", units = "mm")
        da_dict[self.truth_data_name] = da

        return da_dict

    def _construct_path_to_nc_precip_data(self, data_name):
        if (type(self.temporal_res) is int):
            temporal_res_str = f"{self.temporal_res:02d}_hour_precipitation"
        else:
            temporal_res_str = f"{self.temporal_res}_res_precipitation"

        if (data_name == self.data_grid):
            data_dir = f"{data_name}.{self.native_grid_name}.{temporal_res_str}"
        else:
            data_dir = f"{data_name}.{self.data_grid_name}.{temporal_res_str}"
        full_data_dir = os.path.join(self.netcdf_dir, data_dir)
        print(f"Data netcdf directory: {full_data_dir}")
        if (not os.path.exists(full_data_dir)):
            print(f"Error: Obs data directory {full_data_dir} does not exist")
            sys.exit(1)

        return full_data_dir, temporal_res_str

    def _construct_valid_dt_list(self):
        self.start_dt = pdp.check_model_valid_dt_format(self.start_dt_str, resolution = self.temporal_res) 
        self.end_dt = pdp.check_model_valid_dt_format(self.end_dt_str, resolution = self.temporal_res)
        
        # Construct annual datetime list (for annual stats, timeseries, etc.)
        # Because data we're working with are period-ending, if self.end_dt is, for example, 20090101.00,
        # we don't actually have data over the final year (2009 in this example). Remove final year from self.valid_annual_dt_list. 
        self.first_year = dt.datetime.strptime(self.start_dt_str[:4], "%Y")
        self.final_year = dt.datetime.strptime(self.end_dt_str[:4], "%Y")
        self.valid_annual_dt_list = list(dateutil.rrule.rrule(dateutil.rrule.YEARLY, dtstart = self.first_year, until = self.final_year)) 
        if (self.end_dt.month == 1) and (self.end_dt.day == 1) and (self.end_dt.hour == 0):
            self.valid_annual_dt_list = self.valid_annual_dt_list[:-1]
        self.annual_time_period_str = f"{self.valid_annual_dt_list[0]:%Y}-{self.valid_annual_dt_list[-1]:%Y}"
        
        # Construct monthly datetime list (for monthly stats, timeseries, etc.)
        # Because data we're working with are period-ending, if self.end_dt is, for example, 20080601.00,
        # we don't actually have data over the final month (June 2008 in this example). Remove last month from self.valid_monthly_dt_list. 
        self.first_month = dt.datetime.strptime(self.start_dt_str[:6], "%Y%m")
        self.final_month = dt.datetime.strptime(self.end_dt_str[:6], "%Y%m")
        self.valid_monthly_dt_list = list(dateutil.rrule.rrule(dateutil.rrule.MONTHLY, dtstart = self.first_month, until = self.final_month))
        if (self.end_dt.day == 1) and (self.end_dt.hour == 0):
            self.valid_monthly_dt_list = self.valid_monthly_dt_list[:-1]
        self.monthly_time_period_str = f"{self.valid_monthly_dt_list[0]:%Y%m}-{self.valid_monthly_dt_list[-1]:%Y%m}"

        # Construct daily datetime list (which will correspond to cadence of input netCDF files)
        current_daily_dt = dt.datetime(self.start_dt.year, self.start_dt.month, self.start_dt.day)
        end_daily_dt = dt.datetime(self.end_dt.year, self.end_dt.month, self.end_dt.day)
        self.valid_daily_dt_list = [current_daily_dt]
        while (current_daily_dt != end_daily_dt):
            current_daily_dt += dt.timedelta(days = 1)
            self.valid_daily_dt_list.append(current_daily_dt)
        self.valid_daily_dt_list_period_begin = [dtime - dt.timedelta(days = 1) for dtime in self.valid_daily_dt_list[1:]]
        self.daily_time_period_str = f"{self.valid_daily_dt_list[0]:%Y%m%d}-{self.valid_daily_dt_list[-1]:%Y%m%d}"
        self.daily_time_period_str_period_begin = f"{self.valid_daily_dt_list_period_begin[0]:%Y%m%d}-{self.valid_daily_dt_list_period_begin[-1]:%Y%m%d}"

        # Construct datetime list corresponding to temporal resolution of the data
        # AND representing period-END times of the data (e.g., the first valid time for
        # 24-hour precip is the beginning of the second day). 
        if (self.temporal_res == 24):
            self.valid_dt_list = self.valid_daily_dt_list[1:]
        elif (self.temporal_res == 3) or (self.temporal_res == 1):
            current_dt = self.start_dt
            self.valid_dt_list = [current_dt]
            while (current_dt != self.end_dt):
                current_dt += dt.timedelta(hours = self.temporal_res)
                self.valid_dt_list.append(current_dt)
        else:
            raise NotImplementedError
        self.time_period_str = f"{self.valid_dt_list[0]:%Y%m%d.%H}-{self.valid_dt_list[-1]:%Y%m%d.%H}"

        print(f"Annual time period: {self.annual_time_period_str}")
        print(f"Monthly time period: {self.monthly_time_period_str}")
        print(f"Daily time period: {self.daily_time_period_str}")
        print(f"Daily time period (period-beginning): {self.daily_time_period_str_period_begin}")
        print(f"Data resolution time period: {self.time_period_str}")

    def _create_region_mask(self, data_array):
        if (self.region == "US-Mountain"):
            self.region_mask = pputils.create_mountain_states_mask(data_array)
        elif (self.region == "US-WestCoast"):
            self.region_mask = pputils.create_west_coast_states_mask(data_array) 
        elif ("US" in self.region):
            self.region_mask = pputils.create_conus_mask(data_array)
        elif (self.region == "Africa") or \
             (self.region == "Australia") or \
             (self.region == "Europe") or \
             (self.region == "SouthAmerica"):
            self.region_mask = pputils.create_continent_mask(data_array, self.region)
        else:
            self.region_mask = None
            return data_array

        # Use the xarray.DataArray.where method to drop any values (i.e., turn them into nans) that don't fall within the region mask
        data_array = data_array.where(self.region_mask)
        return data_array 

    def _read_region_subset_and_load_data(self):
        self.da_dict = {}
        for data_name in self.data_names:
            print(f"**** Reading dataset {data_name}")
            dataset_dir, temporal_res_str = self._construct_path_to_nc_precip_data(data_name)

            # Collect netCDF file list
            file_list = []
            for dtime in self.valid_daily_dt_list:
                if (data_name == self.data_grid):
                    fname = f"{data_name}.{self.native_grid_name}.{temporal_res_str}.{dtime:%Y%m%d}.nc"
                else:
                    fname = f"{data_name}.{self.data_grid_name}.{temporal_res_str}.{dtime:%Y%m%d}.nc"
                fpath = os.path.join(dataset_dir, fname)
                if (not os.path.exists(fpath)):
                    print(f"Warning: Input file path {fpath} does not exist; not including in input file list")
                    continue
                file_list.append(fpath)

            if (len(file_list) == 0):
                print(f"Error: No input files found in directory {dataset_dir}; can't proceed with verification")
                sys.exit(1)

            # Read multi-file dataset
            dataset = xr.open_mfdataset(file_list)
            precip_da = dataset[f"precipitation_{self.temporal_res}_hour"]
            precip_da.attrs["data_name"] = data_name

            # Index obs data array to correct datetime range
            precip_da = precip_da.loc[self.start_dt.strftime(utils.full_date_format_str):self.end_dt.strftime(utils.full_date_format_str)]

            # Subset data to region
            self.da_dict[data_name] = self._subset_data_to_region(precip_da, data_name = data_name)

        # Ensure the shapes of all the data arrays are the same. 
        print("**** Checking consistency of DataArray shapes")    
        self.truth_da = self.da_dict[self.truth_data_name] 
        truth_da_shape = self.truth_da.shape
        for data_name, da in self.da_dict.items():
            if (da.shape != truth_da_shape): 
                print(f"Error: {data_name} data {da.shape} has a different shape than {self.truth_data_name} {truth_da_shape}; not proceeding with verification")
                sys.exit(1)
            print(f"{data_name} data {da.shape} has the same shape as {self.truth_data_name} data {truth_da_shape}")

        # Load each DataArray. Maintain a separate loop for this in case any of the DataArrays are
        # found to be the wrong shape above. We don't want to spend time loading each DataArray
        # only to discover that one is the wrong shape and we can't perform verification.
        print("**** Loading DataArrays")    
        for data_name, da in self.da_dict.items():
            print(f"Loading region-subsetted {data_name} data") 
            da.load()
   
    # If truth_data_name is not the first index, swap it with whatever
    # is the first index 
    def _reorder_data_names_with_truth_data_first(self):
        truth_data_index = self.data_names.index(self.truth_data_name)
        if (truth_data_index != 0):
            self.data_names[truth_data_index] = self.data_names[0]
            self.data_names[0] = self.truth_data_name 

    def _set_region_info(self, region, region_extent):
        self.region = region

        if self.region in pputils.regions_info_dict.keys(): 
            self.region_extent = pputils.regions_info_dict[self.region].region_extent
        else:
            if (type(region_extent) is list) and \
            (len(region_extent) == 4) and \
            (type(region_extent[0]) is int) and \
            (type(region_extent[1]) is int) and \
            (type(region_extent[2]) is int) and \
            (type(region_extent[3]) is int) and \
            (-180 <= region_extent[0] <= 180) and \
            (-180 <= region_extent[1] <= 180) and \
            (-90  <= region_extent[2] <= 90)  and \
            (-90  <= region_extent[3] <= 90):
                self.region_extent = region_extent
            else:
                print(f"Error: Non-standard region {region} provided without accompanying region extent lat/lon list; can't configure nonstandard region")
                sys.exit(1)

        # Index by lat/lon corresponding to current region. To do this,
        # recall data goes from [0, 360) longitude and (90, -90) latitude.
        # So the longitude bounds must be adjusted first
        self.lower_lon = self.region_extent[0]
        self.upper_lon = self.region_extent[1]
        self.lower_lat = self.region_extent[2]
        self.upper_lat = self.region_extent[3]
        if (self.LONS_360_FLAG):
            if (self.lower_lon < 0):
                self.lower_lon += 360.0
            if (self.upper_lon < 0):
                self.upper_lon += 360.0

        # Determine if a region needs to be split into two subregions to be indexed properly
        # This is the case for Africa and Europe, which split across the meridian
        match self.region:
            case "Africa":
                self.region_spans_meridian = True
            case "Europe":
                self.region_spans_meridian = True
            case _:
                self.region_spans_meridian = False
    
    # FIXME: Discontinuity in IMERG and ERA5 data in Europe and Africa where data crosses meridian 
    def _subset_data_to_region(self, data_array, data_name = None):
        print(f"Subsetting {data_name} data to region {self.region}")

        if (self.region == "Global"):
            return data_array
        else:
            if (self.region_spans_meridian) and (self.LONS_360_FLAG):
                if (self.LATS_FLIP_FLAG):
                    region_subset_west_of_meridian = data_array.sel(lat = slice(self.upper_lat, self.lower_lat), lon = slice(self.lower_lon, 360.0)) 
                    region_subset_east_of_meridian = data_array.sel(lat = slice(self.upper_lat, self.lower_lat), lon = slice(0.0, self.upper_lon))
                else:
                    region_subset_west_of_meridian = data_array.sel(lat = slice(self.lower_lat, self.upper_lat), lon = slice(self.lower_lon, 360.0)) 
                    region_subset_east_of_meridian = data_array.sel(lat = slice(self.lower_lat, self.upper_lat), lon = slice(0.0, self.upper_lon))
                region_subset_data_array = xr.concat([region_subset_west_of_meridian, region_subset_east_of_meridian], dim = "lon")

                # The slicing above will leave us with longitudes that go from (for example), ~340 to 360, then start over at zero. In other words,
                # they are numerically out of order, which makes further manipulation and plotting of the array difficult.
                # So, modify the longitudes to be increasing from negative (west of meridian) to positive (east of meridian),
                # then reorder the data array by longitude accordingly.
                lons_m180to180 = utils.longitude_to_m180to180(region_subset_data_array["lon"].values)
                region_subset_data_array["lon"] = lons_m180to180
                region_subset_data_array = region_subset_data_array.sortby("lon") 
            else:
                if (self.LATS_FLIP_FLAG):
                    region_subset_data_array = data_array.sel(lat = slice(self.upper_lat, self.lower_lat), lon = slice(self.lower_lon, self.upper_lon))
                else:
                    region_subset_data_array = data_array.sel(lat = slice(self.lower_lat, self.upper_lat), lon = slice(self.lower_lon, self.upper_lon))

        print(f"Region subset data array shape: {region_subset_data_array.shape}")
        # For certain regions, take an extra step and mask the data to only the land/geopolitical boundaries of that region.
        # For CONUS, for example, avoids including grid points from the ocean, Mexico, and Canada in the stats (e.g., the very high precip Gulf Stream area).
        print(f"Masking data to only the geopolitical boundaries of this region (if applicable)")
        region_subset_data_array = self._create_region_mask(region_subset_data_array)

        print(f"{data_name} data array shape, subsetted to region: {region_subset_data_array.shape}")
        return region_subset_data_array 

    def _sum_data_over_full_time_period(self):
        print("Calculating data summed over full time period")
        self.da_dict_summed_time = {} 
        for data_name, da in self.da_dict.items():
            # Keep time dimension in summmed data array so we are able to operate on this
            # dimension in various plotting method and related helper methods.
            # Also set min_count to the shape of the DataArray's time dimension, which requires
            # that ALL data along this dimension be non-NaN to take the sum; otherwise, NaN is returned
            da_summed_time = da.sum(dim = utils.period_end_time_dim_str,
                                    keepdims = True,
                                    skipna = True,
                                    min_count = da.coords[utils.period_end_time_dim_str].shape[0])
            end_dt = pd.Timestamp(da[utils.period_end_time_dim_str].values[-1])
            da_summed_time.coords[utils.period_end_time_dim_str] = [end_dt]
 
            # Set attributes properly
            num_time_intervals = da[utils.period_end_time_dim_str].shape[0]
            total_time_period_hours = num_time_intervals * self.temporal_res
            pdp.add_attributes_to_data_array(da_summed_time,
                                               short_name = f"{total_time_period_hours}-hour precipitation", 
                                               long_name = f"Precipitation accumulated over the prior {total_time_period_hours} hour(s)",
                                               units = da.units,
                                               interval_hours = total_time_period_hours) 
            self.da_dict_summed_time[data_name] = da_summed_time 

    ##### PUBLIC METHODS #####
    ############################################################################
    ##### Public methods stats calculations #####
    # Calculate occurrence statistics over entire forecast and obs datasets (i.e., the stats will be derived
    # from grids that are valid in time and space).
    def calculate_occ_stats(self, input_da_dict = None,
                            threshold_list = utils.default_eval_threshold_list_mm): 
        if (input_da_dict is None):
            input_da_dict = self.da_dict

        threshold_da = xr.DataArray(threshold_list)
        pdp.add_attributes_to_data_array(threshold_da, units = "mm")
        data_coords = threshold_da 
 
        occ_stats_dict = {}
        obs_precip = input_da_dict[self.truth_data_name]
        for data_name, da in input_da_dict.items():
            if (data_name == self.truth_data_name):
                continue
            model_precip = input_da_dict[data_name] 
            print(f"Calculating occurence statistics for dataset {data_name}")

            # Calculate FSS for varying evaluation radius, fixed threshold
            hits_list = []
            misses_list = []
            false_alarms_list = []
            correct_negatives_list = []
            total_events_list = []
            frequency_bias_list = []
            CSI_list = []
            ETS_list = []
            for threshold in threshold_list:
                hits = self.calculate_hits(threshold, model_precip, obs_precip)
                hits_list.append(hits)

                misses = self.calculate_misses(threshold, model_precip, obs_precip)
                misses_list.append(misses)
                
                false_alarms = self.calculate_false_alarms(threshold, model_precip, obs_precip)
                false_alarms_list.append(false_alarms)

                correct_negatives = self.calculate_correct_negatives(threshold, model_precip, obs_precip)
                correct_negatives_list.append(correct_negatives)

                total_events = hits + misses + false_alarms + correct_negatives
                total_events_list.append(total_events)

                # Frequency bias
                # Measures the ratio of the frequency of forecast events to the frequency of observed events
                # See https://www.cawcr.gov.au/projects/verification/verif_web_page.html#Methods_for_dichotomous_forecasts
                if (hits + misses > 0):
                    frequency_bias = (hits + false_alarms)/(hits + misses)
                else:
                    frequency_bias = np.nan 
                frequency_bias_list.append(frequency_bias)

                # CSI (Critical Success Index) AKA TS (Threat Score)
                # Measures the fraction of observed and/or forecast events that were correctly predicted
                # See https://www.cawcr.gov.au/projects/verification/verif_web_page.html#Methods_for_dichotomous_forecasts
                if (hits + misses + false_alarms > 0):
                    CSI = hits/(hits + misses + false_alarms)
                else:
                    CSI = np.nan
                CSI_list.append(CSI)

                # ETS (Equitable Threat Score) AKA Gilbert Skill Score
                # Measures the fraction of observed and/or forecast events that were correctly predicted, adjusted for hits associated with random chance
                # https://www.cawcr.gov.au/projects/verification/verif_web_page.html#Methods_for_dichotomous_forecasts
                hits_random = (hits + misses) * (hits + false_alarms) / total_events
                if (hits + misses + false_alarms - hits_random > 0): 
                    ETS = (hits - hits_random)/(hits + misses + false_alarms - hits_random)
                else:
                    ETS = np.nan
                ETS_list.append(ETS)
    
            hits_da = self._convert_occ_stats_np_array_to_data_array(np.array(hits_list), data_coords, "hits")
            misses_da = self._convert_occ_stats_np_array_to_data_array(np.array(misses_list), data_coords, "misses")   
            false_alarms_da = self._convert_occ_stats_np_array_to_data_array(np.array(false_alarms_list), data_coords, "false_alarms")
            correct_negatives_da = self._convert_occ_stats_np_array_to_data_array(np.array(correct_negatives_list), data_coords, "correct_negatives")
            total_events_da = self._convert_occ_stats_np_array_to_data_array(np.array(total_events_list), data_coords, "total_events")
            frequency_bias_da = self._convert_occ_stats_np_array_to_data_array(np.array(frequency_bias_list), data_coords, "frequency_bias")
            CSI_da = self._convert_occ_stats_np_array_to_data_array(np.array(CSI_list), data_coords, "CSI")
            ETS_da = self._convert_occ_stats_np_array_to_data_array(np.array(ETS_list), data_coords, "ETS")

            occ_stats_dict[data_name] = StatsDataClass(
                                                       threshold = threshold_da,
                                                       hits = hits_da,
                                                       misses = misses_da,
                                                       false_alarms = false_alarms_da,
                                                       correct_negatives = correct_negatives_da,
                                                       total_events = total_events_da,
                                                       frequency_bias = frequency_bias_da,
                                                       CSI = CSI_da,
                                                       ETS = ETS_da
                                                      )

        return occ_stats_dict 

    def extract_occ_stat(self, occ_stats_dict, which_stat):
        stat_dict = {}
        for data_name, stats_data in occ_stats_dict.items():
            match which_stat:
                case "hits":
                    stat_dict[data_name] = stats_data.hits
                case "misses":
                    stat_dict[data_name] = stats_data.misses
                case "false_alarms":
                    stat_dict[data_name] = stats_data.false_alarms
                case "correct_negatives":
                    stat_dict[data_name] = stats_data.correct_negatives
                case "total_events":
                    stat_dict[data_name] = stats_data.total_events
                case "frequency_bias":
                    stat_dict[data_name] = stats_data.frequency_bias
                case "CSI":
                    stat_dict[data_name] = stats_data.CSI
                case "ETS":
                    stat_dict[data_name] = stats_data.ETS
                case _:
                    print(f"Error: Unrecognized occurence stat type {which_stat}")
                    return

        return stat_dict

    # Calculate correlation coefficient
    def calculate_pearsonr(self, model_precip, obs_precip):
        model_precip_values_flat = model_precip.values.flatten()
        obs_precip_values_flat = obs_precip.values.flatten()
        model_precip_no_nans = model_precip_values_flat[~np.isnan(model_precip_values_flat) & ~np.isnan(obs_precip_values_flat)]
        obs_precip_no_nans = obs_precip_values_flat[~np.isnan(model_precip_values_flat) & ~np.isnan(obs_precip_values_flat)]
        pearsonr = scipy.stats.pearsonr(model_precip_no_nans, obs_precip_no_nans)
        return pearsonr

    # Calculate RMSE
    def calculate_rmse(self, model_precip, obs_precip):
        squared_errors = (model_precip - obs_precip)**2
        return np.sqrt(squared_errors.mean().load()).item()

    # Calculate mean amount bias
    def calculate_bias(self, model_precip, obs_precip):
        return (model_precip - obs_precip).mean().item()
    
    # Calculate hits
    def calculate_hits(self, threshold, model_precip, obs_precip):
        hits_condition = (model_precip >= threshold) & (obs_precip >= threshold)
        number_of_hits = len(np.where(hits_condition.values.flatten())[0])
        return number_of_hits

    # Calculate misses
    def calculate_misses(self, threshold, model_precip, obs_precip):
        misses_condition = (model_precip < threshold) & (obs_precip >= threshold)
        number_of_misses = len(np.where(misses_condition.values.flatten())[0]) 
        return number_of_misses

    # Calculate false alarms
    def calculate_false_alarms(self, threshold, model_precip, obs_precip):
        false_alarms_condition = (model_precip >= threshold) & (obs_precip < threshold)
        number_of_false_alarms = len(np.where(false_alarms_condition.values.flatten())[0]) 
        return number_of_false_alarms

    # Calculate correct negatives
    def calculate_correct_negatives(self, threshold, model_precip, obs_precip):
        correct_negatives_condition = (model_precip < threshold) & (obs_precip < threshold)
        number_of_correct_negatives = len(np.where(correct_negatives_condition.values.flatten())[0])
        return number_of_correct_negatives

    # Calculate statistics valid for data aggregated in space, time, or space and time. Currently
    # only mean and percentile stats are supported. 
    def calculate_aggregated_stats(self,
                                   input_da_dict = None,
                                   time_period_type = "monthly",
                                   agg_type = "space_time",         
                                   stat_type = "mean",
                                   pctl = 99,
                                   write_to_nc = False):
        if (stat_type == "pctl"):
            print(f"Calculating {agg_type}-aggregated {time_period_type} {pctl:0.1f}th {stat_type}s")
        else:
            print(f"Calculating {agg_type}-aggregated {time_period_type} {stat_type}s")

        if input_da_dict is None:
            input_da_dict = self.da_dict
        
        # Process time_period_type: list of date times, dimension name, etc.
        dtimes, dim_name, time_period_str = self._process_time_period_type_to_dtimes(time_period_type)

        # Process agg_type: determine which dimension(s) to aggregate over
        match agg_type:
            case "space": # Is this even needed? It would be (for example), a spatial mean at each valid time
                agg_dims = ("lat", "lon")
                raise NotImplementedError
            case "time":
                agg_dims = (utils.period_begin_time_dim_str)
            case "space_time":
                agg_dims = (utils.period_begin_time_dim_str, "lat", "lon")

        # Process stat_type: eventual attributes of aggregated data arrays
        short_name = self.truth_da.short_name
        long_name = self.truth_da.long_name
        time_period_type_str = ""
        if (time_period_type is not None):
            time_period_type_str = f"{time_period_type} "
        match stat_type:
            case "mean":
                short_name = short_name + f" {time_period_type_str}{stat_type}"
                long_name = f"{time_period_type_str.title()}{stat_type} of " + long_name 
            case "pctl":
                short_name = short_name + f" {time_period_type_str}{pctl:0.1f}th {stat_type}"
                long_name = f"{pctl:0.1f}th {stat_type} of " + long_name
            case _:
                print(f"Error: Unrecognized stat type {stat_type}")
                return

        agg_data_dict = {}
        for data_name, da in input_da_dict.items():
            # Convert data coordinates to period beginning (easier to aggregate over different time periods this way)
            if (time_period_type is not None):
                data_array = self._convert_period_end_to_period_begin(da)
            else:
                data_array = da

            # If aggregating seasonally, dtimes, which in this case will be a list of lists defining
            # seasonal datetime ranges, wasn't defined above.
            if (time_period_type == "seasonal"):
                dtimes = self._construct_season_dt_ranges(data_array)

            # Calculate spatiotemporal means across regions and months
            data_list = [] 
            for dtime in dtimes:
                data_to_aggregate = self._determine_agg_data_from_time_period_type(data_array, time_period_type, dtime)

                match stat_type:
                    case "mean":
                        data = data_to_aggregate.mean(dim = agg_dims)
                    case "pctl":
                        data = data_to_aggregate.quantile(pctl/100, dim = agg_dims)
                data_list.append(data)

            # Convert data to xarray DataArray via xr.concat
            agg_da = xr.concat(data_list, dim = dim_name)
            if (time_period_type == "seasonal"):
                agg_da.coords[dim_name] = self._construct_seasonal_dt_str_list(dtimes) 
            else:
                agg_da.coords[dim_name] = dtimes 
            pdp.add_attributes_to_data_array(agg_da,
                                               short_name = short_name, 
                                               long_name = long_name, 
                                               units = da.units)
            agg_data_dict[data_name] = agg_da 

            # Output to netCDF
            if write_to_nc:
                match stat_type:
                    case "mean":
                        stat_type_out_str = stat_type
                    case "pctl":
                        stat_type_out_str = f"{pctl:0.1f}th_{stat_type}"

                self._set_output_var_name(agg_da) 
                nc_out_fpath = self._configure_output_stats_nc_fpath(data_name, time_period_str, time_period_type = time_period_type,
                                                                     stat_type = stat_type_out_str, agg_type = agg_type)
                print(f"Writing {nc_out_fpath}")
                agg_da.to_netcdf(nc_out_fpath)

        return agg_data_dict

    # Calculate probability density functions (PDFs)
    def calculate_pdf(self, input_da_dict = None, time_period_type = "full_period", write_to_nc = False):
        if input_da_dict is None:
            input_da_dict = self.da_dict

        # Process time_period_type: list of date times, dimension name, etc.
        dtimes, dim_name, time_period_str = self._process_time_period_type_to_dtimes(time_period_type)

        # If aggregating seasonally, dtimes, which in this case will be a list of lists defining
        # seasonal datetime ranges, wasn't defined above.
        if (time_period_type == "seasonal"):
            dtimes = self._construct_season_dt_ranges(data_array)

        pdf_data_dict = {}
        for dtime in dtimes:
            pdf_each_dtime_dict = {}
            for data_name, da in input_da_dict.items():
                # Convert data coordinates to period beginning (much easier to aggregate over months that way)
                data_array = self._convert_period_end_to_period_begin(da)

                # Calculate pdf 
                data_to_aggregate = self._determine_agg_data_from_time_period_type(data_array, time_period_type, dtime)

                hist_and_bins = data_to_aggregate.plot.hist()
                hist = hist_and_bins[0]
                bins = hist_and_bins[1]
                total_samples = hist.sum()
                pdf = hist/total_samples

                # Including total samples in order to back out original histogram from PDF
                pdf_each_dtime_dict[data_name] = (pdf, bins, total_samples) 
                    
            pdf_data_dict[dtime] = pdf_each_dtime_dict

        # Output to netCDF
        if write_to_nc:
            num_dtimes = len(dtimes)
            # Loop through each data_name ('AORC', etc.), writing pdf data to a data_name-specific netCDF file
            for data_name in self.data_names:
                # Here, just initialize the _full_array variables as empty arrays with size 0
                bins_full_array = np.empty(0)
                probs_full_array = np.empty(0)
                # Loop through each dtime, concatenating respective pdf and bins data 
                total_samples_list = []
                for dtime in dtimes: 
                    total_samples = pdf_data_dict[dtime][data_name][2]
                    total_samples_list.append(total_samples) 
                    bins = np.expand_dims(pdf_data_dict[dtime][data_name][1], 0)
                    probs = np.expand_dims(pdf_data_dict[dtime][data_name][0], 0)
                    if (bins_full_array.size == 0):
                        bins_full_array = bins
                        probs_full_array = probs
                    else:
                        bins_full_array = np.concatenate( (bins_full_array, bins), axis = 0) 
                        probs_full_array = np.concatenate( (probs_full_array, probs), axis = 0)

                # Convert full arrays of bins and pdf data to DataArrays and subsequently unite in a Dataset 
                bins_dim_coords = np.arange(bins_full_array.shape[1])
                bins_da = xr.DataArray(bins_full_array, dims = [dim_name, "bins_dim"], coords = [dtimes, bins_dim_coords])
                probs_dim_coords = np.arange(probs_full_array.shape[1])
                probs_da = xr.DataArray(probs_full_array, dims = [dim_name, "probs_dim"], coords = [dtimes, probs_dim_coords])
                total_samples_da = xr.DataArray(total_samples_list, dims = [dim_name], coords = [dtimes])
                pdf_ds = xr.Dataset({"bins": bins_da, "probs": probs_da, "total_samples": total_samples_da}) 

                # Output Dataset to netCDF
                nc_out_fpath = self._configure_output_stats_nc_fpath(data_name, time_period_str, time_period_type = time_period_type, stat_type = "pdf")
                print(f"Writing {nc_out_fpath}")
                pdf_ds.to_netcdf(nc_out_fpath)

        return pdf_data_dict 

    # Calculate FSS for all QPF datasets, for all valid times. Output FSS at each
    # valid time to a dictionary of DataArrays, so this dict can subsequently be
    # handled similarly to self.da_dict. The dimensions are (num_valid_times * num_eval_radii [num_thresholds]). 
    def calculate_fss(self, eval_type = evaluate_by_radius_kw_str, 
                      grid_cell_size = 0.25, # in degrees lat/lon 
                      fixed_radius = 0.5, # in degrees lat/lon
                      fixed_threshold = 10.0, # in mm 
                      fixed_ari_threshold = 2, # in years
                      eval_radius_list = utils.default_eval_radius_list_deg, 
                      eval_threshold_list = utils.default_eval_threshold_list_mm,
                      eval_ari_list = utils.default_eval_ari_list_years,
                      time_period_type = "full_period",
                      radius_units = "deg", # For degrees lat/lon; otherwise km, etc.
                      is_pctl_threshold = False,
                      include_zeros = False,
                      write_to_nc = False):
        # Process time_period_type: list of date times, dimension name, etc.
        dtimes, dim_name, time_period_str = self._process_time_period_type_to_dtimes(time_period_type)
        self.fss_eval_radius_units = radius_units

        if is_pctl_threshold:
            threshold_units = "th_pctl"
        else:
            threshold_units = "mm"

        if (eval_type == evaluate_by_radius_kw_str):
            print(f"**** Calculating FSS by radius (fixed threshold {fixed_threshold}{threshold_units})")
            self.fixed_fss_eval_threshold = fixed_threshold
            
            self.fss_eval_radius_da = xr.DataArray(eval_radius_list)
            pdp.add_attributes_to_data_array(self.fss_eval_radius_da, units = self.fss_eval_radius_units)
            fss_data_coords = self.fss_eval_radius_da 

            fss_data_dim_name = "radius"
            stat_type = f"fss.by_radius.thresh{self.fixed_fss_eval_threshold:0.1f}{threshold_units}"
        elif (eval_type == evaluate_by_radius_ari_threshold_kw_str):
            print(f"**** Calculating FSS by radius, using {fixed_ari_threshold:02d}-year ARI grid as threshold")
            self.fixed_fss_eval_threshold = fixed_ari_threshold

            self.fss_eval_radius_da = xr.DataArray(eval_radius_list)
            pdp.add_attributes_to_data_array(self.fss_eval_radius_da, units = self.fss_eval_radius_units)
            fss_data_coords = self.fss_eval_radius_da 
            
            fss_data_dim_name = "radius"
            stat_type = f"fss.by_radius.thresh{self.fixed_fss_eval_threshold:02d}_year_ari"

            self.fixed_ari_grid = self._open_ari_threshold_grid(self.fixed_fss_eval_threshold)
        elif (eval_type == evaluate_by_ari_kw_str):
            print(f"**** Calculating FSS by ARI grid threshold (fixed eval radius {fixed_radius} {self.fss_eval_radius_units})")
            self.fixed_fss_eval_radius = fixed_radius

            self.fss_eval_ari_da = xr.DataArray(eval_ari_list)
            pdp.add_attributes_to_data_array(self.fss_eval_ari_da, units = "years")
            fss_data_coords = self.fss_eval_ari_da

            fss_data_dim_name = "ARI"
            stat_type = f"fss.by_ari_threshold.radius{self.fixed_fss_eval_radius:0.1f}{self.fss_eval_radius_units}"

            # Read in ARI grids that will be used as thresholds for FSS calculations
            self.ari_grid_dict = {}
            for ari in eval_ari_list: 
                self.ari_grid_dict[ari] = self._open_ari_threshold_grid(ari)
        else: # If anything else is passed for <eval_type>, evaluate against threshold
            print(f"**** Calculating FSS by threshold (fixed eval radius {fixed_radius} {self.fss_eval_radius_units})")
            self.fixed_fss_eval_radius = fixed_radius

            self.fss_eval_threshold_da = xr.DataArray(eval_threshold_list)
            pdp.add_attributes_to_data_array(self.fss_eval_threshold_da, units = threshold_units)
            fss_data_coords = self.fss_eval_threshold_da 

            fss_data_dim_name = "threshold"
            stat_type = f"fss.by_threshold.radius{self.fixed_fss_eval_radius:0.1f}{self.fss_eval_radius_units}"

        da_dict_fss = {}
        da_dict_f_model = {}
        for data_name, da in self.da_dict.items():
            print(f"Calculating FSS for dataset {data_name}")
            f_obs_list = []
            f_model_list = []
            for v, valid_dt in enumerate(self.valid_dt_list):
                valid_dt_str = f"{valid_dt:%Y-%m-%d %H:%M:%S}"
                qpe = self.truth_da.sel(period_end_time = valid_dt_str)

                # For the observations grid (truth dataset), calculate the F_obs values only 
                # (fractions of observed grid exceeding threshold)
                if (data_name == self.truth_data_name): 
                    if (eval_type == evaluate_by_radius_kw_str):
                        if is_pctl_threshold:
                            if include_zeros:
                                threshold_for_F_obs_calc = qpe.quantile(fixed_threshold/100.0)
                            else:
                                threshold_for_F_obs_calc = qpe.where(qpe > 0.0).quantile(fixed_threshold/100.0)
                        else:
                            threshold_for_F_obs_calc = self.fixed_fss_eval_threshold
                        binary_qpe = self._mask_data_array_based_on_threshold(qpe, threshold_for_F_obs_calc) 
                        F_obs = np.where(binary_qpe)[0].shape[0]/binary_qpe.flatten().shape[0]
                        f_obs_list.append(F_obs)
                    elif (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                        binary_qpe = self._mask_data_array_based_on_threshold(qpe, self.fixed_ari_grid)
                        F_obs = np.where(binary_qpe)[0].shape[0]/binary_qpe.flatten().shape[0]
                        f_obs_list.append(F_obs)
                    continue

                # Calculate FSS
                qpf = da.sel(period_end_time = valid_dt_str)
                fss_list = []
                if (eval_type == evaluate_by_radius_kw_str):
                    for radius in eval_radius_list:
                        FSS = self._calculate_fss_single_grid(qpf, qpe, radius, grid_cell_size, fixed_threshold,
                                                              is_pctl_threshold = is_pctl_threshold, include_zeros = include_zeros)
                        fss_list.append(FSS)
                elif (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                    for radius in eval_radius_list:   
                        FSS = self._calculate_fss_single_grid(qpf, qpe, radius, grid_cell_size, self.fixed_ari_grid)
                        fss_list.append(FSS)
                elif (eval_type == evaluate_by_ari_kw_str):
                    for ari in eval_ari_list:
                        FSS = self._calculate_fss_single_grid(qpf, qpe, fixed_radius, grid_cell_size, self.ari_grid_dict[ari])
                        fss_list.append(FSS)
                else:
                    for threshold in eval_threshold_list:
                        FSS = self._calculate_fss_single_grid(qpf, qpe, fixed_radius, grid_cell_size, threshold,
                                                              is_pctl_threshold = is_pctl_threshold, include_zeros = include_zeros)
                        fss_list.append(FSS)
    
                fss_tmp_array = np.array(fss_list).reshape((1, fss_data_coords.shape[0]))
                if (v == 0):
                    fss_array = np.copy(fss_tmp_array) 
                else:
                    fss_array = np.concatenate((fss_array, fss_tmp_array), axis = 0)

                # Calculate frequency-related values based on the binary qpf and qpe fields
                # F_obs = fraction of observed points exceeding threshold over whole domain
                # F_model = fraction of forecast/model points exceeding threshold over whole domain
                # These values in turn are used to calculate AFSS, FSS_uniform, etc
                # Only do this for by_radius evaluations, since that's the context in which it  is most useful. 
                if (eval_type == evaluate_by_radius_kw_str):
                    if is_pctl_threshold:
                        if include_zeros:
                            threshold_for_F_model_calc = qpf.quantile(fixed_threshold/100.0)
                        else:
                            threshold_for_F_model_calc = qpf.where(qpf > 0.0).quantile(fixed_threshold/100.0)
                    else:
                        threshold_for_F_model_calc = self.fixed_fss_eval_threshold
                    binary_qpf = self._mask_data_array_based_on_threshold(qpf, threshold_for_F_model_calc)
                    F_model = np.where(binary_qpf)[0].shape[0]/binary_qpf.flatten().shape[0]
                    f_model_list.append(F_model)
                elif (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                    binary_qpf = self._mask_data_array_based_on_threshold(qpf, self.fixed_ari_grid)
                    F_model = np.where(binary_qpf)[0].shape[0]/binary_qpf.flatten().shape[0]
                    f_model_list.append(F_model)

            # Convert numpy arrays to DataArrays
            if (data_name == self.truth_data_name):
                if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                    self.f_obs_da = xr.DataArray(f_obs_list, coords = [self.valid_dt_list], dims = [utils.period_end_time_dim_str])
                    self.f_obs_da.name = "observed_fractions"
                continue

            fss_da = xr.DataArray(fss_array, coords = [self.valid_dt_list, fss_data_coords], dims = [utils.period_end_time_dim_str, fss_data_dim_name])
            fss_da.name = "fss" 
            da_dict_fss[data_name] = fss_da
            
            if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                f_model_da = xr.DataArray(f_model_list, coords = [self.valid_dt_list], dims = [utils.period_end_time_dim_str])
                f_model_da.name = "forecast_fractions"
                da_dict_f_model[data_name] = f_model_da

            # Output to netCDF
            if write_to_nc:
                nc_out_fpath = self._configure_output_stats_nc_fpath(data_name, time_period_str, time_period_type = time_period_type, stat_type = stat_type)
                print(f"Writing {nc_out_fpath}")
                fss_da.to_netcdf(nc_out_fpath)

        # Here I decided to add the FSS dictionary as an attribute of the class
        # This will make it easier to access and manipulate the data in the dictionary in other methods.
        if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
            self.fss_dict_by_radius = da_dict_fss
            self.f_model_dict = da_dict_f_model
        elif (eval_type == evaluate_by_ari_kw_str):
            self.fss_dict_by_ari = da_dict_fss
        else:
            self.fss_dict_by_threshold = da_dict_fss

        return da_dict_fss

    def calculate_aggregated_fss(self, external_fss_dict = None, eval_type = evaluate_by_radius_kw_str, time_period_type = "full_period"):
        if (external_fss_dict is not None):
            if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                self.fss_eval_radius_da = external_fss_dict[self.data_names[-1]].radius
                self.fss_dict_by_radius = external_fss_dict
                da_dict_fss = self.fss_dict_by_radius
            elif (eval_type == evaluate_by_ari_kw_str):
                self.fss_eval_ari_da = external_fss_dict[self.data_names[-1]].ARI
                da_dict_fss = self.fss_dict_by_ari
            else:
                self.fss_eval_threshold_da = external_fss_dict[self.data_names[-1]].threshold
                self.fss_dict_by_threshold = external_fss_dict
                da_dict_fss = self.fss_dict_by_threshold 
        else:
            if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                if not(hasattr(self, "fss_dict_by_radius")):
                    print("Error: You need to calculate FSS over varying evaluation radii first to perform aggregation")
                    return
                da_dict_fss = self.fss_dict_by_radius
            elif (eval_type == evaluate_by_ari_kw_str):
                if not(hasattr(self, "fss_dict_by_ari")):
                    print("Error: You need to calculate FSS over varying ARI grids first to perform aggregation")
                    return
                da_dict_fss = self.fss_dict_by_ari
            else:
                if not(hasattr(self, "fss_dict_by_threshold")):
                    print("Error: You need to calculate FSS over varying thresholds first to perform aggregation")
                    return
                da_dict_fss = self.fss_dict_by_threshold
        
        # Process time_period_type: list of date times, dimension name, etc.
        dtimes, dim_name, time_period_str = self._process_time_period_type_to_dtimes(time_period_type)

        # If aggregating seasonally, dtimes, which in this case will be a list of lists defining
        # seasonal datetime ranges, wasn't defined above.
        if (time_period_type == "seasonal"):
            dtimes = self._construct_season_dt_ranges(data_array)
        
        fss_agg_dict = {}
        afss_agg_dict = {}
        fss_uniform_agg_dict = {}
        for dtime in dtimes:
            da_dict_fss_each_dtime = {}
            da_dict_afss_each_dtime = {}
            da_dict_fss_uniform_each_dtime = {}
            for data_name, data_array in da_dict_fss.items():
                if (data_name == self.truth_data_name):
                    continue

                # Convert data coordinates to period beginning (much easier to aggregate over months that way)
                data_array = self._convert_period_end_to_period_begin(data_array)
                data_to_aggregate = self._determine_agg_data_from_time_period_type(data_array, time_period_type, dtime)
                data = data_to_aggregate.mean(dim = utils.period_begin_time_dim_str)

                da_dict_fss_each_dtime[data_name] = data

                # Calculate AFSS and FSS_uniform (useful for plotting of by-radius aggregated FSS results)
                if (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                    f_obs_da = self._convert_period_end_to_period_begin(self.f_obs_da)
                    f_model_da = self._convert_period_end_to_period_begin(self.f_model_dict[data_name])
                    
                    afss_da = (2 * f_obs_da * f_model_da)/(f_obs_da**2 + f_model_da**2)
                    fss_uniform_da = 0.5 * (1 + f_obs_da)

                    afss_data_to_aggregate = self._determine_agg_data_from_time_period_type(afss_da, time_period_type, dtime)
                    fss_uniform_data_to_aggregate = self._determine_agg_data_from_time_period_type(fss_uniform_da, time_period_type, dtime)

                    afss_data = afss_data_to_aggregate.mean().item()
                    fss_uniform_data = fss_uniform_data_to_aggregate.mean().item()

                    da_dict_afss_each_dtime[data_name] = afss_data
                    da_dict_fss_uniform_each_dtime[data_name] = fss_uniform_data

            fss_agg_dict[dtime] = da_dict_fss_each_dtime
            if  (eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str):
                afss_agg_dict[dtime] = da_dict_afss_each_dtime
                fss_uniform_agg_dict[dtime] = da_dict_fss_uniform_each_dtime

        return fss_agg_dict, afss_agg_dict, fss_uniform_agg_dict 

    # Calculated occurence statistics aggregated over specified time periods (common monthly, common seasonal, etc.)
    def calculate_aggregated_occ_stats(self, which_stat, time_period_type = "full_period"):
        # Process time_period_type: list of date times, dimension name, etc.
        dtimes, dim_name, time_period_str = self._process_time_period_type_to_dtimes(time_period_type)

        # If aggregating seasonally, dtimes, which in this case will be a list of lists defining
        # seasonal datetime ranges, wasn't defined above.
        if (time_period_type == "seasonal"):
            dtimes = self._construct_season_dt_ranges(data_array)
        
        occ_stats_agg_dict = {}
        for dtime in dtimes:
            da_dict_each_dtime = {}
            for data_name, data_array in self.da_dict.items(): 
                # Convert data coordinates to period beginning (much easier to aggregate over months that way)
                data_array = self._convert_period_end_to_period_begin(data_array)
                da_dict_each_dtime[data_name] = self._determine_agg_data_from_time_period_type(data_array, time_period_type, dtime)
                
            all_occ_stats_dict = self.calculate_occ_stats(da_dict_each_dtime)
            stat_dict = self.extract_occ_stat(all_occ_stats_dict, which_stat)
            occ_stats_agg_dict[dtime] = stat_dict 

        return occ_stats_agg_dict 
    ##### END Public methods stats calculations #####

    ##### Private methods to support stats calculations #####
    # Based on time_period_type ('monthly', etc.), determine list of dtimes, dimension names, and time period string
    def _process_time_period_type_to_dtimes(self, time_period_type):
        match time_period_type:
            case None: # Don't do any temporal aggregation
                dtimes = self.valid_dt_list
                dim_name = utils.period_begin_time_dim_str
                time_period_str = self.daily_time_period_str
            case "daily":
                dtimes = self.valid_daily_dt_list_period_begin # Need period beginning times for daily data
                dim_name = utils.days_dim_str 
                time_period_str = self.daily_time_period_str_period_begin
            case "monthly":
                dtimes = self.valid_monthly_dt_list
                dim_name = utils.months_dim_str 
                time_period_str = self.monthly_time_period_str
            case "seasonal":
                dtimes = []  # dtimes list defined in methods that use this private method 
                dim_name = utils.seasons_dim_str
                time_period_str = self.monthly_time_period_str 
            case "annual":
                dtimes = self.valid_annual_dt_list
                dim_name = utils.annual_dim_str 
                time_period_str = self.annual_time_period_str
            case "full_period":
                dtimes = [ self.monthly_time_period_str ] 
                dim_name = utils.full_period_dim_str
                time_period_str = self.monthly_time_period_str
            case "common_monthly":
                dtimes = pputils.construct_monthly_string_list()
                dim_name = utils.common_month_dim_str 
                time_period_str = self.monthly_time_period_str 
            case "common_seasonal":
                dtimes = pputils.construct_seasonal_string_list() 
                dim_name = utils.common_season_dim_str
                time_period_str = self.monthly_time_period_str 
            case _:
                print(f"Error: Unrecognized time period type {time_period_type}")
                return

        return dtimes, dim_name, time_period_str

    def _convert_occ_stats_np_array_to_data_array(self, np_array, data_coords, stat_name):
        # Convert numpy array to DataArray
        da = xr.DataArray(np_array, coords = [data_coords], dims = ["threshold"])
        da.name = stat_name 

        return da

    # Convert period-ending time dimension to period beginning; useful for certain statistics
    # to ensure that a period such as January 31 (ending at 00z February 1) is included within January stats.
    def _convert_period_end_to_period_begin(self, data_array):
        #print("Converting time coordinates to period beginning")
        period_end_times = [pd.Timestamp(i) for i in data_array.period_end_time.values]
        period_begin_times = [i - pd.Timedelta(hours = self.temporal_res) for i in period_end_times]
       
        data_array_period_begin = data_array.rename({utils.period_end_time_dim_str: utils.period_begin_time_dim_str})
        data_array_period_begin.coords[utils.period_begin_time_dim_str] = period_begin_times 

        return data_array_period_begin

    def _convert_period_begin_to_period_end(self, data_array):
        pass 

    # Returns a list of date-time ranges corresponding to each season valid within
    # the time dimension of the input data array.
    def _construct_season_dt_ranges(self, da):
        _, time_dim, _ = self._create_datetime_list_from_da_time_dim(da)

        all_dtimes = [pd.Timestamp(i) for i in da[time_dim].values]
        current_dt =  all_dtimes[0]
        
        season_dt_ranges = []
        while (utils.datetime2unix(current_dt) <= utils.datetime2unix(all_dtimes[-1])):
            if (current_dt.month < 3):
                start_dt_next_season = pd.Timestamp(current_dt.year, 3, 1)
            elif (current_dt.month < 6):
                start_dt_next_season = pd.Timestamp(current_dt.year, 6, 1)
            elif (current_dt.month < 9):
                start_dt_next_season = pd.Timestamp(current_dt.year, 9, 1)
            elif (current_dt.month < 12):
                start_dt_next_season = pd.Timestamp(current_dt.year, 12, 1)
            else: # December case: next season starts in March of NEXT year
                start_dt_next_season = pd.Timestamp(current_dt.year + 1, 3, 1)

            # Since slicing will be inclusive, the dt range should end on the final day of the current season
            season_dt_range = [current_dt, start_dt_next_season - pd.Timedelta(days = 1)]
            season_dt_ranges.append(season_dt_range)
            current_dt = start_dt_next_season

        return season_dt_ranges

    # Based on the time period type (e.g., 'monthly', 'common seasonal', etc.), and a given dtime,
    # filter a DataArray to only data within the current dtime, to be subsequently aggregated.
    def _determine_agg_data_from_time_period_type(self, data_array, time_period_type, dtime):
        match time_period_type:
            case None:
                dtime_str = f"{dtime:%Y-%m-%d %H:%M:%S}"
                data_to_aggregate = data_array.sel(period_end_time = dtime_str)
                # If working with daily (as opposed to sub-daily) data, the .sel call above will remove the time dimension. Use expand_dims to add it back.
                if (len(data_to_aggregate.shape) == 2):
                    data_to_aggregate = data_to_aggregate.expand_dims(dim = {utils.period_begin_time_dim_str: [dtime]})
            case "daily":
                dtime_str = f"{dtime:%Y-%m-%d}"
                data_to_aggregate = data_array.sel(period_begin_time = dtime_str)
                if (len(data_to_aggregate.shape) == 2):
                    data_to_aggregate = data_to_aggregate.expand_dims(dim = {utils.period_begin_time_dim_str: [dtime]})
            case "monthly":
                dtime_str = f"{dtime:%Y-%m}"
                data_to_aggregate = data_array.sel(period_begin_time = dtime_str)
            case "seasonal":
                data_to_aggregate = data_array.sel(period_begin_time = slice(dtime[0], dtime[1]))
            case "annual":
                dtime_str = f"{dtime:%Y}"
                data_to_aggregate = data_array.sel(period_begin_time = dtime_str)
            case "full_period":
                data_to_aggregate = data_array
            case "common_monthly":
                data_to_aggregate = self._select_data_by_common_time_period(data_array, dtime)
            case "common_seasonal":
                data_to_aggregate = self._select_data_by_common_time_period(data_array, dtime)
            case _:
                raise NotImplementedError

        return data_to_aggregate

    # Select data from a data array for the same month or same season (e.g., all January's) 
    def _select_data_by_common_time_period(self, data_array, time_period):
        if (type(time_period) is str):
            match time_period.lower()[:3]:
                case "jan":
                    month_list = [1]
                case "feb":
                    month_list = [2]
                case "mar":
                    month_list = [3]
                case "apr":
                    month_list = [4]
                case "may":
                    month_list = [5]
                case "jun":
                    month_list = [6]
                case "jul":
                    month_list = [7]
                case "aug":
                    month_list = [8]
                case "sep":
                    month_list = [9]
                case "oct":
                    month_list = [10]
                case "nov":
                    month_list = [11]
                case "dec":
                    month_list = [12]
                case "djf":
                    month_list = [12,1,2]
                case "mam":
                    month_list = [3,4,5]
                case "jja":
                    month_list = [6,7,8]
                case "son":
                    month_list = [9,10,11]
                case _:
                    print(f"Time period string is {time_period}, not a valid month or season; not selecting data from data array")
                    return data_array
        else:
            print(f"Invalid time period string {time_period}; not selecting data from data array;\n"
                   "must be a string representing the name of a month, the first three letters of a month\n"
                   "(e.g., Aug for August; case insensitive), or three letters representing a season\n"
                   "(options are DJF, MAM, JJA, SON); case insensitive")
            return data_array

        # Convert time coordinates to period beginning so that accumulated precip ending at, for example, 00z Feb 1
        # is interpreted as valid *during* January.
        if (utils.period_end_time_dim_str in data_array.dims): 
            data_array = self._convert_period_end_to_period_begin(data_array)

        da_sel_time_period = data_array.sel(period_begin_time = data_array.period_begin_time.dt.month.isin(month_list))
        return da_sel_time_period

    # Construct strings of the current season and the year to which it
    # corresponds (e.g., Decemember corresponds to DJF of the following year)
    def _construct_seasonal_dt_str_list(self, dtime_ranges):
        seasonal_dt_str_list = []
        for dtime_range in dtime_ranges:
            if (dtime_range[0].month < 3): # Winter
                season_str = "DJF"
                year_str = f"{dtime_range[0]:%y}"
            elif (dtime_range[0].month < 6): # Spring 
                season_str = "MAM"
                year_str = f"{dtime_range[0]:%y}"
            elif (dtime_range[0].month < 9): # Summer
                season_str = "JJA"
                year_str = f"{dtime_range[0]:%y}"
            elif (dtime_range[0].month < 12): # Fall
                season_str = "SON"
                year_str = f"{dtime_range[0]:%y}"
            else: # December, so Winter of the NEXT year
                season_str = "DJF"
                year_str = f"{dtime_range[-1]:%y}"

            seasonal_dt_str_list.append(f"{season_str}{year_str}")
        return seasonal_dt_str_list

    # Convert DataArrays to binary (1/0) values, based on threshold; used in FSS calculation.
    # Set data array to 1 at or above threshold, zero below it
    # NOTE: this function takes an xarray DataArray as input and returns a numpy array.
    def _mask_data_array_based_on_threshold(self, da, threshold):
        # Take the arrays down to 2-D, removing the time dimension
        if (len(da.shape) == 3):
            da = da[0,:,:]

        return np.where(da >= threshold, 1, 0) 

    # Create circular footprint for FSS calculation
    # Potential FIXME: For the expected behavior to be obtained, <radius> must be
    # evenly divisible by <grid_cell_size>, e.g., radius = 0.5 degrees; grid_cell_size = 0.25 degrees.
    # Specifically, if this is not the case, the function will return a square of all 1s in some cases,
    # rather than a square of 0s circumscribing a circle of 1s. This may be OK (won't fix) because it
    # doesn't make sense to use a radius that's equivalent to a non-integer number of grid cells.
    def _get_footprint_for_fss(self, radius, grid_cell_size):
        radius_number_grid_cells = int(radius/grid_cell_size)

        # In this step, we obtain a square of zeros (with side length, in number of grid cells, of radius_number_grid_cells * 2 + 1)
        # circumscribing a circle of ones (with radius, in number of grid cells, of radius_number_grid_cells):

        # 1) Create a footprint: just an array of 1s
        footprint = (np.ones((radius_number_grid_cells * 2 + 1, radius_number_grid_cells * 2 + 1))).astype(int)

        # 2) Set the centerpoint of the array to zero (needed for the subsequent distance calculation)
        footprint[math.ceil(radius_number_grid_cells), math.ceil(radius_number_grid_cells)] = 0

        # 3) Within the footprint, calculate each point's distance from the center point
        dist = scipy.ndimage.distance_transform_edt(footprint, sampling = [grid_cell_size, grid_cell_size])

        # 4) Set the footprint to zeros where distance calculated in step 3) is greater than radius; keep it
        # set to one where distance is less than radius, obtaining the square of zeros circumscribing the circle of ones 
        return np.where(np.greater(dist, radius), 0, 1)

    # Calculate FSS for a single spatial grid (i.e., at a single valid time).
    # Code from Craig Schwartz via Trevor Alcott. See 20250130 email from Trevor
    # which is part of thread entitled "Experience with fractions skill score?"
    # FIXME (potentially): how are NaNs being handled. I think they are being converted to zeros by _mask_data_array_based_on_threshold
    # which may not be desirable. We should keep them as NaNs, but then how will that affect the FSS calculation here?
    def _calculate_fss_single_grid(self, qpf, qpe, radius, grid_cell_size, threshold,
                                   is_pctl_threshold = False, include_zeros = False):
        # Calculate footprint, i.e., evaluation area
        footprint = self._get_footprint_for_fss(radius, grid_cell_size)
       
        # Convert qpf and qpe arrays to numpy arrays containing 1s and 0s based on
        # whether precipitation amount is at or above (set to 1) or below (set to 0) <threshold>.
        if is_pctl_threshold:
            if not(include_zeros):
                threshold_amount_qpf = qpf.where(qpf > 0.0).quantile(threshold/100.0)
                threshold_amount_qpe = qpe.where(qpe > 0.0).quantile(threshold/100.0)
            else:
                threshold_amount_qpf = qpf.quantile(threshold/100.0) 
                threshold_amount_qpe = qpe.quantile(threshold/100.0)
            binary_qpf = self._mask_data_array_based_on_threshold(qpf, threshold_amount_qpf)
            binary_qpe = self._mask_data_array_based_on_threshold(qpe, threshold_amount_qpe)
        else: 
            binary_qpf = self._mask_data_array_based_on_threshold(qpf, threshold)
            binary_qpe = self._mask_data_array_based_on_threshold(qpe, threshold)

        # Calculate forecast_fractions and observed fractions terms in the FSS formula.
        # These are the M (model) and O (observed) terms calculated in Roberts and Lean (2008)
        # equations 2 and 3, and what Trevor's code refers to as pf and po, respectively.
        # CONCEPTUAL PROCEDURE:
            # For every grid point, calculate the number of points within the footprint centered on the grid point
            # for which the binary_qpf and binary_qpe arrays equal 1 (i.e., <qpf> and <qpe> are at or above <threshold>).
            # Divide by the size of the footprint [np.sum(footprint)] to convert this number of points to a spatial
            # fraction of the footprint.
        # IMPLEMENTATION using fftconvolve:
            # I don't yet understand how fftconvolve calculates the number of points equal to one in each grid
            # point's neighborhood other than to state that it uses a Fast Fourier Transform (FFT) technique. 
        forecast_fractions = np.around(scipy.signal.fftconvolve(binary_qpf, footprint, mode = "same"))/np.sum(footprint)
        observed_fractions = np.around(scipy.signal.fftconvolve(binary_qpe, footprint, mode = "same"))/np.sum(footprint)

        # Calculate gridsize (Nx * Ny)
        gridsize = np.shape(binary_qpe)[0] * np.shape(binary_qpe)[1]

        # Calculate numerator [Equation 5 in Roberts and Lean (2008)]
        # which is the mean squared error (MSE) of the forecast fractions (forecast_fractions)
        # compared to the observed fractions (observed_fractions)
        mse = 1/gridsize * np.sum((forecast_fractions - observed_fractions)**2)

        # Calculate denominator [Equation 7 in Roberts and Lean (2008)]
        # which is the mean squared error (MSE) of a low-skill reference forecast
        mse_reference = 1/gridsize * (np.sum(forecast_fractions**2) + np.sum(observed_fractions**2))

        if (mse_reference > 0):
            return 1.0 - float(mse)/float(mse_reference)
        else:
            return np.nan 
    
    def _open_ari_threshold_grid(self, ari):
        ari_nc_dir = os.path.join(self.netcdf_dir, f"ARIs.{self.data_grid_name}")
        data_name = f"ARI.{self.data_grid_name}.{ari:02d}_year.{self.temporal_res:02d}_hour_precipitation"
        ari_fname = f"{data_name}.nc"
        ari_fpath = os.path.join(ari_nc_dir, ari_fname)
        if not(os.path.exists(ari_fpath)):
            print(f"Error: ARI grid file {ari_fpath} does not exist")
            sys.exit(1)
       
        print(f"Reading ARI grid file {ari_fpath}") 
        ari_grid = xr.open_dataset(ari_fpath).precip  
        ari_grid = self._subset_data_to_region(ari_grid, data_name = data_name)

        return ari_grid

    def _set_output_var_name(self, data_array):
        if (self.temporal_res == "native"):
            output_var_name = "precipitation"
        else:
            output_var_name = f"precipitation_{self.temporal_res}_hour"
        data_array.name = output_var_name

    def _configure_output_stats_nc_fpath(self, data_name, time_period_str, time_period_type = None, stat_type = None, agg_type = None):
        if (data_name == self.data_grid):
            main_prefix = f"{data_name}.{self.native_grid_name}.{self.temporal_res:02d}_hour_precipitation"
            dir_name = f"{main_prefix}.stats"
        else:
            main_prefix = f"{data_name}.{self.data_grid_name}.{self.temporal_res:02d}_hour_precipitation"
            dir_name = f"{main_prefix}.stats"

        nc_dir = os.path.join(self.netcdf_dir, dir_name)
        if (not os.path.exists(nc_dir)):
            os.mkdir(nc_dir)

        if (time_period_type is not None):
            main_prefix += f".{time_period_type}"

        if (stat_type is not None):
            main_prefix += f".{stat_type}"

        if (agg_type is not None):
            main_prefix += f".{agg_type}"
        
        fname = f"{main_prefix}.{time_period_str}.{self.region}.nc"
        fpath = os.path.join(nc_dir, fname)
        return fpath
    ##### END Private methods to support stats calculations #####

    ##### Public methods plotting #####
    def plot_aggregated_fss(self, eval_type = evaluate_by_radius_kw_str, xaxis_explicit_values = False,
                            time_period_type = "full_period", include_frequency_bias = False, is_pctl_threshold = False,
                            include_fss_uniform = True):
        # Only plot frequency bias on second axis if the first axis is plotted against amount thresholds.
        if is_pctl_threshold or (eval_type != evaluate_by_threshold_kw_str):
            include_frequency_bias = False

        # Aggregated FSS data to plot
        fss_agg_dict, afss_agg_dict, fss_uniform_agg_dict = self.calculate_aggregated_fss(eval_type = eval_type, time_period_type = time_period_type)

        # Frequency bias data to plot (done for plotting against thresholds, only)
        if include_frequency_bias:
            frequency_bias_dict = self.calculate_aggregated_occ_stats("frequency_bias", time_period_type = time_period_type)
        
        # Based on this particular dataset, get a list of all the valid datetimes we're going to plot
        dtimes = sorted(list(fss_agg_dict.keys()))
        
        # Loop through dtimes, creating a plot for each one 
        for dtime in dtimes: 
            if (type(dtime) is pd.Timestamp) or (type(dtime) is dt.datetime):
                dt_str = dtime.strftime("%Y%m") 
            elif (type(dtime) is str):
                dt_str = dtime
            else:
                dt_str = dtime.strftime("%Y%m")

            if (dt_str in pputils.construct_monthly_string_list()):
                time_period_number = pputils.month_string_to_month_number(dt_str)
                dt_str_ext = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            elif (dt_str in pputils.construct_seasonal_string_list()): 
                time_period_number = pputils.season_string_to_season_number(dt_str)
                dt_str_ext = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            else:
                dt_str_ext = dt_str

            # Create figure
            short_name = pdp.format_short_name(self.da_dict[self.truth_data_name])
            ylims = (0, 1.0)
            yticks = np.arange(0, 1.1, 0.1)
            if (eval_type == evaluate_by_radius_kw_str):
                if is_pctl_threshold:
                    fixed_threshold_units = "th pctl"
                    fixed_threshold_units_no_space = fixed_threshold_units.replace(" ", "_")                    
                else:
                    fixed_threshold_units = " mm"
                    fixed_threshold_units_no_space = fixed_threshold_units.replace(" ", "")
                title = f"FSS vs. radius, {self.region} {short_name} (t = {self.fixed_fss_eval_threshold:0.1f}{fixed_threshold_units}): {dt_str}"
                xlabel = f"Evaluation radius ({self.fss_eval_radius_units})" 
                fig_name = f"FSSradius.{self.data_names_str}thresh{self.fixed_fss_eval_threshold:0.1f}{fixed_threshold_units_no_space}.{time_period_type}.{short_name}.{dt_str_ext}.{self.region}.png"
                if xaxis_explicit_values: # Plot explicitly against the selected eval radii or thresholds, with each value evenly spaced on x-axis
                    xaxis_var = np.arange(self.fss_eval_radius_da.shape[0])
                    xticks = self.fss_eval_radius_da.values
                else:
                    xaxis_var = self.fss_eval_radius_da 
                    xticks = np.arange(0, xaxis_var[-1] + 1, 1)
            elif (eval_type == evaluate_by_radius_ari_threshold_kw_str): 
                title = f"FSS vs. radius, {self.region} {short_name} (t = {self.fixed_fss_eval_threshold:02d} year ARI): {dt_str}"
                xlabel = f"Evaluation radius ({self.fss_eval_radius_units})" 
                fig_name = f"FSSradius.{self.data_names_str}thresh{self.fixed_fss_eval_threshold:02d}year_ari.{time_period_type}.{short_name}.{dt_str_ext}.{self.region}.png"
                if xaxis_explicit_values: # Plot explicitly against the selected eval radii or thresholds, with each value evenly spaced on x-axis
                    xaxis_var = np.arange(self.fss_eval_radius_da.shape[0])
                    xticks = self.fss_eval_radius_da.values
                else:
                    xaxis_var = self.fss_eval_radius_da 
                    xticks = np.arange(0, xaxis_var[-1] + 1, 1)
            elif (eval_type == evaluate_by_ari_kw_str):
                title = f"FSS vs. ARI value, {self.region} {short_name} (r = {self.fixed_fss_eval_radius:0.2f}): {dt_str}"
                xlabel = f"ARI (years)" 
                fig_name = f"FSSari.{self.data_names_str}radius{self.fixed_fss_eval_radius:0.2f}{self.fss_eval_radius_units}.{time_period_type}.{short_name}.{dt_str_ext}.{self.region}.png"
                if xaxis_explicit_values: # Plot explicitly against ARI thresholds, with each value evenly spaced on x-axis
                    xaxis_var = np.arange(self.fss_eval_ari_da.shape[0])
                    xticks = self.fss_eval_ari_da.values
                else:
                    xaxis_var = self.fss_eval_ari_da 
                    xticks = np.arange(0, xaxis_var[-1] + 1, 1)
                ylims = (0, 0.4)
                yticks = np.arange(0, 0.45, 0.05)
            else:
                title_prefix = "FSS"
                figname_prefix = "FSS"
                if include_frequency_bias:
                    title_prefix += ", frequency bias"
                    figname_prefix += ".freqBias."
                title = f"{title_prefix} vs. threshold, {self.region} {short_name} (r = {self.fixed_fss_eval_radius:0.2f} {self.fss_eval_radius_units}): {dt_str}"
                if is_pctl_threshold:
                    threshold_units = "pctl"
                else:
                    threshold_units = "mm"
                xlabel = f"Threshold ({threshold_units})" 
                fig_name = f"{figname_prefix}threshold_{threshold_units}.{self.data_names_str}radius{self.fixed_fss_eval_radius:0.2f}{self.fss_eval_radius_units}.{time_period_type}.{short_name}.{dt_str_ext}.{self.region}.png"
                if xaxis_explicit_values:
                    xaxis_var = np.arange(self.fss_eval_threshold_da.shape[0])
                    xticks = self.fss_eval_threshold_da.values
                else:
                    xaxis_var = self.fss_eval_threshold_da
                    xticks = np.arange(0, xaxis_var[-1] + 10, 10)
            fig = plt.figure(figsize = (13, 10))

            plot_dicts_list = [ fss_agg_dict[dtime] ]
            ylabels_list = ["Fractions Skill Score (FSS)"]
            ylims_list = [ ylims ]
            yticks_list = [ yticks ]
            subplot_titles_list = ["FSS"]
            if include_frequency_bias:
                axes_list = [
                            plt.subplot2grid((1, 2), (0, 0), colspan = 1, rowspan = 1),
                            plt.subplot2grid((1, 2), (0, 1), colspan = 1, rowspan = 1),
                            ]
                plot_dicts_list.append(frequency_bias_dict[dtime])
                ylabels_list.append("Frequency Bias")
                ylims_list.append( (0, 2.0) )
                yticks_list.append( np.arange(0, 2.2, 0.2) )
                subplot_titles_list.append("Frequency Bias")
            else:
                axes_list = [
                            plt.subplot2grid((1, 1), (0, 0)),
                            ]

            # Plot data
            for axis, plot_dict, ylabel, ylims, yticks, subplot_title in zip(axes_list, plot_dicts_list, ylabels_list, ylims_list, yticks_list, subplot_titles_list):
                axis.set_xlabel(xlabel, size = 15)
                axis.set_ylabel(ylabel, size = 15)
                axis.set_xlim(xaxis_var[0], xaxis_var[-1])
                axis.set_ylim(ylims)
                if xaxis_explicit_values:
                    axis.set_xticks(xaxis_var, xticks)
                else:
                    axis.set_xticks(xticks)
                axis.set_yticks(yticks) 
                axis.tick_params(axis = "both", labelsize = 15)
                axis.grid(True, linewidth = 1.5)

                for data_name, da in plot_dict.items():
                    if (data_name == self.truth_data_name):
                        continue
                    axis.plot(xaxis_var, da, linewidth = 2, label = data_name,
                             color = pputils.time_series_color_dict[data_name])
                if (include_fss_uniform) and (subplot_title == "FSS") and \
                ((eval_type == evaluate_by_radius_kw_str) or (eval_type == evaluate_by_radius_ari_threshold_kw_str)):
                    fss_uniform = fss_uniform_agg_dict[dtime][self.data_names[-1]]
                    axis.plot([0, xticks[-1]], [fss_uniform, fss_uniform], linewidth = 3, color = "black", linestyle = "dashed") 
                if include_frequency_bias:
                    axis.set_title(subplot_title, fontsize = 15)
                    if (subplot_title == "Frequency Bias"): # If we're working on the frequency bias axis, add a line at bias = 1 (unbiased forecast)
                        axis.plot([0, xaxis_var[-1]], [1, 1], linewidth = 3, color = "black") 
                axis.legend(loc = "best", prop = {"size": 15})

            # Save figure 
            fig.suptitle(title, size = 15)
            fig.tight_layout()
            fig_path = os.path.join(self.plot_output_dir, fig_name)
            print(f"Saving {fig_path}")
            plt.savefig(fig_path)

    def how_to_plot_fss_timeseries(self):
        print("plot_fss_timeseries(eval_radius, eval_threshold)")

    # Plot time series of FSS for each valid time (i.e., each individual grid evaluation)
    def plot_fss_timeseries(self, eval_radius, eval_threshold):
        if not(hasattr(self, "fss_dict_by_radius")) or not(hasattr(self, "fss_dict_by_threshold")):
            print("Error: You need to calculate FSS over varying evaluation radii and thresholds")
            return

        if (eval_radius not in self.fss_eval_radius_da) and (eval_radius != self.fixed_fss_eval_radius):
            print(f"Error: FSS not calculated for radius {eval_radius:0.1f}")
            return

        if (eval_threshold not in self.fss_eval_threshold_da) and (eval_threshold != self.fixed_fss_eval_threshold):
            print(f"Error: FSS not calculated for threshold {eval_threshold:0.1f}")
            return

        # Create figure 
        plt.figure(figsize = (15, 10))
        short_name = pdp.format_short_name(self.da_dict[self.truth_data_name])
        plt.title(f"FSS time series, {self.region} {short_name} (r = {eval_radius:0.2f} {self.fss_eval_radius_units}, t = {eval_threshold:0.1f} mm): {self.daily_time_period_str}", size = 15)
        plt.xlabel("Period end time", size = 15)
        plt.ylabel("Fractions Skill Score (FSS)", size = 15)
        plt.xlim(self.valid_dt_list[0], self.valid_dt_list[-1])
        plt.ylim(0, 1.0)
        xticks = self.valid_dt_list[::2]
        if (self.temporal_res == 24):
            xtick_labels = [f"{xtick:%m/%d}" for xtick in xticks]
        else:
            xtick_labels = [f"{xtick:%m/%d %H}" for xtick in xticks]
        plt.xticks(xticks, xtick_labels, fontsize = 15, rotation = 60)
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize = 15) 
        plt.grid(True, linewidth = 0.5)
        
        # Plot data
        for data_name in self.data_names:
            if (data_name == self.truth_data_name):
                continue
       
            # If the threshold is equal to the fixed evaluation threshold,
            # data will be within self.fss_dict_by_radius DataArrays. Otherwise, it will
            # be in self.fss_dict_by_threshold DataArrays. 
            if (eval_threshold == self.fixed_fss_eval_threshold):
                da = self.fss_dict_by_radius[data_name].loc[:, eval_radius]
            else: 
                da = self.fss_dict_by_threshold[data_name].loc[:, eval_threshold]
            plt.plot(da.period_end_time.values, da, color = pputils.time_series_color_dict[data_name], linewidth = 2, label = data_name)  

        # Save figure
        plt.tight_layout()
        plt.legend(loc = "best", prop = {"size": 15})
        fig_name = f"FSStimeseries.{self.data_names_str}radius{eval_radius:0.2f}{self.fss_eval_radius_units}.threshold{eval_threshold:0.1f}mm.{short_name}.{self.daily_time_period_str}.{self.region}.png"
        fig_path = os.path.join(self.plot_output_dir, fig_name)
        print(f"Saving {fig_path}")
        plt.savefig(fig_path)

    def how_to_plot_cmap_multi_panel(self):
        print("NOTE: If data_dict is passed it will be used directly, without aggregation using time_period_type.\n"
              "So, the contents of data_dict must already be properly aggregated for desired cmaps.")
        print('plot_cmap_multi_panel(data_dict = None, plot_levels = np.arange(0, 85, 5),\n'
              '                      time_period_type = None, stat_type = "mean", pctl = 99,\n'
              '                      single_colorbar = True, single_set_of_levels = True, cmap = pputils.DEFAULT_PRECIP_CMAP,\n'
              '                      plot_errors = False, write_to_nc = False)')

    # Contour maps with the correct number of panels, with the "truth" dataset always in the top left
    # The input da_dict must have the same contents as self.data_dict: {da_name1: da1, ...., da_nameN, daN}
    # So, best practice is to have whatever data you want to plot exist as concatenated xarray DataArrays
    # with the proper time dimensions, as this method will make a plot for each coordinate of the time dimension.
    def plot_cmap_multi_panel(self, data_dict = None, plot_levels = np.arange(0, 85, 5),
                              time_period_type = None, stat_type = "mean", pctl = 99,
                              single_colorbar = True, single_set_of_levels = True, cmap = pputils.DEFAULT_PRECIP_CMAP, 
                              plot_errors = False, write_to_nc = False):
        if (data_dict is None):
            if self.LOAD_DATA_FLAG: 
                data_dict = self.calculate_aggregated_stats(time_period_type = time_period_type, 
                                                            stat_type = stat_type,
                                                            agg_type = "time",
                                                            pctl = pctl,
                                                            write_to_nc = write_to_nc) 
            else:
                data_dict = self.da_dict

        # Configure basic info about the data
        truth_da = data_dict[self.truth_data_name]
        num_da = len(data_dict.items())
        data_names_str = "".join(f"{key}." for key in data_dict.keys())

        # Get the contour levels for the plot
        if (plot_levels is not None):
            levels = plot_levels
            _, error_levels = self._calculate_levels_for_cmap(data_dict)
        else: 
            levels, error_levels = self._calculate_levels_for_cmap(data_dict)
        
        # Based on this particular dataset, get a list of all the valid datetimes we're going to plot 
        dtimes, time_dim, dt_format = self._create_datetime_list_from_da_time_dim(truth_da)

        # Set map projection to be used for all subplots in the figure 
        proj = ccrs.PlateCarree()

        # Loop through these datetimes, making figures with subplots corresponding to each of the data arrays in data_dict
        for dtime in dtimes:
            figsize = pputils.set_figsize_based_on_num_da(num_da, self.region, single_colorbar = single_colorbar)
            fig = plt.figure(figsize = figsize) 
            axes_list, cbar_ax = pputils.create_gridded_subplots(num_da, proj, single_colorbar = single_colorbar) 

            if (type(dtime) is pd.Timestamp):
                loc_str = dtime.strftime(utils.full_date_format_str) # Format is %Y-%m-%d %H:%M:%S
                dt_str = dtime.strftime(dt_format) 
            elif (type(dtime) is str):
                loc_str = dtime
                dt_str = dtime
            else:
                print(f"Invalid datetime {dtime} to select data; not continuing to make plots")
                return 
      
            # Create different contour levels for each dtime rather than a fixed set of levels for all dtimes.
            # Note the levels will all be the same (e.g., error levels) within one single figure, as one output
            # figure corresponds to one dtime. 
            if not(single_set_of_levels):
                data_to_plot_dict = {}
                for data_name, da in data_dict.items():
                    data_to_plot_dict[data_name] = da.loc[loc_str]
                levels, error_levels = self._calculate_levels_for_cmap(data_to_plot_dict)

            # Loop through each of the subplot axes defined above (one axis for each DataArray) and plot the data 
            for axis, (data_name, da) in zip(axes_list, data_dict.items()):
                pputils.add_cartopy_features_to_map_proj(axis, self.region, proj, draw_labels = False)

                if (plot_errors) and (data_name != self.truth_data_name):
                    data_to_plot = (da - truth_da).loc[loc_str]
                    plot_levels = error_levels
                    subplot_title = f"{data_name} errors"
                    cmap = "seismic_r"
                    extend_kwarg = "both"
                else:
                    data_to_plot = da.loc[loc_str]
                    plot_levels = levels
                    subplot_title = data_name
                    cmap = cmap 
                    extend_kwarg = "both" # TODO: Get max to work (currently skews the colorbar off-center)

                # One colorbar for entire figure; add as its own separate axis defined using subplot2grid 
                if single_colorbar:
                    plot_handle = data_to_plot.plot(ax = axis, levels = plot_levels, transform = proj, extend = extend_kwarg, cmap = cmap,
                                                    add_colorbar = not(single_colorbar))
                    cbar = fig.colorbar(plot_handle, cax = cbar_ax, ticks = plot_levels, shrink = 0.5, orientation = "horizontal")
                    cbar.set_label(da.units, size = 15)
                    cbar_tick_labels_rotation, cbar_tick_labels_fontsize = pputils.set_cbar_labels_rotation_and_fontsize(plot_levels, self.region, num_da, for_single_cbar = True)
                    cbar.ax.set_xticklabels(plot_levels, rotation = cbar_tick_labels_rotation) 
                    cbar.ax.tick_params(labelsize = cbar_tick_labels_fontsize)
                # Separate colorbar for each subplot
                else:
                    plot_handle = data_to_plot.plot(ax = axis, levels = plot_levels, transform = proj, extend = extend_kwarg, cmap = cmap,
                                                    cbar_kwargs = {"shrink": 0.6, "ticks": plot_levels, "pad": 0.02, "orientation": "horizontal"})
                    plot_handle.colorbar.set_label(da.units, size = 15, labelpad = -1.3)
                    cbar_tick_labels = pputils.create_sparse_cbar_ticks(plot_levels) # 20241126: Label every other tick on subplot colorbars
                    cbar_tick_labels_rotation, cbar_tick_labels_fontsize = \
                    pputils.set_cbar_labels_rotation_and_fontsize(cbar_tick_labels, self.region, num_da, for_single_cbar = False)
                    plot_handle.colorbar.ax.set_xticklabels(cbar_tick_labels, rotation = cbar_tick_labels_rotation) 
                    plot_handle.colorbar.ax.tick_params(labelsize = cbar_tick_labels_fontsize)

                axis.set_title(subplot_title, fontsize = 16 + self.poster_font_increase)

            formatted_short_name = pdp.format_short_name(truth_da)
            if plot_errors:
                formatted_short_name += "_errors"
            # Create the plot title. If we're looping through individual Timestamp objects, they represent
            # the period end time of the data. Indicate this explicitly in the title.
            if (type(dtime) is pd.Timestamp):
                title_string = f"{self.region} {self._format_short_short_name(truth_da)} ending at {dt_str}"
            else:
                title_string = f"{self.region} {self._format_short_short_name(truth_da)}: {dt_str}"
            fig.suptitle(title_string, fontsize = 16 + self.poster_font_increase, fontweight = "bold")
            fig.tight_layout()

            # Save figure
            if (dt_str in pputils.construct_monthly_string_list()):
                time_period_number = pputils.month_string_to_month_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            elif (dt_str in pputils.construct_seasonal_string_list()): 
                time_period_number = pputils.season_string_to_season_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"

            fig_name = f"cmap.{data_names_str}{formatted_short_name}.{dt_str}.{self.region}.png"
            fig_path = os.path.join(self.plot_output_dir, fig_name)
            print(f"Saving {fig_path}")
            plt.savefig(fig_path)

    def how_to_plot_timeseries(self):
        print("NOTE: If data_dict is passed it will be used directly, without aggregation using time_period_type.\n"
              "So, the contents of data_dict must already be properly aggregated for a time series.")
        print('plot_timeseries(data_dict = None, time_period_type = None, stat_type = "mean", pctl = 99, write_to_nc = False,\n'
              '                ann_plot = True, which_ann_text = "all", pct_errors_ann_text = True, write_stats = True)')

    def plot_timeseries(self, data_dict = None, time_period_type = None, stat_type = "mean", pctl = 99, write_to_nc = False,
                        ann_plot = True, which_ann_text = "all", pct_errors_ann_text = True, write_stats = True):
        if (data_dict is None):
            if self.LOAD_DATA_FLAG:
                data_dict = self.calculate_aggregated_stats(time_period_type = time_period_type, 
                                                            stat_type = stat_type,
                                                            agg_type = "space_time",
                                                            pctl = pctl,
                                                            write_to_nc = write_to_nc)
            else:
                data_dict = self.da_dict

        truth_da = data_dict[self.truth_data_name] 
        num_da = len(data_dict.items())

        match stat_type:
            case "pctl":
                title_string = f"{pctl:0.1f}th {stat_type}, {self.region}"
                fig_name_prefix = "{pctl:0.1f}th_pctl"
                yticks = pputils.variable_pctl_plot_limits("accum_precip", self.temporal_res)
                axis_label = pdp.format_short_name(truth_da)
            case "mean": 
                title_string = f"{stat_type.title()}, {self.region}"
                fig_name_prefix = "mean"
                yticks = pputils.regions_info_dict[self.region].ts_mean_precip_range
                if (time_period_type == None) or (time_period_type == "daily"):
                    yticks = 2.0 * np.copy(yticks) 
                axis_label = self._format_short_short_name(truth_da)
            case _:
                raise NotImplementedError

        if (time_period_type is not None):
            fig_name_prefix = f"{time_period_type}_{fig_name_prefix}"

        match time_period_type:
            case None:
                time_period_str = self.daily_time_period_str
                title_string = f"{title_string}: {time_period_str}"
                xlabel = "Period end time"
            case "daily":
                time_period_str = self.daily_time_period_str_period_begin 
                title_string = f"Daily {title_string}: {time_period_str}"
                xlabel = "Days"
            case "monthly":
                time_period_str = self.monthly_time_period_str
                title_string = f"Monthly {title_string}: {time_period_str}"
                xlabel = "Months"
            case "seasonal":
                time_period_str = self.monthly_time_period_str
                title_string = f"Seasonal {title_string}: {time_period_str}"
                xlabel = "Seasons"
            case "annual":
                time_period_str = self.annual_time_period_str
                title_string = f"Annual {title_string}: {time_period_str}"
                xlabel = "Years"
            case "common_monthly":
                time_period_str = self.monthly_time_period_str
                title_string = f"Common monthly {title_string}: {time_period_str}"
                xlabel = "Months"
            case "common_seasonal": 
                time_period_str = self.monthly_time_period_str
                title_string = f"Common seasonal {title_string}: {time_period_str}"
                xlabel = "Seasons"

        fig_name = f"timeseries.{fig_name_prefix}.{axis_label}.{time_period_str}.{self.region}.png"
        dtimes, time_dim, dt_format = self._create_datetime_list_from_da_time_dim(truth_da)
       
        # Plot the data
        fig = plt.figure(figsize = (15, 10))
        for data_name, data_array in data_dict.items():
            color = pputils.time_series_color_dict[data_name]
            data_array.plot(label = data_name, color = color, linewidth = 3, linestyle = "solid")
        plt.legend(loc = "upper right", prop = {"size": 15 + self.poster_font_increase})

        # Configure the plot
        plt.grid(True, linewidth = 0.3, linestyle = "dashed")
        plt.title(title_string, size = 16 + self.poster_font_increase, fontweight = "bold")
        plt.xlabel(xlabel, size = 16) 
        plt.ylabel(f"{axis_label} [{truth_da.units}]", size = 16 + self.poster_font_increase)

        if (len(dtimes) <= 60):
            xticks = dtimes
        elif (len(dtimes) <= 120):
            xticks = dtimes[::2]
        else:
            xticks = dtimes[::4]

        if (dt_format != ""):
            if (time_period_type == None):
                if (self.temporal_res == 24):
                    dt_format = "%m/%d"
                else:
                    dt_format = "%m/%d %H"
            xtick_labels = [xtick.strftime(dt_format) for xtick in xticks]
        else:
            xtick_labels = [xtick for xtick in xticks]

        plt.xticks(xticks, xtick_labels, fontsize = 15 + self.poster_font_increase, rotation = 60)
        plt.yticks(yticks, fontsize = 15 + self.poster_font_increase)
        plt.xlim(dtimes[0], dtimes[-1])
        plt.ylim(yticks[0], yticks[-1])
        plt.tight_layout()

        # Calculate basic statistics and annotate on the plot 
        print("Calculating statistics to annotate to timeseries plot")
        if (num_da < 3):
            ann_pos_step = 0.40
            ann_size = 15
        elif (num_da == 3):
            ann_pos_step = 0.35
            ann_size = 15
        elif (num_da == 4):
            ann_pos_step = 0.25
            ann_size = 14
        elif (num_da == 5):
            ann_pos_step = 0.20
            ann_size = 12
        else:
            ann_pos_step = 0.17
            ann_size = 10
        
        ann_pos_horz = 0.01
        ann_pos_vert = 0.86
        range_truth_data = truth_da.max().load().item() - truth_da.min().load().item()
        ann_text_dict = {}
        for data_name, da in data_dict.items():
            if (data_name == self.truth_data_name):
                continue

            # Calculate stats
            correlation = self.calculate_pearsonr(da, truth_da)
            errors = (da - truth_da).load()
            squared_errors = errors**2
            mean_bias = errors.mean().item()
            norm_mean_bias = 100 * mean_bias/range_truth_data
            rmse = np.sqrt(squared_errors.mean().item())
            norm_rmse = 100 * rmse/range_truth_data

            # Create the text
            data_name_text = f"{data_name}:\n"
            corr_text = f"Corr: {correlation.statistic:0.2f} (p-value: {correlation.pvalue:0.2f})\n"
            bias_text = f"Mean bias: {mean_bias:0.2f}{truth_da.units}\n"
            rmse_text = f"RMSE: {rmse:0.2f}{truth_da.units}\n"

            if pct_errors_ann_text:
                bias_text = bias_text.strip("\n") + f" ({norm_mean_bias:0.1f}%)\n"
                rmse_text = rmse_text.strip("\n") + f" ({norm_rmse:0.1f}%)\n"

            if (which_ann_text == "corr"):
                ann_text = data_name_text + corr_text
            elif (which_ann_text == "errors"):
                ann_text = data_name_text + bias_text + rmse_text
            else: 
                ann_text = data_name_text + corr_text + bias_text + rmse_text

            # Annotate the plot
            print(ann_text)
            ann_text_dict[data_name] = ann_text
            if ann_plot:
                plt.annotate(ann_text, weight = "bold",
                             xy = (ann_pos_horz, ann_pos_vert),
                             xytext = (ann_pos_horz, ann_pos_vert),
                             xycoords = "axes fraction", size = ann_size)
            ann_pos_horz += ann_pos_step

            if write_stats:
                stats_dir = os.path.join(self.home_dir, "stats")
                stats_fname = fig_name.split(".png")[0] + ".txt"
                stats_fpath = os.path.join(stats_dir, stats_fname)
                print(f"Writing stats to {stats_fpath}")
                Fstats = open(stats_fpath, "w")

                for data_name, ann_text in ann_text_dict.items():
                    Fstats.write(ann_text)
                Fstats.close()

        # Save figure
        fig_fpath = os.path.join(self.plot_output_dir, fig_name)
        print(f"Saving {fig_fpath}")
        plt.savefig(fig_fpath)

    # TODO: Consider plotting at midpoint of bin rather than left edge of bin
    def plot_pdf(self, data_dict = None, time_period_type = "full_period", write_to_nc = False):
        if (data_dict is None):
            if self.LOAD_DATA_FLAG:
                data_dict = self.calculate_pdf(time_period_type = time_period_type) 
            else:
                data_dict = self.da_dict

        # Plot data
        for dtime in data_dict.keys():
            da_dict_this_dtime = data_dict[dtime]
            data_names_str = "".join(f"{key}." for key in da_dict_this_dtime.keys())

            # Calculate reasonable axis ticks and bounds 
            # Compile lists of maximum bins and minimum probabilities
            max_bin_list = []
            min_prob_list = []
            max_total_samples_list = []
            for data_name, (pdf_values, pdf_bins, total_samples) in da_dict_this_dtime.items():
                pdf_bin_max = np.max(pdf_bins) 
                max_bin_list.append(pdf_bin_max)
                prob_min = np.min(pdf_values[pdf_values > 0.0])
                min_prob_list.append(prob_min)
                max_total_samples_list.append(total_samples)
            
            # Calculate x-axis ticks based on overall max bin 
            overall_max = np.max(np.array(max_bin_list)) 
            rounded_overall_max = np.round(overall_max, decimals = -1) + 10
            step_size = np.round(rounded_overall_max/10., decimals = -1) 
            xticks = np.arange(0, rounded_overall_max + step_size, step_size)

            # Calculate y-axis (log scale) ticks based on overall min probability
            overall_min = np.min(np.array(min_prob_list))
            exp_overall_min = np.floor(np.log10(overall_min))
            yticks = [10**i for i in np.arange(exp_overall_min, 1, 1)] 

            # Initialize figure and plot data 
            plt.figure(figsize = (10,10))
            for data_name, (pdf_values, pdf_bins, total_samples) in da_dict_this_dtime.items():
                plt.plot(pdf_bins[:-1], pdf_values, label = data_name, color = pputils.time_series_color_dict[data_name], linewidth = 2)
            plt.gca().set_yscale("log")
            plt.grid(True, linewidth = 0.5)
            plt.legend(loc = "upper right", prop = {"size": 15})

            # Configure plot title and figure name
            short_name = self.da_dict[self.truth_data_name].short_name
            units = self.da_dict[self.truth_data_name].units
            if (type(dtime) is pd.Timestamp):
                dt_str = dtime.strftime("%Y%m") 
            elif (type(dtime) is str):
                dt_str = dtime
            plt.xlabel(f"{short_name} [{units}]", size = 16)
            plt.ylabel("Probability", size = 16)
            plt.xlim(xticks[0], xticks[-1])
            plt.xticks(xticks, fontsize = 15 + self.poster_font_increase)
            plt.ylim(yticks[0], yticks[-1])
            plt.yticks(yticks, fontsize = 15 + self.poster_font_increase)
            title_string = f"{self._format_short_short_name(self.truth_da)} PDF, {self.region}: {dt_str}"
            plt.title(title_string, size = 16 + self.poster_font_increase, fontweight = "bold")
            plt.tight_layout()

            # Save figure
            if (dt_str in pputils.construct_monthly_string_list()):
                time_period_number = pputils.month_string_to_month_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"
            elif (dt_str in pputils.construct_seasonal_string_list()): 
                time_period_number = pputils.season_string_to_season_number(dt_str)
                dt_str = f"{time_period_number:02d}{dt_str}.{self.monthly_time_period_str}"

            fig_name = f"pdf.{data_names_str}{time_period_type}.{dt_str}.{self.region}.png"
            fig_fpath = os.path.join(self.plot_output_dir, fig_name)
            print(f"Saving {fig_fpath}")
            plt.savefig(fig_fpath)

    # Basic scatter plot of truth dataset versus other dataset 
    def plot_scatter_plot(self, data_dict = None, summed_over_time_period = False):
        if (data_dict is None):
            data_dict = self.da_dict
            truth_da = self.truth_da

        if summed_over_time_period: 
            data_dict = self.da_dict_summed_time
            truth_da = self.truth_da_summed_time
        
        for data_name, da in data_dict.items(): 
            if (data_name == self.truth_data_name):
                continue

            # Plot data
            plt.figure(figsize = (10, 10))
            current_axes = pylab.gca()
            plt.scatter(da, truth_da) 
            plt.xlabel(f"{data_name} {da.short_name} [{da.units}]", size = 15)
            plt.ylabel(f"{self.truth_data_name} {self.truth_da.short_name} [{self.truth_da.units}]", size = 15)

            # Annotate with correlation
            correlation = self.calculate_pearsonr(da, truth_da)
            ann_text = f"Corr: {correlation.statistic:0.2f} (p-value: {correlation.pvalue:0.2f})"
            plt.annotate(ann_text, xy = (0.05, 0.95), xytext = (0.05, 0.95), xycoords = "axes fraction", size = 12)

            # Format plot and save figure
            plt.title(f"{self.truth_data_name} vs. {data_name} scatter plot: {da.short_name}, {self.daily_time_period_str}", size = 15)
            plt.grid(True, linewidth = 0.5, linestyle = "dashed")
            axes_labels, _ = self._calculate_levels_for_cmap({data_name: da, self.truth_data_name: self.truth_da})
            plot_lims = [axes_labels[0], axes_labels[-1]] 
            plt.xlim(plot_lims)
            plt.ylim(plot_lims)
            plt.plot(plot_lims, plot_lims, color = "black", linewidth = 2.5)
            fig_name = f"scatter_plot.{self.truth_data_name}.{data_name}.{pdp.format_short_name(da)}.{self.daily_time_period_str}.{self.region}.png"
            fig_path = os.path.join(self.plot_output_dir, fig_name)
            print(f"Saving {fig_path}")
            plt.savefig(fig_path)
    ##### END Public methods plotting #####

    ##### Private methods to support plotting #####
    # Calculate plot levels for a filled contour map
    def _calculate_levels_for_cmap(self, da_dict, saturate = True):
        max_list = []
        for data_name, da in da_dict.items():
            da_max = da.max()
            max_list.append(da_max)
        max_da = xr.concat(max_list, dim = "x")

        # 20241015: Changed from taking the max of the max of each data array to the
        # mean to add more "pop"/saturation to the contour levels, especially for very high pctls (e.g. 99.9th).
        # This approach attempts to address the issues where one dataset has a much higher max than the rest,
        # which can result in washing out contrast in the contour plots. 
        overall_max = max_da.mean().item()

        if (overall_max <= 25):
            step_size = 1
            error_levels = np.arange(-5, 5.5, 0.5)
        elif (overall_max <= 50):
            step_size = 2
            error_levels = np.arange(-5, 5.5, 0.5)
        elif (overall_max <= 75):
            step_size = 4
            error_levels = np.arange(-10, 11, 1)
        elif (overall_max <= 100):
            step_size = 5
            error_levels = np.arange(-10, 11, 1)
        elif (overall_max <= 180):
            step_size = 5
            error_levels = np.arange(-20, 22, 2)
        elif (overall_max <= 300):
            step_size = 10
            error_levels = np.arange(-40, 45, 5) 
        elif (overall_max <= 800):
            step_size = 20
            error_levels = np.arange(-140, 150, 10)
        elif (overall_max <= 2000):
            step_size = 50
            error_levels = np.arange(-300, 325, 25)
        else:
            step_size = 100
            error_levels = np.arange(-600, 650, 50)

        overall_max = int(np.round(overall_max, decimals = 0) + step_size)
        overall_max -= overall_max % step_size
        
        # The idea of saturate is to make the plots 'pop' and really highlight
        # the areas of maximum precip. Otherwise if saturate=False, attempt
        # to capture the entire dynamic range of precipitation (the downside in that
        # case is that areas of low precip become very hard to differentiate). 
        if saturate:
            levels = np.arange(0, overall_max - 3 * step_size, step_size)
        else:
            levels = np.arange(0, overall_max + step_size, step_size)
       
        return levels, error_levels

    # Create a list of datetimes (which may be actual datetime objects
    # or strings (e.g., season strings like 'DJF') depending on the 
    # dimension name of the input data array.
    def _create_datetime_list_from_da_time_dim(self, da):
        dims = da.dims
        create_dt_list = True

        if (utils.time_dim_str in dims):
            time_dim = utils.time_dim_str
            dt_format = "%Y%m%d.%H%M" 
        elif (utils.days_dim_str in dims):
            time_dim = utils.days_dim_str
            dt_format = "%Y%m%d"
        elif (utils.period_begin_time_dim_str in dims):
            time_dim = utils.period_begin_time_dim_str 
            dt_format = "%Y%m%d.%H" 
        elif (utils.period_end_time_dim_str in dims): 
            time_dim = utils.period_end_time_dim_str
            dt_format = "%Y%m%d.%H" 
        elif (utils.months_dim_str in dims):
            time_dim = utils.months_dim_str
            dt_format = "%Y%m"
        elif (utils.seasons_dim_str in dims):
            time_dim = utils.seasons_dim_str
            dt_format = ""
            create_dt_list = False
            dtimes = list(da[utils.seasons_dim_str].values)
        elif (utils.annual_dim_str in dims):
            time_dim = utils.annual_dim_str
            dt_format = "%Y"
        elif (utils.full_period_dim_str in dims):
            time_dim = utils.full_period_dim_str
            dt_format = ""
            create_dt_list = False
            dtimes = [ str(da[time_dim].values[0]) ] 
        elif (utils.common_month_dim_str in dims):
            time_dim = utils.common_month_dim_str
            dt_format = ""
            create_dt_list = False
            dtimes = pputils.construct_monthly_string_list()
        elif (utils.common_season_dim_str in dims):
            time_dim = utils.common_season_dim_str
            dt_format = "" 
            create_dt_list = False
            dtimes = pputils.construct_seasonal_string_list()
        else:
            print(f"Error: time dimension not recognized from dimension list {dims}; not making monthly timeseries plot")
            return
        
        if create_dt_list:
            dtimes = [pd.Timestamp(i) for i in da[time_dim].values]
 
        return dtimes, time_dim, dt_format

    # Further format a DataArray's short name from the format separated
    # by underscores to only the first three items separated by underscores.
    def _format_short_short_name(self, data_array):
        formatted_short_name = pdp.format_short_name(data_array)
        str_split = formatted_short_name.split("_")
        formatted_short_short_name = f"{str_split[0]}_{str_split[1]}_{str_split[2]}" 
        return formatted_short_short_name
    ##### END Private methods to support plotting #####

