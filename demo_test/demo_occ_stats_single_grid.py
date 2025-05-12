# NOTE: These methods are meant to be used with PrecipVerificationProcessor.

    # Calculate occurrence statistics for individual forecast grids. May be necessary to 
    # use ARI grids for thresholds
    def calculate_occ_stats_single_grid(self, input_da_dict = None,
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
                
            dtimes, time_dim, dt_format = self._create_datetime_list_from_da_time_dim(da)
            for d, dtime in enumerate(dtimes):
                model_precip = da.sel(period_end_time = dtime) # FIXME (here and elsewhere): generalize to any dimension name
                obs_precip = self.truth_da.sel(period_end_time = dtime) 

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

                hits_tmp_array = np.array(hits_list).reshape((1, data_coords.shape[0]))
                misses_tmp_array = np.array(misses_list).reshape((1, data_coords.shape[0]))
                false_alarms_tmp_array = np.array(false_alarms_list).reshape((1, data_coords.shape[0]))
                correct_negatives_tmp_array = np.array(correct_negatives_list).reshape((1, data_coords.shape[0]))
                total_events_tmp_array = np.array(total_events_list).reshape((1, data_coords.shape[0]))
                frequency_bias_tmp_array = np.array(frequency_bias_list).reshape((1, data_coords.shape[0]))
                CSI_tmp_array = np.array(CSI_list).reshape((1, data_coords.shape[0]))
                ETS_tmp_array = np.array(ETS_list).reshape((1, data_coords.shape[0]))
                if (d == 0):
                    hits_array = np.copy(hits_tmp_array) 
                    misses_array = np.copy(misses_tmp_array) 
                    false_alarms_array = np.copy(false_alarms_tmp_array) 
                    correct_negatives_array = np.copy(correct_negatives_tmp_array) 
                    total_events_array = np.copy(total_events_tmp_array) 
                    frequency_bias_array = np.copy(frequency_bias_tmp_array) 
                    CSI_array = np.copy(CSI_tmp_array) 
                    ETS_array = np.copy(ETS_tmp_array) 
                else:
                    hits_array = np.concatenate((hits_array, hits_tmp_array), axis = 0)
                    misses_array = np.concatenate((misses_array, misses_tmp_array), axis = 0)
                    false_alarms_array = np.concatenate((false_alarms_array, false_alarms_tmp_array), axis = 0)
                    correct_negatives_array = np.concatenate((correct_negatives_array, correct_negatives_tmp_array), axis = 0)
                    total_events_array = np.concatenate((total_events_array, total_events_tmp_array), axis = 0)
                    frequency_bias_array = np.concatenate((frequency_bias_array, frequency_bias_tmp_array), axis = 0)
                    CSI_array = np.concatenate((CSI_array, CSI_tmp_array), axis = 0)
                    ETS_array = np.concatenate((ETS_array, ETS_tmp_array), axis = 0)

            # Convert numpy array to DataArray
            hits_da = xr.DataArray(hits_array, coords = [dtimes, data_coords], dims = [utils.period_end_time_dim_str, dim_name])
            misses_da = xr.DataArray(misses_array, coords = [dtimes, data_coords], dims = [utils.period_end_time_dim_str, dim_name])
            false_alarms_da = xr.DataArray(false_alarms_array, coords = [dtimes, data_coords], dims = [utils.period_end_time_dim_str, dim_name])
            correct_negatives_da = xr.DataArray(correct_negatives_array, coords = [dtimes, data_coords], dims = [utils.period_end_time_dim_str, dim_name])
            total_events_da = xr.DataArray(total_events_array, coords = [dtimes, data_coords], dims = [utils.period_end_time_dim_str, dim_name])
            frequency_bias_da = xr.DataArray(frequency_bias_array, coords = [dtimes, data_coords], dims = [utils.period_end_time_dim_str, dim_name])
            CSI_da = xr.DataArray(CSI_array, coords = [dtimes, data_coords], dims = [utils.period_end_time_dim_str, dim_name])
            ETS_da = xr.DataArray(ETS_array, coords = [dtimes, data_coords], dims = [utils.period_end_time_dim_str, dim_name])
            hits_da.name = "hits" 
            misses_da.name = "misses" 
            false_alarms_da.name = "false_alarms" 
            correct_negatives_da.name = "correct_negatives" 
            total_events_da.name = "total_events" 
            frequency_bias_da.name = "frequency_bias" 
            CSI_da.name = "CSI" 
            ETS_da.name = "ETS" 

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

    # Calculated occurence statistics aggregated over specified time periods (common monthly, common seasonal, etc.)
    # starting from stats valid over single forecast grids
    def calculate_aggregated_occ_stats_single_grid(self, which_stat, time_period_type = "full_period"):
        # Process time_period_type: list of date times, dimension name, etc.
        dtimes, dim_name, time_period_str = self._process_time_period_type_to_dtimes(time_period_type)

        # If aggregating seasonally, dtimes, which in this case will be a list of lists defining
        # seasonal datetime ranges, wasn't defined above.
        if (time_period_type == "seasonal"):
            dtimes = self._construct_season_dt_ranges(data_array)
        
        all_occ_stats_dict = self.calculate_occ_stats_single_grid()
        stat_dict = self.extract_occ_stat(all_occ_stats_dict, which_stat)

        occ_stat_agg_dict = {}
        for dtime in dtimes:
            da_dict_occ_stat_each_dtime = {}
            for data_name, data_array in stat_dict.items():
                if (data_name == self.truth_data_name):
                    continue

                # Convert data coordinates to period beginning (much easier to aggregate over months that way)
                data_array = self._convert_period_end_to_period_begin(data_array)
                data_to_aggregate = self._determine_agg_data_from_time_period_type(data_array, time_period_type, dtime)
                data = data_to_aggregate.mean(dim = utils.period_begin_time_dim_str) # Final result should be a 1-D data array with dimensions = thresholds

                da_dict_occ_stat_each_dtime[data_name] = data 

            occ_stat_agg_dict[dtime] = da_dict_occ_stat_each_dtime 

        return occ_stat_agg_dict 

