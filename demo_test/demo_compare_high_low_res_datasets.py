#!/usr/bin/env python

import argparse
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import precip_plotting_utilities as pputils
import precip_verification_processor
import sys
import utilities as utils
import warnings

def main():
    utils.suppress_warnings()
    program_description = ("Compare high- and low-res precipitation datasets")
    parser = argparse.ArgumentParser(description = program_description)
    parser.add_argument("start_dt_str",
                        help = "Start date/time of verification; format consistent with temporal_res flag, e.g., 'YYYYmmdd' for 24-hourly data")
    parser.add_argument("end_dt_str",
                        help = "End date/time of verification; format consistent with temporal_res flag, e.g., 'YYYYmmdd' for 24-hourly data")
    parser.add_argument("--region", dest = "region", default = "CONUS",
                        help = "Region to zoom plot to; default CONUS")
    parser.add_argument("--temporal_res", default = 24, type = int, choices = [1, 3, 24],
                        help = "Temporal resolution of precip data used in verification (default 24)")
    args = parser.parse_args()

    # Process high-res datasets (AORC, CONUS404, NestedReplay), on AORC grid
    print("***** PROCESSING HIGH-RES DATA") 
    data_names, truth_data_name, data_grid = precip_verification_processor.map_region_to_data_names(args.region, high_res = True) 
    verif_high = precip_verification_processor.PrecipVerificationProcessor(args.start_dt_str, 
                                                                           args.end_dt_str,
                                                                           data_names = data_names, 
                                                                           truth_data_name = truth_data_name,
                                                                           data_grid = data_grid,
                                                                           region = args.region,
                                                                           temporal_res = args.temporal_res)
                                                                      

    # Process low-res datasets (AORC, CONUS404, ERA5, IMERG, Replay), on Replay grid
    print("***** PROCESSING LOW-RES DATA") 
    data_names, truth_data_name, data_grid = precip_verification_processor.map_region_to_data_names(args.region, high_res = False) 
    verif_low = precip_verification_processor.PrecipVerificationProcessor(args.start_dt_str, 
                                                                          args.end_dt_str,
                                                                          data_names = data_names, 
                                                                          truth_data_name = truth_data_name,
                                                                          data_grid = data_grid,
                                                                          region = args.region,
                                                                          temporal_res = args.temporal_res)
   
    # Create new data dictionary combining high-res datasets and (for now),
    # just global Replay from the low-res datasets.
    new_da_dict = {} 
    for data_name, da in verif_high.da_dict.items():
        new_da_dict[data_name] = da
    new_da_dict["Replay"] = verif_low.da_dict["Replay"]

    return verif_high, verif_low, new_da_dict

if __name__ == "__main__":
    verif_high, verif_low, new_da_dict = main()


