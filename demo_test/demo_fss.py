#!/usr/bin/env python

import argparse
import datetime as dt
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import precip_plotting_utilities as pputils
import regionmask
import scipy
import utilities as utils
import xarray as xr

def main():
    parser = argparse.ArgumentParser(description = "Read in single files from all 24-hour precip datasets; calculate FSS against AORC")
    parser.add_argument("eval_dt_str", 
                        help = "Evaluation date (24-hour precip evaluated will be valid over that current day); format YYYYmmdd")
    parser.add_argument("-r", "--eval_radius", dest = "eval_radius", type = float, default = 0.5,
                        help = "Evaluation radius in degrees latitude/longitude; default 0.5")
    parser.add_argument("-g", "--grid_size", dest = "grid_size", type = float, default = 0.25,
                        help = "Grid size in degrees latitude/longitude; default 0.25")
    parser.add_argument("-t", "--threshold", dest = "threshold", type = float, default = 1,
                        help = "Threshold in mm; default = 1.0")
    parser.add_argument("-p", "--plot", dest = "plot", action = "store_true",
                        help = "Set to make plots")
    args = parser.parse_args()

    print(f"Radius: {args.eval_radius} degrees")
    print(f"Grid size: {args.grid_size} degrees")
    print(f"Threshold: {args.threshold} mm")

    # Read data
    da_dict = {}
    for dataset_name in ["AORC", "CONUS404", "ERA5", "IMERG", "Replay"]:
        if (dataset_name != "Replay"):
            dir_name = f"{dataset_name}.ReplayGrid.24_hour_precipitation"
        else:
            dir_name = f"{dataset_name}.NativeGrid.24_hour_precipitation"

        eval_dt = dt.datetime.strptime(args.eval_dt_str, "%Y%m%d")
        fname = f"{dir_name}.{args.eval_dt_str}.nc"
        fpath = os.path.join(utils.data_nc_dir, dir_name, fname)

        print(f"Reading {fpath}")
        da = xr.open_dataset(fpath).precipitation_24_hour

        # Mask data to CONUS only (since we're using AORC as truth)
        region_mask = create_conus_mask(da) 
        da = da.where(region_mask)
        da_dict[dataset_name] = da
    
    # Calculate FSS
    truth_dataset_name = "AORC"
    truth_da = da_dict[truth_dataset_name]

    """
    for dataset_name, da in da_dict.items():
        if (dataset_name == truth_dataset_name):
            continue

        # Calculate FSS
        FSS = calculate_fss(da, truth_da, args.eval_radius, args.grid_size, args.threshold)
        print(f"FSS for {dataset_name}: {FSS:0.3f}")
    """

    # Plot data
    if args.plot:
        #for dataset_name, da in da_dict.items():
            #pputils.plot_cmap_single_panel(da, dataset_name, "CONUS")

        # Plot FSS for single threshold for varying eval radii
        eval_radius_list = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
        plot_fss_versus_eval_radius(da_dict, eval_dt, eval_radius_list, args.grid_size, args.threshold)

        # Plot FSS for single eval radius for varying thresholds
        threshold_list = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0] # in mm
        plot_fss_versus_threshold(da_dict, eval_dt, threshold_list, args.eval_radius, args.grid_size)

    return da_dict

def create_conus_mask(data_array):
    # Define a regionmask object representing all 50 US states
    states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
    mask = states.mask(data_array)

    # Get the integers representing the non-CONUS states Alaska and Hawaii (don't want to include data over these states in CONUS stats) 
    AK_index = states.map_keys("Alaska")
    HI_index = states.map_keys("Hawaii")

    # Convert the mask to a boolean DataArray that is True over all the states except AK and HI
    # We'll use this boolean DataArray to pull out only data from the precip DataArrays within the lower-48 (CONUS) states 
    mask_conus = (mask != AK_index) & (mask != HI_index) & (mask >= 0) 

    return mask_conus

# Set data array to 1 at or above threshold, zero below it
# NOTE: this function takes an xarray DataArray as input and returns a numpy array.
def mask_data_array_based_on_threshold(da, threshold):
    # Take the arrays down to 2-D, removing the time dimension
    if (len(da.shape) == 3):
        da = da[0,:,:]

    return np.where(da >= threshold, 1, 0) 

# Create circular footprint for FSS calculation
def get_footprint(radius, grid_size):
    radius_over_grid_size = int(radius/grid_size)

    # In this step, we obtain a square of zeros (with size length, in grid squares, of radius/grid_size * 2 + 1)
    # circumscribing a circle of ones (with radius, in grid_squares, of radius/grid_size)
    footprint = (np.ones((radius_over_grid_size * 2 + 1, radius_over_grid_size * 2 + 1))).astype(int)
    footprint[int(math.ceil(radius_over_grid_size)), int(math.ceil(radius_over_grid_size))] = 0

    dist = scipy.ndimage.distance_transform_edt(footprint, sampling = [grid_size, grid_size])

    footprint = np.where(np.greater(dist, radius), 0, 1)

    return footprint

# FSS calculation from Craig Schwartz via Trevor Alcott
def calculate_fss(qpf, qpe, radius, grid_size, threshold):
    #print(f"Calculating FSS for radius {radius}, threshold {threshold}")

    # Calculate footprint, i.e., evaluation area
    footprint = get_footprint(radius, grid_size)

    # Convert qpf and qpe arrays to numpy arrays containing 1s and 0s based on
    # whether precipitation amount is at or above or below threshold   
    binary_qpf =  mask_data_array_based_on_threshold(qpf, threshold)
    binary_qpe = mask_data_array_based_on_threshold(qpe, threshold)

    # Calculate pf and po terms in the FSS formula (equivalent to M and O terms in Roberts and Lean (2008) Equations 5,7)
    # NOTE: Modification from the code received by trevor. I created masked 1/0 arrays above, before using to calculate pf/po.
    # He calculates the masked arrays in-line using np.where(np.greater_equal(qpf, threshold), 1, 0) 
    #pf = np.around(scipy.signal.fftconvolve(np.where(np.greater_equal(qpf, threshold), 1, 0), footprint, mode = "same"))/np.sum(footprint)
    #po = np.around(scipy.signal.fftconvolve(np.where(np.greater_equal(qpe, threshold), 1, 0), footprint, mode = "same"))/np.sum(footprint)
    pf = np.around(scipy.signal.fftconvolve(binary_qpf, footprint, mode = "same"))/np.sum(footprint)
    po = np.around(scipy.signal.fftconvolve(binary_qpe, footprint, mode = "same"))/np.sum(footprint)

    # Calculate gridsize (Nx * Ny)
    gridsize = np.shape(qpe)[0] * np.shape(qpe)[1]

    # Calculate numerator
    fbs = 1/gridsize * np.sum((pf - po)**2)

    # Calculate denominator
    fbs_worst = 1/gridsize * (np.sum(pf**2) + np.sum(po**2))

    if (fbs_worst > 0):
        fss = 1.0 - float(fbs)/float(fbs_worst)
    else:
        fss = np.nan 

    return fss

# Plot against varying evaluation radii
def plot_fss_versus_eval_radius(da_dict, valid_dtime, eval_radius_list, grid_size, threshold,
                                region = "CONUS", truth_dataset_name = "AORC"):
    # Create figure
    plt.figure(figsize = (12, 10))
    plt.title(f"FSS versus evaluation radius for threshold {threshold} mm", size = 15)
    plt.xlabel("Evaluation radius (degrees lat/lon)", size = 15)
    plt.ylabel("FSS", size = 15)
    plt.xlim(0, eval_radius_list[-1])
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(0, eval_radius_list[-1] + 1, 1), fontsize = 15)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize = 15) 
    plt.grid(True, linewidth = 1.5)

    truth_da = da_dict[truth_dataset_name]
    for dataset_name, da in da_dict.items():
        print(f"Working on dataset {dataset_name}")
        if (dataset_name == truth_dataset_name):
            continue

        fss_list = []
        for radius in eval_radius_list:
            # Calculate FSS
            FSS = calculate_fss(da, truth_da, radius, grid_size, threshold)
            fss_list.append(FSS)
            
        # Gather data to plot and plot data
        plt.plot(eval_radius_list, fss_list, linewidth = 1.5, label = dataset_name,
                 color = pputils.time_series_color_dict[dataset_name])

    # Save figure 
    plt.legend(loc = "best", prop = {"size": 15})
    data_names_str = "".join(f"{key}." for key in da_dict.keys())
    fig_name = f"demoFSS_versus_eval_radius.{data_names_str}thresh{threshold}mm.{valid_dtime:%Y%m%d.%H}.{region}.png"
    fig_path = os.path.join(utils.plot_output_dir, fig_name)
    print(f"Saving {fig_path}")
    plt.savefig(fig_path)

# Plot against varying precip amoun thresholds
def plot_fss_versus_threshold(da_dict, valid_dtime, threshold_list, eval_radius, grid_size,
                              region = "CONUS", truth_dataset_name = "AORC"):
    # Create figure
    plt.figure(figsize = (12, 10))
    plt.title(f"FSS versus precipitation amount threshold for radius {eval_radius:0.2f} degrees", size = 15)
    plt.xlabel("Threshold (mm)", size = 15)
    plt.ylabel("FSS", size = 15)
    plt.xlim(0, threshold_list[-1])
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(0, threshold_list[-1] + 10, 10), fontsize = 15)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize = 15) 
    plt.grid(True, linewidth = 1.5)

    truth_da = da_dict[truth_dataset_name]
    for dataset_name, da in da_dict.items():
        f"Working on dataset {dataset_name}"
        if (dataset_name == truth_dataset_name):
            continue

        fss_list = []
        for threshold in threshold_list:
            # Calculate FSS
            FSS = calculate_fss(da, truth_da, eval_radius, grid_size, threshold)
            fss_list.append(FSS)
            
        # Gather data to plot and plot data
        plt.plot(threshold_list, fss_list, linewidth = 1.5, label = dataset_name,
                 color = pputils.time_series_color_dict[dataset_name])

    # Save figure 
    plt.legend(loc = "best", prop = {"size": 15})
    data_names_str = "".join(f"{key}." for key in da_dict.keys())
    fig_name = f"demoFSS_versus_threshold.{data_names_str}eval_radius{eval_radius}deg.{valid_dtime:%Y%m%d.%H}.{region}.png"
    fig_path = os.path.join(utils.plot_output_dir, fig_name)
    print(f"Saving {fig_path}")
    plt.savefig(fig_path)

if __name__ == "__main__":
    da_dict = main()
