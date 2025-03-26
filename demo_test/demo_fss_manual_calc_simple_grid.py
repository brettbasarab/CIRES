#!/usr/bin/env python

import argparse
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# TODO: Try larger 4x4 grids still with an eval radius of 1 (so an eval area of 9)
# What FSS values do these produce. For example, if the windows with valid data for fcst
# and obs never overlap at all, is FSS zero?

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--eval_radius", dest = "eval_radius", type = int, default = 1,
                        help = "Evaluation radius; default 1 grid square")
    parser.add_argument("-t", "--threshold", dest = "threshold", type = int, default = 1,
                        help = "Threshold; default = 1.0")
    parser.add_argument("--desired_fss", dest = "desired_fss", type = float, default = 0.62,
                        help = "Desired FSS: Fcst and obs grid will be modified accordingly to produce this FSS value; default 0.62") 
    args = parser.parse_args()

    # Calculate evaluation area
    eval_area = calculate_eval_area(args.eval_radius)

    # Initialize forecast and obs arrays with all zeros
    num_x = 3
    num_y = 3
    fcst = np.zeros((num_y, num_x))
    obs = np.zeros((num_y, num_x))

    # Populate arbitrary fcst and obs grid points with 1.
    fcst, obs = create_grids_for_desired_fss(fcst, obs, args.desired_fss) 

    # Loop through every grid point
    fss_numerator = 0.0
    fss_denominator = 0.0
    for j in range(num_y):
        for i in range(num_x):
            # At each grid point, examine all the neighbors within the evaluation radius
            # Count points in the neighborhood above a threshold
            count_fcst, count_obs = count_points_in_neighborhood_above_threshold(fcst, obs, j, i,
                                                                                 eval_radius = args.eval_radius,
                                                                                 threshold = args.threshold)

            # Calculate fractions for grid point (i, j)
            fcst_fraction = count_fcst/eval_area
            obs_fraction = count_obs/eval_area

            # Add to numerator and denominator of FSS term 
            fss_numerator += (obs_fraction - fcst_fraction)**2 
            fss_denominator += obs_fraction**2 + fcst_fraction**2
 
            print(f"j: {j}, i: {i}, fcst_fraction: {fcst_fraction}, obs_fraction: {obs_fraction}")
            print(f"Numerator: {fss_numerator}")
            print(f"Denominator: {fss_denominator}")

    # Final FSS calculation 
    FSS = 1 - fss_numerator/fss_denominator
    print(f"FSS: {FSS}")

    plot_fcst_and_obs(fcst, obs, FSS)

def calculate_eval_area(eval_radius):
    return (2 * eval_radius + 1)**2

def count_points_in_neighborhood_above_threshold(fcst, obs, j, i, eval_radius = 1.0, threshold = 1.0):
    count_fcst = 0
    count_obs = 0

    num_y = fcst.shape[0]
    num_x = fcst.shape[1]
    
    for jdx in range(j - eval_radius, j + eval_radius + 1):
        for idx in range(i - eval_radius, i + eval_radius + 1):
            #print(f"j: {j}, i: {i}, jdx: {jdx}, idx: {idx}")
            # If clause here to ensure the points we consider are inside the grid
            if (idx >= 0) and (idx < num_x) and (jdx >= 0) and (jdx < num_y): 
                # Count forecasts above threshold 
                val_fcst = fcst[jdx, idx]
                if (val_fcst >= threshold):
                    count_fcst += 1

                # Count obs above threshold
                val_obs = obs[jdx, idx] 
                if (val_obs >= threshold):
                    count_obs += 1

    #print(f"j: {j}, i: {i}, count_fcst: {count_fcst}, count_obs: {count_obs}")
    return count_fcst, count_obs

def plot_fcst_and_obs(fcst, obs, FSS):
    fig = plt.figure(figsize = (10,5))
    axes_list = [
                plt.subplot2grid((5, 4), (0, 0), colspan = 2, rowspan = 4), 
                plt.subplot2grid((5, 4), (0, 2), colspan = 2, rowspan = 4)
                ]
    cbar_ax = plt.subplot2grid((5,4), (4,1), colspan = 2, rowspan = 1) 
    
    fcst_plot = axes_list[0].imshow(fcst, cmap = "viridis", vmin = 0.0, vmax = 1.0)
    axes_list[0].set_title("Forecast", fontsize = 15)
    obs_plot = axes_list[1].imshow(obs, cmap = "viridis", vmin = 0.0, vmax = 1.0)
    axes_list[1].set_title("Observations", fontsize = 15) 
    
    axes_list[0].set_xticks(range(0, fcst.shape[1], 1))
    axes_list[0].set_yticks(range(0, fcst.shape[0], 1))
    axes_list[1].set_xticks(range(0, fcst.shape[1], 1))
    axes_list[1].set_yticks(range(0, fcst.shape[0], 1))
    
    fig.suptitle(f"Forecast and obs plots; FSS = {FSS:0.2f}", fontsize = 15)
    fig.colorbar(fcst_plot, cax = cbar_ax, orientation = "horizontal", shrink = 0.4) 
    plt.tight_layout()
    fig_name = f"/home/bbasarab/plots/demo_fss.FSS{FSS:0.2f}.png"
    print(f"Saving {fig_name}")
    plt.savefig(fig_name)

# Functions to "manufacture" various FSS values for demo purposes
# and to ensure my FSS code above is correct!
# Take in fcst and obs arrays populated with all zeros and insert 1s
# such that fss will be the resulting value
# NOTE: Will only work for evaluation radius of 1 
def create_grids_for_desired_fss(fcst, obs, desired_fss):
    if (desired_fss == 1.0): # Grids that exactly match; should give FSS of 1
        fcst[0,0] = 1
        fcst[1,1] = 1
        fcst[1,2] = 1
        fcst[2,2] = 1
        obs[0,0] = 1
        obs[1,1] = 1
        obs[1,2] = 1
        obs[2,2] = 1
    elif (desired_fss == 0.62):
        fcst[0,0] = 1
        obs[1,1] = 1
    elif (desired_fss == 0.25):
        fcst[0,0] = 1
        obs[2,2] = 1
    elif (desired_fss == 0.0):
        obs[1,1] = 1 
    else: # FSS of ~0.94
        fcst[0,0] = 1
        fcst[1,1] = 1
        fcst[1,2] = 1
        fcst[2,2] = 1
        obs[0,1] = 1
        obs[0,2] = 1
        obs[1,1] = 1
        obs[2,0] = 1

    return fcst, obs

if __name__ == "__main__":
    main()
