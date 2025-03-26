#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import precip_plotting_utilities as pputils
import precip_verification_processor
import xarray as xr

# Create an intance of precip verif class
data_names, truth_data_name = precip_verification_processor.map_region_to_data_names("CONUS") 
verif = precip_verification_processor.PrecipVerificationProcessor("20170801.03", 
                                                                  "20170901.00",
                                                                  LOAD_DATA_FLAG = True, 
                                                                  external_da_dict = None, 
                                                                  data_names = data_names, 
                                                                  truth_data_name = truth_data_name,
                                                                  region = "CONUS", 
                                                                  temporal_res = 24) 

# Collect individual data arrays
aorc = verif.da_dict["AORC"]
conus404 = verif.da_dict["CONUS404"]
era5 = verif.da_dict["ERA5"]
imerg = verif.da_dict["IMERG"]
replay = verif.da_dict["Replay"]

# Create and plot histograms with different bins capturing the full range of each dataset
aorc_hist = aorc.plot.hist()
conus404_hist = conus404.plot.hist()
era5_hist = era5.plot.hist() 
imerg_hist = imerg.plot.hist()
replay_hist = replay.plot.hist()

total_vals = aorc_hist[0].sum() 

aorc_pdf = aorc_hist[0]/total_vals
conus404_pdf = conus404_hist[0]/total_vals
era5_pdf = era5_hist[0]/total_vals
imerg_pdf = imerg_hist[0]/total_vals
replay_pdf = replay_hist[0]/total_vals

aorc_bins = aorc_hist[1]
conus404_bins = conus404_hist[1]
era5_bins = era5_hist[1]
imerg_bins = imerg_hist[1]
replay_bins = replay_hist[1]

plt.figure(figsize = (10,10))
plt.plot(aorc_bins[:-1], aorc_pdf, label = "AORC", color = pputils.time_series_color_dict["AORC"])
plt.plot(conus404_bins[:-1], conus404_pdf, label = "CONUS404", color = pputils.time_series_color_dict["CONUS404"])
plt.plot(era5_bins[:-1], era5_pdf, label = "ERA5", color = pputils.time_series_color_dict["ERA5"])
plt.plot(imerg_bins[:-1], imerg_pdf, label = "IMERG", color = pputils.time_series_color_dict["IMERG"])
plt.plot(replay_bins[:-1], replay_pdf, label = "Replay", color = pputils.time_series_color_dict["Replay"])
plt.gca().set_yscale("log")
plt.grid(True, linewidth = 0.5)
plt.legend(loc = "best")
plt.savefig("/home/bbasarab/plots/test_pdf.png")
        
# Create and plot histograms with standard set of bins used for all datasets
max_list = []
for data_name, da in verif.da_dict.items():
    da_max = da.max()
    max_list.append(da_max)
max_da = xr.concat(max_list, dim = "x")
overall_max = max_da.max().item()
standard_bins = np.arange(0, np.round(overall_max, decimals = -2) + 100, 100)
 
aorc_hist_standard_bins = aorc.plot.hist(bins = standard_bins)
conus404_hist_standard_bins = conus404.plot.hist(bins = standard_bins)
era5_hist_standard_bins = era5.plot.hist(bins = standard_bins)
imerg_hist_standard_bins = imerg.plot.hist(bins = standard_bins)
replay_hist_standard_bins = replay.plot.hist(bins = standard_bins)

aorc_pdf_standard_bins = aorc_hist_standard_bins[0]/total_vals
conus404_pdf_standard_bins = conus404_hist_standard_bins[0]/total_vals
era5_pdf_standard_bins = era5_hist_standard_bins[0]/total_vals
imerg_pdf_standard_bins = imerg_hist_standard_bins[0]/total_vals
replay_pdf_standard_bins = replay_hist_standard_bins[0]/total_vals

plt.figure(figsize = (10,10))
plt.plot(standard_bins[:-1], aorc_pdf_standard_bins, label = "AORC", color = pputils.time_series_color_dict["AORC"])
plt.plot(standard_bins[:-1], conus404_pdf_standard_bins, label = "CONUS404", color = pputils.time_series_color_dict["CONUS404"])
plt.plot(standard_bins[:-1], era5_pdf_standard_bins, label = "ERA5", color = pputils.time_series_color_dict["ERA5"])
plt.plot(standard_bins[:-1], imerg_pdf_standard_bins, label = "IMERG", color = pputils.time_series_color_dict["IMERG"])
plt.plot(standard_bins[:-1], replay_pdf_standard_bins, label = "Replay", color = pputils.time_series_color_dict["Replay"])
plt.gca().set_yscale("log")
plt.grid(True, linewidth = 0.5)
plt.legend(loc = "best")
plt.savefig("/home/bbasarab/plots/test_pdf_standard_bins.png")
