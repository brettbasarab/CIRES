#!/bin/bash

source /home/bbasarab/shell/regions_list.sh

export data_names_flag=""
export temporal_res_flag="--temporal_res 24"
export stats_flags="--cmaps --timeseries"
export write_to_nc_flag="--write_to_nc" # set to '--write_to_nc' to write stats to netCDF files
export plot_flag="" # Set to '--plot' to make plots

for region in $regions_list
do
echo "******** $region"
verify_precip.py 20020101.03 20211229.00 $data_names_flag $temporal_res_flag $stats_flags $write_to_nc_flag $plot_flag --region $region >& ~/std_out/verify_precip.cmaps_timeseries.${region}.out
done
