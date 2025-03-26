#!/bin/bash

source /home/bbasarab/shell/regions_list.sh

# Set to '--write_to_nc' to write stats to netCDF files
export write_to_nc_flag="--write_to_nc"

# Set to '--plot' to make plots
export plot_flag="--plot"

# Set to "--poster" for larger poster fonts
export poster_flag=""

for region in $regions_list
do
echo "******** $region"
verify_precip.py 20020101.03 20211229.00 --fss $write_to_nc_flag $plot_flag $poster_flag --region $region >& ~/std_out/verify_precip.fss.${region}.out
done
