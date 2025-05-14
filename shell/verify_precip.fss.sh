#!/bin/bash

source /home/bbasarab/shell/regions_list.sh

# Set to '--write_to_nc' to write stats to netCDF files
export write_to_nc_flag=""

# Set to '--plot' to make plots
export plot_flag="--plot"

# Set to "--poster" for larger poster fonts
export poster_flag=""

for region in $regions_list
do
echo "******** $region"
# FSS against amount thresholds
verify_precip.py 20020101 20211229 --fss $write_to_nc_flag $plot_flag $poster_flag --region $region >& ~/std_out/verify_precip.fss.${region}.out

# FSS against pctl thresholds
verify_precip.py 20020101 20211229 --fss --fss_pctl_threshold $plot_flag $poster_flag --region $region >& ~/std_out/verify_precip.fss_pctl.${region}.out

# FSS against ARI grids
verify_precip.py 20020101 20211229 --fss --fss_ari_grids $plot_flag $poster_flag --region $region >& ~/std_out/verify_precip.fss_ari.${region}.out 
done
