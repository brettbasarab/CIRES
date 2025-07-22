#!/bin/bash

source /home/bbasarab/shell/regions_list.sh

# Set to '--cmaps' to plot contour maps
export cmaps="--cmaps"

# Set to '--fss --fss_pctl --fss_ari' to calculate FSS for various thresholds/radii
export fss="--fss --fss_pctl --fss_ari"

# Set to '--pdfs' to calculate probability density functions (PDFs)
export pdfs="--pdfs"

# Set to '--timeseries' to plot time series
export timeseries="--timeseries"

# Set to '--include_hrrr' to run verification for shorter time period and include HRRR
export include_hrrr=""
if [ -z $include_hrrr ]; then
  export date_range_str="20020101 20220101"
else
  export date_range_str="20150101 20220101"
fi

# Set to '--plot' to make plots
export plot="--plot"

# Set to '--write_to_nc' to write stats to netCDF files
export write_to_nc=""

# Set to '--poster' for larger poster fonts on plots
export poster=""

for region in $regions_list
do
echo "******** $region"
verify_precip.py $date_range_str $include_hrrr $cmaps $fss $pdfs $timeseries $plot $write_to_nc $poster --region $region >& ~/std_out/verify_precip.${region}.out
done
