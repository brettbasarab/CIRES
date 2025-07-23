#!/bin/bash

source /home/bbasarab/shell/regions_list.sh

# Set to 'true' to print verify_precip.py command below but not run it
TESTING=false

# Set to '--cmaps' to plot contour maps
cmaps="--cmaps"

# Set to '--fss --fss_pctl --fss_ari' to calculate FSS for various thresholds/radii
fss="--fss --fss_pctl --fss_ari"

# Set to '--pdfs' to calculate probability density functions (PDFs)
pdfs="--pdfs"

# Set to '--timeseries' to plot time series
timeseries="--timeseries"

# Set to '--include_hrrr' to run verification for shorter time period and include HRRR
include_hrrr=""
if [ -z $include_hrrr ]; then
  date_range_str="20020101 20220101"
else
  date_range_str="20150101 20220101"
fi

# Set to '--plot' to make plots
plot="--plot"

# Set to '--write_to_nc' to write stats to netCDF files
write_to_nc="--write_to_nc"

# Set to '--poster' for larger poster fonts on plots
poster=""

for region in $regions_list
do
echo "******** $region"
cmd="verify_precip.py $date_range_str $include_hrrr $cmaps $fss $pdfs $timeseries $plot $write_to_nc $poster --region $region"
echo $cmd
if [ $TESTING == false ]; then
  $cmd >& ~/std_out/verify_precip.${region}.out
fi
done
