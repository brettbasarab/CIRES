#!/bin/bash

# Set to 'true' to print verify_precip.py command below but not run it
TESTING=false

# Set to '--cmaps [mean, pctls, all]' to plot contour maps
cmaps="--cmaps all"

# Set to '--fss --fss_pctl --fss_ari' to calculate FSS for various thresholds/radii
fss="--fss --fss_pctl --fss_ari"

# Set to '--pdfs' to calculate probability density functions (PDFs)
pdfs="--pdfs"

# Set to '--timeseries' to plot time series
timeseries="--timeseries"

# Set to '--include_hrrr' to run verification for shorter time period and include HRRR
include_hrrr="--include_hrrr"
if [ "$include_hrrr" = "--include_hrrr" ]; then
  date_range_str="20150101 20220101"
else
  date_range_str="20020101 20220101"
fi

# Set to '--plot' to make plots
plot="--plot"

# Set to '--write_to_nc' to write stats to netCDF files
write_to_nc="--write_to_nc"

# Set to '--poster' for larger poster fonts on plots
poster=""

# Set list of regions to verify
regions_list=`regions_list.sh`

cmd="verify_precip.py $date_range_str $include_hrrr $cmaps $fss $pdfs $timeseries $plot $write_to_nc $poster --regions $regions_list"
echo $cmd
if [ $TESTING == false ]; then
  $cmd >& ~/std_out/verify_precip.out
fi
