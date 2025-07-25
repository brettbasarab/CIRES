#!/bin/bash

source /home/bbasarab/shell/stats_list_for_external_data.sh
source /home/bbasarab/shell/regions_list.sh

# Set to 'true' to print verify_precip.py command below but not run it
TESTING=false

# Set to '--include_hrrr' to run verification for shorter time period and include HRRR
include_hrrr=""
if [ "$include_hrrr" = "--include_hrrr" ]; then
  date_range_str="20150101 20220101"
else
  date_range_str="20020101 20220101"
fi

# Set to '--poster' for larger poster fonts on plots
poster=""

for region in $regions_list
do
echo "***** $region"
cmd="verify_precip_external_data.py $date_range_str $stats_list $include_hrrr $poster_flag --region $region"
echo $cmd
if [ $TESTING == false ]; then
  $cmd
fi
done 
