#!/bin/bash

source /home/bbasarab/shell/regions_list.sh

export stats_list="full_period.pdf common_seasonal.pdf"

# Set to "--poster" for larger poster fonts
export poster_flag="--poster"

for region in $regions_list
do
echo "***** $region"
verify_precip_distributions_external_data.py 20020101.03 20211229.00 $stats_list $poster_flag --region $region
done 
