#!/bin/bash

source /home/bbasarab/shell/stats_list_for_external_data.sh
source /home/bbasarab/shell/regions_list.sh

# Set to "--poster" for larger poster fonts
export poster_flag="--poster"

for region in $regions_list
do
echo "***** $region"
verify_precip_external_data.py 20020101.03 20211229.00 $stats_list --region $region $poster_flag
done 
