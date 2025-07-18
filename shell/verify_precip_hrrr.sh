#!/bin/bash

source /home/bbasarab/shell/regions_list.sh

for region in $regions_list
do
echo "******** $region"
verify_precip.py 20170101 20220101 --include_hrrr --fss --fss_pctl --fss_ari --pdfs --timeseries --plot --region $region >& ~/std_out/verify_precip_hrrr.${region}.out
done
