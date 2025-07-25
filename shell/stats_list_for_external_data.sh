#!/bin/bash

####################
# CONTOUR MAPS: full period, common seasonal, common monthly ALL stats; TIMESERIES: monthly, seasonal
export stats_list=\
"full_period.mean.time full_period.95.0th_pctl.time full_period.99.0th_pctl.time full_period.99.9th_pctl.time \
common_seasonal.mean.time common_seasonal.95.0th_pctl.time common_seasonal.99.0th_pctl.time common_seasonal.99.9th_pctl.time \
common_monthly.mean.time common_monthly.95.0th_pctl.time common_monthly.99.0th_pctl.time common_monthly.99.9th_pctl.time \
monthly.mean.space_time seasonal.mean.space_time"

# CONTOUR MAPS: full period, common seasonal, common monthly means; TIMESERIES: monthly, seasonal
#export stats_list="full_period.mean.time common_seasonal.mean.time common_monthly.mean.time monthly.mean.space_time seasonal.mean.space_time"

# CONTOUR MAPS: full period, common seasonal, common monthly means
#export stats_list="full_period.mean.time common_seasonal.mean.time common_monthly.mean.time"

# CONTOUR MAPS: full period, common seasonal means
#export stats_list="full_period.mean.time common_seasonal.mean.time"

# TIMESERIES: monthly and seasonal
#export stats_list="monthly.mean.space_time seasonal.mean.space_time"

# CONTOUR MAPS: full period, common seasonal means; TIMESERIES: monthly and seasonal
#export stats_list="full_period.mean.time common_seasonal.mean.time monthly.mean.space_time seasonal.mean.space_time"

echo $stats_list
