#!/bin/bash

# In a directory (must be in this directory), count number of files for each specified year (currently 2002-2024)
for year in 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21; do echo "******** YEAR: 20${year}"; ls *.20${year}*.nc | wc -l; done
