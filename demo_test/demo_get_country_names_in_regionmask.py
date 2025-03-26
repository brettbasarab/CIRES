#!/usr/bin/env python

import argparse
import regionmask

parser = argparse.ArgumentParser()
parser.add_argument("region")
args = parser.parse_args()

countries = regionmask.defined_regions.natural_earth_v5_1_2.countries_50
my_countries = [i.strip() for i in open(f"/home/bbasarab/list_of_countries_for_region_masking/{args.region}.csv").readlines()]

for i in my_countries: 
    try:
        x = countries.map_keys(i)
        #print(f"{i} key: {countries.map_keys(i)}")
    except KeyError:
        print(f"Error: country {i} not found")
