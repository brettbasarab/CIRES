#!/usr/bin/env python

import numpy as np
from score_hv import harvester_base

def main():
    config_fpath = "/home/bbasarab/harvester_config.yml"
    print(f"Config file: {config_fpath}")
    harvested_data_list = harvester_base.harvest(config_fpath)
    print(f"Files being considered: {harvested_data_list[0].filenames}")

    current_var = "" 
    for harvested_data in harvested_data_list:
        #print(harvested_data)
        if harvested_data.variable != current_var:
            print(f"******* Variable name: {harvested_data.longname}")
        print(f"{harvested_data.statistic}: {harvested_data.value:0.2f} {harvested_data.units}")
        current_var = harvested_data.variable

if __name__ == "__main__":
    main()
