#!/usr/bin/env python

import argparse
import datetime as dt
import utilities as utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("valid_dt_str", help = "Valid date-time; format YYYYmmdd.HHMM")
    args = parser.parse_args()

    valid_dt = dt.datetime.strptime(args.valid_dt_str, "%Y%m%d.%H%M")
    minutes_into_day = construct_minutes_into_day_string(valid_dt)
    print(f"Valid_dt: {valid_dt}; minutes into day: {minutes_into_day}")

    return valid_dt
    
# Number of minutes into the day (so from 0000-1410) is needed
# since it's part of the file name. 
def construct_minutes_into_day_string(valid_dt):
    valid_dt_beginning_of_day = dt.datetime(valid_dt.year, valid_dt.month, valid_dt.day, 0, 0)
    valid_seconds_into_day = utils.datetime2unix(valid_dt) - utils.datetime2unix(valid_dt_beginning_of_day)
    print(f"Valid datetime seconds into day: {valid_seconds_into_day}")
    valid_minutes_into_day = int(valid_seconds_into_day/60.)
    print(f"Valid datetime minutes into day: {valid_minutes_into_day}")
    valid_minutes_into_day_str = f"{valid_minutes_into_day:04d}"
    
    #print(f"valid_dt in seconds: {utils.datetime2unix(valid_dt)}")
    #print(f"valid_date in seconds: {utils.datetime2unix(valid_date)}")

    return valid_minutes_into_day_str

if __name__ == "__main__":
    valid_dt = main()
