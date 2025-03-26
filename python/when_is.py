#!/usr/bin/env python

import argparse
import datetime as dt
import sys
import utilities as utils

# Convert a Unix time (seconds since 1970-01-01 00:00:00 UTC) to a Python datetime object (in UTC).
# Print out the datetime with the format YYYY-mm-dd HH:MM:SS

# TODO: Write some unit tests for this program to get practice with Python unit testing!
# I bet there are some creative ways to make this program fail.

def main():
    program_description = ("This program converts a Unix time (seconds since 1970-01-01 00:00:00 UTC) to a Python datetime object (in UTC) \
                            or it converts a date/time string in the format YYYY-mm-dd HH:MM:SS to Unix time. \
                            If converting from Unix time, it prints the datetime with the format YYYY-mm-dd HH:MM:SS. \
                            If converting to Unix time, it prints the Unix time as an integer.")
    parser = argparse.ArgumentParser(description = program_description) 
    parser.add_argument("input_time",
                        help = "Unix time (in seconds since the Unix epoch of 1970-01-01 00:00:00 UTC) or datetime with the format 'YYYY-mm-dd HH:MM:SS'",
                        type = str) 
    args = parser.parse_args()

    # Determine whether input argument is a unix time or datetime string.
    try:
        input_time = int(args.input_time)
        input_is_unix_time = True
    except ValueError:
        try:
            dtime = dt.datetime.strptime(args.input_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print(f"Error: Invalid datetime input format; must be {utils.full_date_format_str}") 
            sys.exit(1)
        input_is_unix_time = False 

    # Depending on the type of the input time, convert to/from Unix time. 
    if input_is_unix_time: 
        dtime = utils.unix2datetime(input_time)
        print(f"{dtime:%Y-%m-%d %H:%M:%S}")
    else:
        unix_time = utils.datetime2unix(dtime)
        print(unix_time)

if __name__ == "__main__":
    main()
