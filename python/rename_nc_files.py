#!/usr/bin/env python

import argparse
import glob
import os
import sys
import utilities as utils

def main():
    parser = argparse.ArgumentParser(description = f"Program to rename files in {utils.data_nc_dir} subdirectories")
    parser.add_argument("dir_name",
                        help = f"Directory within {utils.data_nc_dir} in which to rename files (program will cd into this directory")
    parser.add_argument("old_str",
                        help = "Old common string in file names to replace")
    parser.add_argument("new_str",
                        help = "New string to use to replace old_str in file names")
    parser.add_argument("-t", "--testing", dest = "testing", default = False, action = "store_true",
                        help = "Testing mode: print but don't run mv commands")
    args = parser.parse_args()

    dir_name = os.path.join(utils.data_nc_dir, args.dir_name)
    if not(os.path.exists(dir_name)):
        print(f"Error: Directory {args.dir_name} in which to rename files does not exist")
        sys.exit(1)

    os.chdir(dir_name)
    print(f"Current working directory is {os.getcwd()}")

    flist = sorted(glob.glob("*.nc"))
    if (len(flist) == 0):
        print(f"No files found in {dir_name} to rename")
        sys.exit(0)

    for fold in flist:
        #fnew = fold.replace("_mean", ".mean").replace("_9", ".9")
        fnew = fold.replace(args.old_str, args.new_str)

        print(f"Old name: {fold}")
        print(f"New name: {fnew}")
        cmd = f"mv {fold} {fnew}"
        print(cmd)
        print("\n")

        if not(args.testing):
            os.system(cmd)

if __name__ == "__main__":
    main()
