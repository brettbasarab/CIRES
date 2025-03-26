#!/usr/bin/env python

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("raw_overall_max", type = float,
                    help = "Raw overall max")
parser.add_argument("-r", "--round_decimal", dest = "round_decimal", type = int, default = 0,
                    help = "Decimal to round at in np.round; default 0")
parser.add_argument("-s", "--step_size", dest = "step_size", type = int, default = 2,
                    help = "Step size; default 2")
args = parser.parse_args()

print(f"Raw overall max: {args.raw_overall_max}")

overall_max = int(np.round(args.raw_overall_max, decimals = args.round_decimal) + args.step_size)
overall_max -= overall_max % args.step_size
print(f"Overall max: {overall_max}")

levels =  np.arange(0, overall_max + args.step_size, args.step_size)
print("Levels:")
print(levels) 
