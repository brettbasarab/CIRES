#!/usr/bin/env python

import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr

# Single-dimensional array, dimensions with coordinates
data_size = 10
first_valid_dt = pd.Timestamp(2002, 1, 1)
data = np.random.randint(45, high = 55, size = data_size)
coords = pd.date_range(first_valid_dt, periods = data_size, freq = dt.timedelta(days = 1))
da = xr.DataArray(data, coords = [coords], dims = ["time"])

# Multi-dimensional array, dimensions without coordinates
da2d_nocoords = xr.DataArray([ [15, 12, 9, 1, 0], [12, 3, 4, 110, 10] ])

# Multi-dimensional array, dimensions with coordinates

# Multidimensional boolean array, without coordinates
da_bool = xr.DataArray([ [True, False, True, True], [False, False, True, True] ])

# To count the number of indices that are True (e.g., for precip stats calculations):
    # 1) Get the values of the DataArray as a numpy array using the .values attribute.
    # 2) Call flatten() method to collapse the numpy array to one dimension.
    # 3) Call np.where() to get an array of indices where the flattened array is True.
    # 4) Get the number of indices where True using the array's .shape attribute or by using len().
num_indices_where_true = len(np.where(da_bool.values.flatten())[0])
print(f"Number of indices where True: {num_indices_where_true}")
