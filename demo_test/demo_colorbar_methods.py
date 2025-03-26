#!/usr/bin/env python

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as mpltkax

# Vertical color bar at right for 3-panel plot
fig = plt.figure(figsize = (12, 12))
proj = ccrs.PlateCarree()
axes_list = [
            plt.subplot2grid((2, 9), (0, 0), colspan = 4, rowspan = 1, projection = proj),
            plt.subplot2grid((2, 9), (0, 4), colspan = 4, rowspan = 1, projection = proj),
           plt.subplot2grid((2, 9), (1, 2), colspan = 4, rowspan = 1, projection = proj),
            ]
cbar_ax = plt.subplot2grid((2, 9), (0, 8), colspan = 1, rowspan = 2)

# COLORBAR METHOD using mpl toolkits: doesn't work but may not be what we want (adds a colorbar to each axis)
#divider = mpltkax.make_axes_locatable(axis)
#cax = divider.append_axes("right", size = "5%", pad = 0.05)
#plt.colorbar(plot_handle, cax = cax)
            
# COLORBAR METHOD standard to add single colorbar: doesn't work with tight_layout
#fig.subplots_adjust(right = 0.8)
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
#fig.colorbar(plot_handle, cax = cbar_ax)

