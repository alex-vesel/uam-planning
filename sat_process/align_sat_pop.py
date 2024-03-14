
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from rasterio.crs import CRS
import rioxarray as rxr
import rasterio
import xarray as xr
import cv2

area_name = "sf"
area_name = "nyc"

# open sat_img.tif using rxr and convert to 4326
sat_img = rxr.open_rasterio(f'{area_name}_sat.tif')
sat_img = sat_img.rio.reproject('EPSG:4326')

pop_density_data = rxr.open_rasterio('us_population.tif')

if area_name == "sf":
    sat_img = sat_img.rio.clip_box(-122.55, 37, -121.6, 38)
    pop_density = pop_density_data.rio.clip_box(-125, 37, -121, 38)
elif area_name == "nyc":
    sat_img = sat_img.rio.clip_box(-74.1, 40.6, -73.8, 40.9)
    pop_density = pop_density_data.rio.clip_box(-74.1, 40.6, -73.8, 40.9)

pop_density = pop_density.rio.reproject_match(sat_img)

# get area of each pixel in square kilometers
pop_density_meters = pop_density.rio.reproject('EPSG:3857')
area_m = abs(pop_density_meters.rio.transform().a) * abs(pop_density_meters.rio.transform().e)
area_km = area_m / 1e6

# get sf and pop_density data as numpy
sat_img = sat_img.values
# turn blue green red to red green blue
sat_img = sat_img[[2, 1, 0], :, :]
sat_img = sat_img.transpose(1, 2, 0)

pop_density = pop_density.values
pop_density = pop_density.transpose(1, 2, 0)
pop_density[pop_density < 0] = 0
pop = (pop_density * area_km)

# stack and save
sat_img_pop = np.concatenate((sat_img, pop, pop_density), axis=2)

# save 5 channel using np
np.save(f'{area_name}_sat_pop.npy', sat_img_pop)