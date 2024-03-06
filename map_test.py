
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from rasterio.crs import CRS
import rioxarray as rxr
import rasterio
import xarray as xr
import cv2

# open sf_sat.tif using rxr and convert to 4326
sf_sat = rxr.open_rasterio('sf_sat.tif')
sf_sat = sf_sat.rio.reproject('EPSG:4326')
sf_sat = sf_sat.rio.clip_box(-122.55, 37, -121.6, 38)

pop_density_data = rxr.open_rasterio('us_population.tif')
pop_density = pop_density_data.rio.clip_box(-125, 37, -121, 38)
pop_density = pop_density.rio.reproject_match(sf_sat)

# get area of each pixel in square kilometers
pop_density_meters = pop_density.rio.reproject('EPSG:3857')
area_m = abs(pop_density_meters.rio.transform().a) * abs(pop_density_meters.rio.transform().e)
area_km = area_m / 1e6

# get sf and pop_density data as numpy
sf_sat = sf_sat.values
# turn blue green red to red green blue
sf_sat = sf_sat[[2, 1, 0], :, :]
sf_sat = sf_sat.transpose(1, 2, 0)

pop_density = pop_density.values
pop_density = pop_density.transpose(1, 2, 0)
pop_density[pop_density < 0] = 0
pop = (pop_density * area_km)

# stack and save
sf_sat_pop = np.concatenate((sf_sat, pop, pop_density), axis=2)
# save 5 channel using np
np.save('sf_sat_pop.npy', sf_sat_pop)

pop = pop.squeeze()

# sample locations in proportion to pop_densityulation
pop_prob = pop / np.sum(pop)
pop_prob = pop_prob/ np.sum(pop_prob)
pop_prob = pop_prob.flatten()

# sample
n_samples = 30

def get_circle(x, y, radius, max_x, max_y):
    circle = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i**2 + j**2 <= radius**2 and 0 <= x + i < max_x and 0 <= y + j < max_y:
                circle.append((x + i, y + j))
    return np.array(circle)

pop_per_vp = np.sum(pop) / n_samples
print(pop_per_vp)

coords = np.zeros((2, n_samples))
# sample based on pop_density_prob, then zero out region with 10,000 pop_densityulation and sample again
for i in range(n_samples):
    sample_locs = np.random.choice(np.arange(pop_density.size), size=1, p=pop_prob)
    x, y, _ = np.unravel_index(sample_locs, pop_density.shape)
    x, y = x[0], y[0]
    coords[0, i], coords[1, i] = x, y
    # zero out region near coords until pop_density, grow largest radius
    radius = 50
    circle = get_circle(x, y, radius, pop_density.shape[0], pop_density.shape[1])
    while np.sum(pop[circle[:, 0], circle[:, 1]]) < pop_per_vp and radius < 150:
        radius += 15
        circle = get_circle(x, y, radius, pop_density.shape[0], pop_density.shape[1])
    print(radius)
    pop[circle[:, 0], circle[:, 1]] = 0

    pop_prob = pop / np.sum(pop)
    pop_prob = pop_prob/ np.sum(pop_prob)
    pop_prob = pop_prob.flatten()



# plot on top of each other using pop_density as heatmap
fig, ax = plt.subplots()
ax.imshow(sf_sat)
# ax.imshow(pop_density, alpha=0.5)
# don't show pop_density=0 on heatmap
# plt.imshow(pop_density, cmap='hot', alpha=0.6)
plt.scatter(coords[1], coords[0], c='r', s=10)
plt.show()
