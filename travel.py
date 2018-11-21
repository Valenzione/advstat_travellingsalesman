from math import asin, cos, radians, sin, sqrt

import geopandas as gpd
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from descartes import PolygonPatch
from IPython.display import HTML
from matplotlib import animation, rc
from matplotlib.collections import PatchCollection
from shapely.geometry import MultiPolygon, Point, Polygon



# Randomly swap two items in an array
def permute(order):
    temp_order = order.copy()
    indexes = np.arange(len(temp_order))
    np.random.shuffle(indexes)
    a, b = indexes[:2]
    temp = temp_order[a]
    temp_order[a] = temp_order[b]
    temp_order[b] = temp
    return temp_order

# Compute distance between 2 points given in lat, lonh by Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r

# Compute total distance for given permutation of cities
def dist(order, xs, ys):
    x, y = xs[order[0]], ys[order[0]]
    cum_dist = 0
    for i in order[1:]:
        cum_dist += haversine(x, y, xs[i], ys[i])
        x, y = xs[i], ys[i]
    return cum_dist

# Simulated annealing
def sa(xs, ys, T, alpha, iterations):
    order = np.arange(0, len(xs)) # initial order
    np.random.shuffle(order)
    distance = dist(order, xs, ys) 
    orders = [order]
    distances = [distance]
    temperatures = [T]
    for i in range(iterations):
        new_order = permute(order)
        thresh = np.random.uniform() 
        new_distance = dist(new_order, xs, ys)
        d = (new_distance - distance) / distance;
        # if new distance is better, take it, else compare np.exp( - d / T)  to np.random.uniform() 
        p = 1 if d < 0 else np.exp( - d / T) 
        if p >= thresh:
            order = new_order
            distance = new_distance
        T = T * alpha # decrease temperature
        distances.append(distance)
        orders.append(order)
        temperatures.append(T)
    return orders, distances, temperatures

# Load the population data
data = pd.read_csv("cities.csv")
data['Население'] = pd.to_numeric(data['Население'], errors='coerce')
data.sort_values("Население", ascending=False, inplace=True)
data = data.iloc[:30]
data.iloc[0, 6] = "Москва"
data.iloc[1, 6] = "Санкт-Петербург"
data = data[["Город", "Широта", "Долгота", "Население"]]

# Load geojson of russian regions
regions_df = gpd.read_file("geo.json")

# Convert each region to Polygon patch to draw later
all_patches = []
for i, r in regions_df.iterrows():
    p = r['geometry']
    if isinstance(p, MultiPolygon):
        mp = p
        patches = []
        for p in mp:
            patches.append(PolygonPatch(p, fc='blue', lw=1, alpha=0.2))
        all_patches.extend(patches)
    else:   
        patches = [PolygonPatch(p, fc='blue', lw=1, alpha=0.2)]
        all_patches.extend(patches)

# Launch SA
ys = np.array(data['Широта'])
xs = np.array(data['Долгота'])
names = np.array(data['Город'])
paths, dists, temps = sa(xs, ys, 10000, 0.98, 2000)

# First set up the figure, the axis, and the plot element we want to animate
ax = plt.subplot(211)
axT = plt.subplot(223)
axD = plt.subplot(224)
fig = plt.gcf()
fig.set_size_inches(20, 9)

# Set limits of all axes
ax.set_xlim((15, 185))
ax.set_ylim((40, 90))
ax.set_title("Salesman path")
axT.set_xlim((0, len(paths) + 1))
axT.set_ylim((0, np.max(temps)*1.1)) 
axT.set_title("Annealing temperature")
axD.set_xlim((0, len(paths) + 1))
axD.set_ylim((0, np.max(dists) + 200))
axD.set_title("Total distance")

# Initialize line plots on all axes
line_path, = ax.plot([], [], c = "r", alpha = 0.75)
line_temp, = axT.plot([], [], c = "blue")
line_dist, = axD.plot([], [], c = "blue")

# initialization function: plot the background of each frame
def init():
    ax.add_collection(PatchCollection(all_patches, match_original=True))
    ax.scatter(xs, ys, s = 30, alpha = 0.3, c="red")
    texts = [ax.text(x, y, name) for (x, y, name) in zip(xs, ys, names)]
    _ = adjust_text(texts)
    line_path.set_data([], [])
    line_temp.set_data([], [])
    line_dist.set_data([], [])
    return line_path, line_temp, line_dist, 

# animation function. This is called sequentially to update each frame
def animate(i):
    o = paths[i]
    x = xs[o]
    y = ys[o]
    line_path.set_data(x, y)
    line_temp.set_data(range(0, i), temps[:i])
    line_dist.set_data(range(0, i), dists[:i])
    return line_path, line_temp, line_dist, 

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames = len(paths),
                               interval = 1, blit=True)
plt.tight_layout()

anim.save('animation.mp4', writer='ffmpeg', fps=30)

