
# coding: utf-8

# In[615]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import geopandas as gpd
from matplotlib.collections import PatchCollection
from shapely.geometry import Point, Polygon, MultiPolygon
from descartes import PolygonPatch
from adjustText import adjust_text
from matplotlib import animation, rc
from IPython.display import HTML
from math import radians, cos, sin, asin, sqrt


rc('animation', html='html5')
plt.rcParams['figure.figsize'] = (10, 5)


# In[174]:


data = pd.read_csv("cities.csv")
data['Население'] = pd.to_numeric(data['Население'], errors='coerce')
data.sort_values("Население", ascending=False, inplace=True)
data = data.iloc[:30]
data.iloc[0, 6] = "Москва"
data.iloc[1, 6] = "Санкт-Петербург"
data = data[["Город", "Широта", "Долгота", "Население"]]


# In[595]:


def permute(order):
    temp_order = order.copy()
    indexes = np.arange(len(temp_order))
    np.random.shuffle(indexes)
    a, b = indexes[:2]
    temp = temp_order[a]
    temp_order[a] = temp_order[b]
    temp_order[b] = temp
    return temp_order

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r

def dist(order, xs, ys):
    x, y = xs[order[0]], ys[order[0]]
    cum_dist = 0
    for i in order[1:]:
        cum_dist += haversine(x, y, xs[i], ys[i])
        x, y = xs[i], ys[i]
    return cum_dist
    
def p(order, xs, ys, T):
    distance = dist(order, xs, ys)
    return np.exp(- distance / T)

def sa(xs, ys, T, alpha, iterations):
    order = np.arange(0, len(xs))
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
        p = 1 if d < 0 else np.exp( - d / T)
        if p >= thresh:
            order = new_order
            distance = new_distance
        T = T * alpha
        distances.append(distance)
        orders.append(order)
        temperatures.append(T)
    return orders, distances, temperatures


# In[601]:


paths, dists, temps = sa(xs, ys, 10000, 0.95, 10000)


# In[607]:


# fig, axs = plt.subplots()
# all_patches = []
# for i, r in regions_df.iterrows():
#     p = r['geometry']
#     if isinstance(p, MultiPolygon):
#         mp = p
#         patches = []
#         for p in mp:
#             patches.append(PolygonPatch(p, fc='blue', lw=1, alpha=0.2))
#         all_patches.extend(patches)
#     else:   
#         patches = [PolygonPatch(p, fc='blue', lw=1, alpha=0.2)]
#         all_patches.extend(patches)

# axs.add_collection(PatchCollection(all_patches, match_original=True))


# import pyproj as proj


# ys = np.array(data['Широта'])
# xs = np.array(data['Долгота'])
# ms = 20 * np.log10(data['Население'])
# names = data['Город']
# texts = [plt.text(x, y, name) for (x, y, name) in zip(xs, ys, names)]

# plt.scatter(xs, ys, s = 20, c="r", alpha = 0.5)

# plt.plot(xs[paths[-1]], ys[paths[-1]])

# plt.xlim(15, 185)
# plt.ylim(40, 90)
# adjust_text(texts)

# plt.show()


# In[620]:


# First set up the figure, the axis, and the plot element we want to animate

ax = plt.subplot(211)
axT = plt.subplot(223)
axD = plt.subplot(224)
fig = plt.gcf()

ax.set_xlim((15, 185))
ax.set_ylim((40, 90))
ax.set_title("Salesman path")
axT.set_xlim((0, len(paths) + 1))
axT.set_ylim((0, np.max(temps)*1.1)) 
axT.set_title("Annealing temperature")
axD.set_xlim((0, len(paths) + 1))
axD.set_ylim((0, np.max(dists) + 200))
axD.set_title("Total distance")

line_path, = ax.plot([], [], c = "r")
line_temp, = ax.plot([], [], c = "blue")
line_dist, = ax.plot([], [], c = "blue")

pass

# initialization function: plot the background of each frame
def init():
    ax.add_collection(PatchCollection(all_patches, match_original=True))
    ys = data['Широта']
    xx = data['Долгота']
    names = data['Город']
    ax.scatter(xs, ys, s = 30, alpha = 0.2, c="red")
    line_path.set_data([], [])
    return (line_path, line_temp, line_dist, )

# animation function. This is called sequentially
def animate(i):
    ys = data['Широта']
    xx = data['Долгота']
    names = data['Город']
    texts = [ax.text(x, y, name) for (x, y, name) in zip(xs, ys, names)]
    o = paths[i]
    x = xs[o]
    y = ys[o]
    _ = adjust_text(texts, x=x, y=y)
    line_path.set_data(x, y)
    line_temp.set_data(range(i), temps[:i])
    line_dist.set_data(range(i), dists[:i])
    
    return (line_path, line_temp, line_dist, )


# In[621]:


# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=2, interval=200, blit=False)


# In[622]:


anim

