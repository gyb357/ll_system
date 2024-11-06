# %%
# follium 설치
# %pip install folium

# %%
import folium
from folium.plugins import HeatMap

wku_loc = (35.967835, 126.957068)
m = folium.Map(location=wku_loc, zoom_start=16)
m

# %%
folium.Marker(wku_loc, popup='WKU').add_to(m)

# %%
# kagglehub 설치
# %pip install kagglehub

# %%
import kagglehub

path = kagglehub.dataset_download('kimjmin/seoul-metro-usage')

# %%
import pandas as pd

data = pd.read_csv('dataset/seoul-metro-2015.logs.csv')
data.info()
data.head()

station_info = pd.read_csv('dataset/seoul-metro-station-info.csv', nrows=0)
station_info.info()
station_info.head()

# %%
station_sum = data.groupby('station_code').sum()
station_sum

# %%
joined_data = station_sum.join(station_info)
joined_data.head()
joined_data.info()

# %%
seoul_in = folium.Map(location=(37.55, 126.98), zoom_start=12)
seoul_in

# %%
# HeatMap(data=joined_data['geo.latitude', 'geo.longitude', 'people_in']).add_to(seoul_in)
