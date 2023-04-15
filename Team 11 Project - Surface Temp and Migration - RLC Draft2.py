#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import mapclassify
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


px.set_mapbox_access_token('pk.eyJ1IjoicmFjaGVsY2xhcmsiLCJhIjoiY2xmaWIwamR3MWV1ejQzbThzbXA3Mnk1OCJ9.qT_XvaHKFUdh-MESwYgnSA')


# In[3]:


MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoicmFjaGVsY2xhcmsiLCJhIjoiY2xmaWIwamR3MWV1ejQzbThzbXA3Mnk1OCJ9.qT_XvaHKFUdh-MESwYgnSA'


# In[4]:


temperature = pd.read_csv('global_temp_data.csv', encoding='ISO-8859-1')
temperature.replace(to_replace=[-9999, -99], value=np.nan, inplace=True)
temperature.fillna(temperature.mean(), inplace=True)
temperature = temperature.round(2)


# In[5]:


temperature_numeric = temperature.select_dtypes(include=[np.number])
temperature_numeric = temperature_numeric[(temperature_numeric > -2) & (temperature_numeric < 2)]


# In[6]:


temperature


# In[7]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# In[8]:


temperature['name'] = temperature['Country'].str.title()
merged = world.merge(temperature, on='name')


# In[9]:


melted = merged.melt(
    id_vars=['Continent', 'Country', 'Subregion', 'geometry'],
    var_name='year',
    value_name='temp_anomaly'
)


# In[10]:


melted['temp_anomaly'] = pd.to_numeric(melted['temp_anomaly'], errors='coerce')


# In[11]:


ax = melted.plot(column='temp_anomaly', cmap='viridis_r', figsize=(12, 10), scheme='quantiles', legend=True)
ax.set_title('Global Temperature Anomalies')
plt.show()


# In[12]:


emperor = pd.read_csv('EmperorMigrationRough.csv')
emperor.loc[emperor["Identifier"] != "J91_2003_YK3_5845", "Identifier"] = "J91_2003_YK3_5845"
emperor['Name'] = emperor['Name'].replace('Chen canagica', 'Emperor Goose')
emperor


# In[13]:


hawk = pd.read_csv('HawkMigrationRough.csv')
hawk.loc[hawk["Identifier"] != "71526a", "Identifier"] = "71526a"
hawk['Name'] = hawk['Name'].replace('Aquila chrysaetos', 'Golden Eagle')
hawk


# In[14]:


puffin=pd.read_csv('PuffinMigrationRough.csv')
puffin.loc[puffin["Identifier"] != "z", "Identifier"] = 3
puffin['Name'] = puffin['Name'].replace('Fratercula arctica', 'Atlantic Puffin')
puffin


# In[15]:


herring=pd.read_csv('HerringMigrationRough.csv')
herring.loc[herring["Identifier"] != "z", "Identifier"] = 4
herring['Name'] = herring['Name'].replace('Larus argentatus', 'European Herring Gull')
herring


# In[16]:


loon=pd.read_csv('RedLoonMigrationRough.csv')
loon.loc[loon["Identifier"] != "z", "Identifier"] = 5
loon['Name'] = loon['Name'].replace('Gavia stellata', 'Red Throated Loon')
loon


# In[17]:


swan=pd.read_csv('SwanMigrationRough.csv')
swan.loc[swan["Identifier"] != "z", "Identifier"] = 6
swan['Name'] = swan['Name'].replace('Cygnus columbianus', 'Tundra Swan')
swan


# In[18]:


birds = pd.concat([hawk.assign(source=1), emperor.assign(source=2), puffin.assign(source=3), herring.assign(source=4), loon.assign(source=5), swan.assign(source=6)], keys=[0, 1, 2, 3, 4, 5], ignore_index=True)
birds['Identifier'] = birds['Identifier'].replace({'71526a': 1, 'J91_2003_YK3_5845': 2})
birds


# In[19]:


birds['Timestamp'] = pd.to_datetime(birds['Timestamp'])
birds.sort_values('Timestamp', inplace=True)  # make sure data is sorted by time
opacity = birds['Timestamp'].dt.hour / 24  # calculate opacity values
birds


# In[20]:


fig = px.scatter_mapbox(birds, lat="Lat", lon="Long", color="Identifier",
                        hover_name="Name", zoom=3, opacity=.1,
                        color_discrete_sequence=px.colors.qualitative.Dark24)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(title="Bird sightings")
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})


# In[21]:


sns.set(style='whitegrid', palette="bright", font_scale=1.2)
sns.lmplot(x='Long', y='Lat', data=birds, hue='Name', height=5, aspect=1.5, scatter_kws={"s": 10})


# From here down was all stuff I was playing with.  It can be deleted later.

# In[16]:




scl = [0,"rgb(150,0,90)"],[0.125,"rgb(0, 0, 200)"],[0.25,"rgb(0, 25, 255)"],[0.375,"rgb(0, 152, 255)"],[0.5,"rgb(44, 255, 150)"],[0.625,"rgb(151, 255, 0)"],[0.75,"rgb(255, 234, 0)"],[0.875,"rgb(255, 111, 0)"],[1,"rgb(255, 0, 0)"]

fig = go.Figure(data=go.Scattergeo(
    lat = birds['Lat'],
    lon = birds['Long'],
    text = birds['Name'].astype(str),
    marker = dict(
        color = birds['Identifier'],
        colorscale = scl,
        reversescale = True,
        opacity = 0.7,
        size = 2,
        colorbar = dict(
            titleside = "right",
            outlinecolor = "rgba(68, 68, 68, 0)",
            ticks = "outside",
            showticksuffix = "last",
            dtick = 0.1
        )
    )
))

fig.update_layout(
    geo = dict(
        scope = 'north america',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        subunitcolor = "rgb(255, 255, 255)",
        countrycolor = "rgb(255, 255, 255)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = True,
        showcountries = True,
        resolution = 50,
        projection = dict(
            type = 'conic conformal',
            rotation_lon = -100
        ),
        lonaxis = dict(
            showgrid = True,
            gridwidth = 0.5,
            range= [ -140.0, -55.0 ],
            dtick = 5
        ),
        lataxis = dict (
            showgrid = True,
            gridwidth = 0.5,
            range= [ 20.0, 60.0 ],
            dtick = 5
        )
    ),
    title='Bird Migration',
)
fig.show()


# In[17]:


birds


# In[19]:


def location_map(birds, color, title):
    fig = px.scatter_mapbox(birds, lat="Lat", lon="Long",
                            color=color,
                            size_max=5,
                            zoom=8, 
                            height=500,
                            title = title,
                            hoverdata={'Long': True,
                                       'Lat': True,
                                       'Timestamp': True,
                                      }
                           )
    fig.update.layout(mapbox_style="open-street-map")
    fig.show()


# In[ ]:


print(temperature.head())


# **Convert the temperature data from Country name to Lat Long - center of the country**
# **Add Lat Long features to the Temperature dataset**

# In[ ]:


import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim

# Define geolocate function
def geolocate(country):
    geolocator = Nominatim(user_agent='center_by_country_name')
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return np.nan

# Load the temperature.csv file into a pandas DataFrame
df_temp = temperature

# Apply the geolocate function to the 'Country' column and store the results in new columns 'Lat' and 'Long'
df_temp[['Lat', 'Long']] = df_temp['Country'].apply(lambda x: pd.Series(geolocate(x)))

# Save the updated DataFrame back to the original CSV file
df_temp.to_csv('temperature.csv', index=False)


# In[22]:


df_temp = pd.read_csv('temperature.csv')


# In[23]:


birds.to_csv('birdstimestamps.csv', index=False)


# In[24]:


birds = pd.read_csv('birdstimestamps.csv')


# In[25]:


df_temp = pd.melt(df_temp, id_vars=['Continent', 'Country', 'Subregion', 'Lat', 'Long'], var_name='Year', value_name='Temp')


# In[26]:


# Convert the Temp column to numeric data type
df_temp['Temp'] = pd.to_numeric(df_temp['Temp'], errors='coerce')

# Drop any rows with NaN values in the Temp column
df_temp = df_temp.dropna(subset=['Temp'])


# In[27]:


fig = px.choropleth(df_temp, 
                    locations='Country', 
                    locationmode='country names', 
                    color='Temp', 
                    animation_frame='Year', 
                    range_color=(-2, 2),
                    color_continuous_scale=[(0, 'blue'), (0.5, 'yellow'), (1, 'red')])

fig.show()


# In[ ]:


# Create a pivot table of bird sightings by latitude, longitude, and timestamp
pivot = birds.pivot_table(index=["Lat", "Long"], columns="Timestamp", values="Identifier")

# Create a heatmap of the pivot table
fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, color_continuous_scale="viridis")

# Customize the layout
fig.update_layout(title="Bird sightings heatmap over time",
                  xaxis_title="Timestamp",
                  yaxis_title="Latitude, Longitude")

# Show the figure
fig.show()


# In[ ]:





# bird_names = pd.unique(birds.Name) 
# 
# # To move forward, we need to specify a 
# # specific projection that we're interested 
# # in using. 
# proj = ccrs.Mercator() 
# 
# plt.figure(figsize=(10,10)) 
# ax = plt.axes(projection=proj) 
# fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': proj, 'aspect': 'auto'}) 
# ax.add_feature(cfeature.LAND) 
# ax.add_feature(cfeature.OCEAN) 
# ax.add_feature(cfeature.COASTLINE) 
# ax.add_feature(cfeature.BORDERS, linestyle=':') 
# for name in bird_names: 
# 	ix = birds['Name'] == name 
# 	x,y = birds.Long[ix], birds.Lat[ix] 
# 	ax.plot(x,y,'.', transform=ccrs.Geodetic(), label=name) 
# plt.legend(loc="upper left") 
# plt.show() 

# In[ ]:




