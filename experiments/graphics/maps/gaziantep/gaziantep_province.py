import pandas as pd
import folium

# Load the data
data = pd.read_csv("../nodes_with_colors.csv")

# Create a map centered around Gaziantep
m = folium.Map(location=[37.0594, 37.3825], zoom_start=12)

# Add the nodes to the map
for idx, row in data.iterrows():
    folium.Marker(location=[row["Latitude"], row["Longitude"]],
                  icon=folium.Icon(color=row["Color"])).add_to(m)

# Display the map
m.save("gaziantep_map.html")
