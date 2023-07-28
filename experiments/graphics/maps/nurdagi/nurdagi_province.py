import pandas as pd
import folium

# Load the node data
nodes = pd.read_csv("../nodes_with_colors.csv")

# Create a map centered around Gaziantep
m = folium.Map(location=[37.2, 36.85], zoom_start=10)  # Coordinates adjusted to center the map better

# Define the boundary coordinates
boundaries = [[37.05, 36.65], [37.35, 36.65], [37.35, 37.05], [37.05, 37.05]]

# Add a polygon to the map using the boundary coordinates
folium.vector_layers.Polygon(locations=boundaries, color="green", fill=True).add_to(m)

# Add the nodes to the map
for idx, row in nodes.iterrows():
    folium.Marker(location=[row["Latitude"], row["Longitude"]],
                  icon=folium.Icon(color=row["Color"])).add_to(m)

# Save the map to an HTML file
m.save("map.html")
