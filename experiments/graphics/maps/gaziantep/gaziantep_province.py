from pathlib import Path
import pandas as pd
import folium

# Data and output are resolved relative to this script rather than the working directory
NODES_FILE = Path(__file__).resolve().parent.parent / "nodes_with_colors.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "gaziantep_map.html"

# Load the data
data = pd.read_csv(NODES_FILE)

# Create a map centered around Gaziantep
m = folium.Map(location=[37.0594, 37.3825], zoom_start=12)

# Add the nodes to the map
for idx, row in data.iterrows():
    folium.Marker(location=[row["Latitude"], row["Longitude"]],
                  icon=folium.Icon(color=row["Color"])).add_to(m)

# Display the map
m.save(OUTPUT_FILE)
