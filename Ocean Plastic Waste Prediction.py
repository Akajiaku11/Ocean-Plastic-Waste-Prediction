# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import geopandas as gpd  # For geographical data
import folium  # For interactive map visualization

# Step 1: Load and Preprocess Data
# Assuming we have a dataset with columns like 'latitude', 'longitude', 'plastic_waste_density', 'ocean_current', 'wind_speed'
data = pd.read_csv("ocean_plastic_data.csv")
data.dropna(inplace=True)

# Step 2: Feature Selection
features = data[['latitude', 'longitude', 'ocean_current', 'wind_speed']]
target = data['plastic_waste_density']

# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Step 4: Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make Predictions
predictions = model.predict(X_test)

# Step 6: Evaluate Model
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Step 7: Visualize Predicted Accumulation Areas
# Example for visualizing prediction results on a map
map_data = X_test.copy()
map_data['predicted_density'] = predictions
map_data = gpd.GeoDataFrame(map_data, geometry=gpd.points_from_xy(map_data.longitude, map_data.latitude))

# Create a map centered around a specific location
m = folium.Map(location=[map_data['latitude'].mean(), map_data['longitude'].mean()], zoom_start=2)

# Add predicted accumulation points to the map
for _, row in map_data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='blue' if row['predicted_density'] < 1 else 'red',
        fill=True,
        fill_opacity=0.6,
        popup=f"Density: {row['predicted_density']}"
    ).add_to(m)

# Save or display map
m.save("ocean_plastic_prediction_map.html")
