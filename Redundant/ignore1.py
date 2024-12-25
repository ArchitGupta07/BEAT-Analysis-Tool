import pandas as pd
from sklearn.cluster import Birch, KMeans
import numpy as np
import folium
from scipy.spatial import ConvexHull
from geopy.distance import geodesic

# Function to load and preprocess data
def load_and_preprocess_data(file_path, distributor_location):
    data = pd.read_excel(file_path)

    if not {'Latitude', 'Longitude', 'Coverage Type'}.issubset(data.columns):
        raise ValueError("The Excel file must contain 'Latitude', 'Longitude', and 'Coverage Type' columns.")

    data = data.dropna(subset=['Latitude', 'Longitude', 'Coverage Type'])
    
    # Filter for coordinates in Mumbai
    data = data[(data['Latitude'] >= 18.87) & (data['Latitude'] <= 19.30) & 
                (data['Longitude'] >= 72.77) & (data['Longitude'] <= 72.98)]

    # Filter by Coverage Type = 'Regular'
    data = data[data['Coverage Type'] == 'Regular']

    # Remove points that lie within a 30-meter radius of the distributor location
    data['distance_to_distributor'] = data.apply(
        lambda row: geodesic((row['Latitude'], row['Longitude']), distributor_location).meters, axis=1
    )

    # Define a threshold distance (30 meters)
    threshold_distance = 30
    initial_count = len(data)
    data = data[data['distance_to_distributor'] > threshold_distance]

    ignored_count = initial_count - len(data)
    print(f"Number of coordinates ignored (within 20 meters of distributor): {ignored_count}")

    return data

# Function to apply BIRCH clustering
def apply_birch_clustering(data, threshold=0.1):
    birch = Birch(n_clusters=None, threshold=threshold)
    data['birch_cluster'] = birch.fit_predict(data[['Latitude', 'Longitude']])
    return data

# Function to split large clusters
def split_large_clusters(data, max_cluster_size=45):
    final_clusters = []

    for cluster_id in data['birch_cluster'].unique():
        cluster_data = data[data['birch_cluster'] == cluster_id]

        if len(cluster_data) > max_cluster_size:
            print(f"Splitting cluster {cluster_id} with {len(cluster_data)} points.")
            kmeans = KMeans(n_clusters=int(len(cluster_data) / max_cluster_size) + 1, random_state=42)
            cluster_data['split_cluster'] = kmeans.fit_predict(cluster_data[['Latitude', 'Longitude']])
        else:
            cluster_data['split_cluster'] = cluster_id

        final_clusters.append(cluster_data)

    return pd.concat(final_clusters, ignore_index=True)

# Function to create a Folium map for a dataset
def create_folium_map(data, distributor_location, output_path, title='Cluster Map'):
    mean_lat, mean_lon = data['Latitude'].mean(), data['Longitude'].mean()
    cluster_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

    # Adding a unique marker for the distributor location
    folium.Marker(
        location=distributor_location,
        popup="Distributor Location",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(cluster_map)

    colors = [
        "#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#33FFF3",
        "#F333FF", "#FFA833", "#33FFA8", "#A833FF", "#FF33A8"
    ]

    for cluster_id in data['split_cluster'].unique():
        cluster_data = data[data['split_cluster'] == cluster_id]
        cluster_color = colors[cluster_id % len(colors)]

        if len(cluster_data) > 2:
            points = cluster_data[['Latitude', 'Longitude']].to_numpy()
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_polygon = [tuple(point) for point in hull_points] + [tuple(hull_points[0])]

            folium.PolyLine(
                locations=hull_polygon,
                color=cluster_color,
                weight=2,
                opacity=0.7
            ).add_to(cluster_map)

        for _, row in cluster_data.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=cluster_color,
                fill=True,
                fill_color=cluster_color,
                fill_opacity=0.7,
            ).add_to(cluster_map)

    cluster_map.save(output_path)
    print(f"Map saved to {output_path}")

# Function to handle large clusters recursively
def handle_large_clusters(data, distributor_location, max_cluster_size=70):
    for cluster_id in data['split_cluster'].unique():
        cluster_data = data[data['split_cluster'] == cluster_id]

        if len(cluster_data) > max_cluster_size:
            print(f"Handling large cluster {cluster_id} with {len(cluster_data)} points.")
            sub_map_path = f"cluster_{cluster_id}_subdivided.html"

            reclustered_data = split_large_clusters(cluster_data, max_cluster_size=45)
            create_folium_map(reclustered_data, distributor_location, sub_map_path)

# Main function
def main():
    distributor_location = (19.107342, 72.92511)  # Correct distributor location
    file_path = 'Data/Samarth_Sales.xlsx'  # Correct file path
    output_map_path = 'final_cluster_map.html'

    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(file_path, distributor_location)

    print("Applying BIRCH clustering...")
    data = apply_birch_clustering(data)

    print("Splitting large clusters...")
    data = split_large_clusters(data, max_cluster_size=45)

    print("Creating main cluster map...")
    create_folium_map(data, distributor_location, output_map_path)

    print("Handling large clusters recursively...")
    handle_large_clusters(data, distributor_location, max_cluster_size=70)

    print("All maps generated successfully!")

# Run the script
if __name__ == "__main__":
    main()
