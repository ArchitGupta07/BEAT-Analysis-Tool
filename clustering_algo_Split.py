import pandas as pd
import os
from sklearn.cluster import KMeans
import numpy as np
import folium
from scipy.spatial import ConvexHull

class TextColor:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"  # Reset to default color

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)

    if not {'Latitude', 'Longitude', 'Coverage Type'}.issubset(data.columns):
        raise ValueError("The Excel file must contain 'Latitude', 'Longitude', and 'Coverage Type' columns.")

    data = data.dropna(subset=['Latitude', 'Longitude', 'Coverage Type'])

    # Exclude rows where Channel is 'Wholesalers'
    data = data[data['Channel'] != 'Wholesalers']

    # Filter for coordinates in Mumbai
    data = data[(data['Latitude'] >= 18.87) & (data['Latitude'] <= 19.30) & 
                (data['Longitude'] >= 72.77) & (data['Longitude'] <= 72.98)]

    return data

# Function to apply KMeans clustering for 'Split Coverage' only
def apply_kmeans_clustering(data, max_split_cluster_size=20, min_cluster_size=10):
    # Filter for 'Split Coverage' only
    split_data = data[data['Coverage Type'] == 'Split Coverage']

    # Apply KMeans clustering for 'Split Coverage'
    split_data = cluster_and_split(split_data, max_split_cluster_size, min_cluster_size)

    return split_data  # Return only the split data

# Function to split large clusters and merge small ones
def cluster_and_split(data, max_cluster_size, min_cluster_size):
    if data.empty:
        return data

    # Initial KMeans clustering
    initial_clusters = max(1, len(data) // max_cluster_size)
    kmeans = KMeans(n_clusters=initial_clusters, random_state=42)
    data['cluster_id'] = kmeans.fit_predict(data[['Latitude', 'Longitude']])

    unique_id_offset = 0
    final_data = []

    for cluster_id in data['cluster_id'].unique():
        cluster_data = data.loc[data['cluster_id'] == cluster_id].copy()

        # Split clusters that exceed max_cluster_size
        if len(cluster_data) > max_cluster_size:
            print(f"Splitting cluster {cluster_id} with {len(cluster_data)} points.")
            num_new_clusters = int(len(cluster_data) / max_cluster_size) + 1
            new_kmeans = KMeans(n_clusters=num_new_clusters, random_state=42)
            cluster_data['cluster_id'] = (
                new_kmeans.fit_predict(cluster_data[['Latitude', 'Longitude']]) + unique_id_offset
            )
            unique_id_offset += num_new_clusters
        else:
            cluster_data['cluster_id'] = unique_id_offset
            unique_id_offset += 1

        # Handle small clusters
        small_clusters = cluster_data['cluster_id'].value_counts()
        small_clusters = small_clusters[small_clusters < min_cluster_size]

        for small_cluster_id in small_clusters.index:
            small_cluster_data = cluster_data[cluster_data['cluster_id'] == small_cluster_id]
            print(f"Cluster {small_cluster_id} has fewer than {min_cluster_size} points, merging with a larger cluster.")

            # Find the nearest larger cluster to merge with
            larger_cluster_id = cluster_data['cluster_id'].value_counts().idxmax()
            cluster_data.loc[cluster_data['cluster_id'] == small_cluster_id, 'cluster_id'] = larger_cluster_id

        final_data.append(cluster_data)

    final_data = pd.concat(final_data, ignore_index=True)
    final_data['cluster_id'] = pd.factorize(final_data['cluster_id'])[0] + 1  # Renumber clusters sequentially
    return final_data

def save_to_excel(data, output_excel_path):
    # Save the data with cluster IDs to an Excel file
    data.to_excel(output_excel_path, index=False)
    print(f"Data with cluster IDs saved to {output_excel_path}")

# Function to create a Folium map for 'Split Coverage' clusters
def create_folium_map(data, output_path, title='Cluster Map'):
    mean_lat, mean_lon = data['Latitude'].mean(), data['Longitude'].mean()
    # Use a grayscale tile layer
    cluster_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles='CartoDB positron')

    colors = [
        "#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#33FFF3",
        "#F333FF", "#FFA833", "#33FFA8", "#A833FF", "#FF33A8"
    ]

    cluster_sizes = {}  # Dictionary to store cluster sizes

    # Only process 'Split Coverage' data
    coverage_data = data[data['Coverage Type'] == 'Split Coverage']

    for cluster_id in coverage_data['cluster_id'].unique():
        cluster_data = coverage_data[coverage_data['cluster_id'] == cluster_id]
        cluster_color = colors[cluster_id % len(colors)]

        # Count the number of points in the current cluster
        cluster_sizes[cluster_id] = len(cluster_data)

        print(f"Cluster {cluster_id} (Split Coverage) contains {cluster_sizes[cluster_id]} points.")

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

        # Add markers with popups and tooltips
        for _, row in cluster_data.iterrows():
            popup_content = (
                f"Cluster ID: {cluster_id}"
                f"Coverage: {row['Coverage Type']}"
                f"Retail Name: {row.get('Retail Name', 'N/A')}"
                f"Channel Sub Type: {row.get('Channel Sub Type', 'N/A')}"
                f"Coordinates: ({row['Latitude']}, {row['Longitude']})"
            )
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,  # Smaller radius for sharper points
                color=cluster_color,
                fill=True,
                fill_color=cluster_color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Cluster {cluster_id} (Split Coverage): Click for details"
            ).add_to(cluster_map)

    # Get distributor coordinates
    distributor_lat, distributor_lon = get_distributor_coordinates(19.234424, 72.969097)

    # Add distributor marker with unique design
    folium.Marker(
        location=(distributor_lat, distributor_lon),
        popup=folium.Popup(f"Distributor Location<br>Coordinates: ({distributor_lat}, {distributor_lon})", max_width=300),
        icon=folium.Icon(color='red', icon='glyphicon glyphicon-map-marker', prefix='glyphicon'),
        tooltip="Distributor Location"
    ).add_to(cluster_map)


    folium.Marker(
        location=(distributor_lat, distributor_lon),
        popup=folium.Popup(f"Distributor Location<br>Coordinates: ({distributor_lat}, {distributor_lon})", max_width=300),
        icon=folium.Icon(color='red', icon='glyphicon glyphicon-map-marker', prefix='glyphicon'),
        tooltip="Distributor Location"
    ).add_to(cluster_map)

    cluster_map.save(output_path)
    print(f"Map saved to {output_path}")

    # Print cluster sizes in ascending order after map is generated
    print("\nCluster sizes in ascending order:")
    for cluster_id, size in sorted(cluster_sizes.items(), key=lambda item: item[1]):
        print(f"Cluster {cluster_id}: {size} points")



def get_distributor_coordinates(default_lat, default_lon):
    """
    Ask the user if they want to enter custom distributor coordinates.
    If yes, take input; otherwise, return default values.
    """
    use_custom_coords = input("Do you want to enter custom Distributor Latitude and Longitude? (yes/no): ").strip().lower()
    if use_custom_coords == 'yes':
        try:
            lat = float(input("Enter Distributor Latitude: ").strip())
            lon = float(input("Enter Distributor Longitude: ").strip())
            return lat, lon
        except ValueError:
            print("Invalid input! Using default coordinates.")
            return default_lat, default_lon
    else:
        return default_lat, default_lon
    

# Main function
def clustering_algo_split_main(max_split_cluster_size=20, min_cluster_size=10):
    data_folder = 'Data'
    file_name = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')][0]
    file_path = os.path.join(data_folder, file_name)
    output_map_path = 'Cluster_Split.html'

    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(file_path)

    print("Applying KMeans clustering for 'Split Coverage'...")
    data = apply_kmeans_clustering(data, max_split_cluster_size, min_cluster_size)

    print("Creating cluster map...")
    create_folium_map(data, output_map_path)

    print("All maps generated successfully!")
    
    # Ask the user if they want to see the map
    view_map = input(f"{TextColor.BLUE}Do you want to view the map? (Yes/No): {TextColor.RESET}").strip().lower()
    if view_map == 'yes':
        import webbrowser
        webbrowser.open(output_map_path)

    # Ask the user if they want to generatey an Excel file with cluster names
    generate_excel = input(f"{TextColor.BLUE}Do you want to generate an Excel file with cluster names? (yes/no): {TextColor.RESET}").strip().lower()
    if generate_excel == 'yes':
        output_excel_path = 'New_Clusters/New_Clustering_SplitOnly.xlsx'
        save_to_excel(data, output_excel_path)

# # Run the script
if __name__ == "__main__":
    clustering_algo_split_main()