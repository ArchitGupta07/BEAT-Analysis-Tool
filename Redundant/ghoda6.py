import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import folium
from scipy.spatial import ConvexHull

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)

    if not {'Latitude', 'Longitude', 'Coverage Type'}.issubset(data.columns):
        raise ValueError("The Excel file must contain 'Latitude', 'Longitude', and 'Coverage Type' columns.")

    data = data.dropna(subset=['Latitude', 'Longitude', 'Coverage Type'])

    # Filter for coordinates in Mumbai
    data = data[(data['Latitude'] >= 18.87) & (data['Latitude'] <= 19.30) & 
                (data['Longitude'] >= 72.77) & (data['Longitude'] <= 72.98)]

    return data

# Function to apply KMeans clustering dynamically for each Coverage Type
def apply_kmeans_clustering(data, max_split_cluster_size=20, max_regular_cluster_size=50, min_cluster_size=10):
    # Split the data into 'Split Coverage' and 'Regular'
    split_data = data[data['Coverage Type'] == 'Split Coverage']
    regular_data = data[data['Coverage Type'] == 'Regular']

    # Apply KMeans clustering for 'Split Coverage'
    split_data = cluster_and_split(split_data, max_split_cluster_size, min_cluster_size)

    # Apply KMeans clustering for 'Regular'
    regular_data = cluster_and_split(regular_data, max_regular_cluster_size, min_cluster_size)

    # Combine the data back
    data = pd.concat([split_data, regular_data], ignore_index=True)

    return data

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

# Function to create a Folium map for clusters
def create_folium_map(data, output_path, title='Cluster Map'):
    mean_lat, mean_lon = data['Latitude'].mean(), data['Longitude'].mean()
    # Use a grayscale tile layer
    cluster_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles='CartoDB positron')

    colors = [
        "#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#33FFF3",
        "#F333FF", "#FFA833", "#33FFA8", "#A833FF", "#FF33A8"
    ]

    cluster_sizes = {}  # Dictionary to store cluster sizes

    # Separate points by 'Coverage Type' category
    for coverage_type in data['Coverage Type'].unique():
        coverage_data = data[data['Coverage Type'] == coverage_type]

        for cluster_id in coverage_data['cluster_id'].unique():
            cluster_data = coverage_data[coverage_data['cluster_id'] == cluster_id]
            cluster_color = colors[cluster_id % len(colors)]

            # Count the number of points in the current cluster
            cluster_sizes[cluster_id] = len(cluster_data)

            print(f"Cluster {cluster_id} ({coverage_type}) contains {cluster_sizes[cluster_id]} points.")

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
                    f"Coverage: {coverage_type}"
                    f"Retail Name: {row.get('Retailer Name', 'N/A')}"
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
                    tooltip=f"Cluster {cluster_id} ({coverage_type}): Click for details"
                ).add_to(cluster_map)

    cluster_map.save(output_path)
    print(f"Map saved to {output_path}")

    # Print cluster sizes in ascending order after map is generated
    print("\nCluster sizes in ascending order:")
    for cluster_id, size in sorted(cluster_sizes.items(), key=lambda item: item[1]):
        print(f"Cluster {cluster_id}: {size} points")

# Main function
def clustering_algo_main():
    file_path = 'data/Samarth_Sales_RR.xlsx'
    output_map_path = 'final_cluster_map_with_coverage.html'

    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(file_path)

    print("Applying KMeans clustering with different limits...")
    # Set different max cluster sizes for 'Split Coverage' and 'Regular'
    data = apply_kmeans_clustering(data, max_split_cluster_size=20, max_regular_cluster_size=50)

    print("Creating cluster map...")
    create_folium_map(data, output_map_path)

    print("All maps generated successfully!")
    
    # Ask the user if they want to see the map
    view_map = input("Do you want to view the map? (Yes/No): ").strip().lower()
    if view_map == 'yes':
        import webbrowser
        webbrowser.open(output_map_path)

    # Ask the user if they want to generate an Excel file with cluster names
    generate_excel = input("Do you want to generate an Excel file with cluster names? (yes/no): ").strip().lower()
    if generate_excel == 'yes':
        output_excel_path = 'New_Clusters/New_Clustering.xlsx'
        save_to_excel(data, output_excel_path)

# Run the script
if __name__ == "__main__":
    clustering_algo_main()