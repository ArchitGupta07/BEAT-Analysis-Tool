import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

# Set recursion limit (to handle deeper clustering if needed)
sys.setrecursionlimit(2000)

# Constants for clustering
min_size = 10  # Minimum size of a cluster
max_size = 100  # Maximum size of a cluster
max_depth = 10  # Maximum recursion depth

def recursive_clustering(data, labels, cluster_id, depth=0):
    """
    Perform recursive clustering on the data.
    """
    if depth > max_depth or len(data) < min_size:
        # Assign current points to the same cluster
        for point in data:
            indices = np.where((data == point).all(axis=1))[0]  # Use 'data' here instead of 'df'
            labels[indices] = cluster_id
        return cluster_id + 1

    # Determine number of clusters dynamically
    n_clusters = min(len(np.unique(data, axis=0)), max(2, len(data) // min_size))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(data)
    sub_labels = kmeans.labels_

    # Recurse into each sub-cluster
    for sub_label in np.unique(sub_labels):
        points = data[sub_labels == sub_label]
        cluster_id = recursive_clustering(points, labels, cluster_id, depth + 1)

    return cluster_id

def perform_clustering(df):
    """
    Cluster the input dataframe and return cluster labels.
    """
    labels = np.zeros(len(df), dtype=int)
    cluster_id = 0
    cluster_id = recursive_clustering(df[['Latitude', 'Longitude']].values, labels, cluster_id)
    return labels

def visualize_clusters(df, labels, map_file):
    """
    Visualize the clustered points on a map.
    """
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=10)
    
    # Use Matplotlib to generate a color palette
    num_clusters = len(np.unique(labels))
    color_palette = cm.get_cmap('Set1', num_clusters)
    
    for cluster_id in np.unique(labels):
        cluster_points = df[labels == cluster_id]
        
        # Convert the colormap to a color for the current cluster
        color = "#{:02x}{:02x}{:02x}".format(
            int(color_palette(cluster_id)[0] * 255),
            int(color_palette(cluster_id)[1] * 255),
            int(color_palette(cluster_id)[2] * 255),
        )
        
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in cluster_points.iterrows():
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                icon=folium.Icon(color='blue')
            ).add_to(marker_cluster)

        # Draw ConvexHull if there are enough points
        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points[['Latitude', 'Longitude']])
            for simplex in hull.simplices:
                points = cluster_points.iloc[simplex][['Latitude', 'Longitude']].values
                folium.PolyLine(points, color=color, weight=2).add_to(m)

    m.save(map_file)
    print(f"Map saved to {map_file}")

def main(file_path, output_file, map_file):
    """
    Main function to read data, perform clustering, and visualize results.
    """
    # Load the dataset
    df = pd.read_excel(file_path)
    # Remove duplicate and NaN coordinates
    df = df.drop_duplicates(subset=['Latitude', 'Longitude'])
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Perform clustering
    labels = perform_clustering(df)
    df['Cluster'] = labels

    # Save the clustered data to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Clustered data saved to {output_file}.")

    # Visualize the clusters on a map
    visualize_clusters(df, labels, map_file)

if __name__ == "__main__":
    # Specify file paths
    file_path = "Data/Samarth_Sales_RR.xlsx"  # Replace with your input file path
    output_file = "clustered_data.xlsx"  # Replace with your desired output file name
    map_file = "clusters_map.html"  # Replace with your desired map output file name

    # Run the main function
    main(file_path, output_file, map_file)
