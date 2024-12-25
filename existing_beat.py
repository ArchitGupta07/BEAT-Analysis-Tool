import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

class TextColor:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"  # Reset to default color


def load_and_filter_data(file_path):
    """Load and filter data based on coverage type and location bounds."""
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['Latitude', 'Longitude'])

    distributor_code = data['Distr Code'].iloc[0]


    # Filter coordinates within Mumbai's approximate bounds
    # lat_min, lat_max = 18.87, 19.27
    # lon_min, lon_max = 72.77, 72.97
    data = data[(data['Latitude'] >= 18.50) & (data['Latitude'] <= 19.90) & 
                (data['Longitude'] >= 72.60) & (data['Longitude'] <= 73.10)]
    
    # Exclude rows where Channel is 'Wholesalers'
    data = data[data['Channel'] != 'Wholesalers']

    # Filter data for Regular and Split Coverage
    regular_data = data[data['Coverage Type'] == 'Regular']
    split_data = data[data['Coverage Type'] == 'Split Coverage']

    return regular_data, split_data,distributor_code


def get_convex_hull(points):
    """Return convex hull vertices for given points."""
    if len(points) > 2:  # At least 3 points are required for a convex hull
        hull = ConvexHull(points)
        return points[hull.vertices]
    return None


def generate_color_map(route_codes, colors):
    """Generate a color map for unique route codes."""
    return {route: colors[i % len(colors)] for i, route in enumerate(route_codes)}


def create_convex_hulls(data, color_map):
    """Create convex hull polygons for Regular routes."""
    hulls = []
    for route_code, group in data.groupby("Route Code"):
        points = group[['Latitude', 'Longitude']].to_numpy()
        hull = get_convex_hull(points)
        if hull is not None:
            hull = np.vstack([hull, hull[0]])  # Close the polygon
            hulls.append((hull, color_map[route_code], route_code))
    return hulls


def create_split_convex_hulls(split_data, color_map):
    """Create convex hull polygons for Split routes."""
    hulls = []
    for route_code, group in split_data.groupby("Route Code"):
        points = group[['Latitude', 'Longitude']].to_numpy()
        hull = get_convex_hull(points)
        if hull is not None:
            hull = np.vstack([hull, hull[0]])  # Close the polygon
            hulls.append((hull, color_map[route_code], route_code))
    return hulls


def plot_regular_routes(fig, regular_data, hulls, colors):
    """Plot Regular routes with scatter points and convex hulls."""
    # Add scatter plot for regular routes
    fig = px.scatter_mapbox(
        regular_data,
        lat='Latitude',
        lon='Longitude',
        hover_name='Retailer Name',
        hover_data=['Route Code', 'Channel Sub Type'],
        color='Route Code',
        color_discrete_sequence=colors,
        mapbox_style="carto-positron",
        zoom=12,
        center={"lat": 19.076, "lon": 72.8777},
        size_max=10
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))

    # Add convex hulls for regular routes
    for hull, color, route_code in hulls:
        fig.add_trace(go.Scattermapbox(
            lat=hull[:, 0],
            lon=hull[:, 1],
            mode='lines',
            line=dict(color=color, width=2),
            name=f"Regular Route {route_code}"
        ))
    return fig


def plot_split_routes(fig, split_data, split_hulls):
    """Add scatter plot and boundaries for split routes."""
    for route_code, group in split_data.groupby("Route Code"):
        # Add scatter points for split routes
        fig.add_trace(go.Scattermapbox(
            lat=group['Latitude'],
            lon=group['Longitude'],
            mode='markers',
            marker=dict(size=8, opacity=0.8),
            hoverinfo='text',
            text=group.apply(lambda row: f"{row['Retailer Name']}<br>"
                                         f"Route Code: {row['Route Code']}<br>"
                                         f"Channel Sub Type: {row['Channel Sub Type']}<br>"
                                         f"Coordinates: ({row['Latitude']}, {row['Longitude']})", axis=1),
            name=f'Split Route {route_code}'
        ))

    # Add convex hulls for split routes
    for hull, color, route_code in split_hulls:
        fig.add_trace(go.Scattermapbox(
            lat=hull[:, 0],
            lon=hull[:, 1],
            mode='lines',
            line=dict(color=color, width=2),  # Removed 'dash'
            name=f"Split Route {route_code}"
        ))

    return fig


# def get_distributor_coordinates(default_lat, default_lon):
#     """Ask the user if they want to enter custom distributor coordinates."""
#     use_custom_coords = input(f"{TextColor.BLUE}Do you want to enter custom Distributor Latitude and Longitude? (yes/no):{TextColor.RESET}").strip().lower()
#     if use_custom_coords == 'yes':
#         try:
#             lat = float(input("Enter Distributor Latitude: ").strip())
#             lon = float(input("Enter Distributor Longitude: ").strip())
#             return lat, lon
#         except ValueError:
#             print(f"{TextColor.RED}Invalid input! Using default coordinates.{TextColor.RESET}")
#             return default_lat, default_lon
#     else:
#         return default_lat, default_lon

def get_distributor_coordinates(distributor_code):
    """
    Ask the user if they want to enter custom distributor coordinates.
    If yes, take input; otherwise, return default values.
    """

    data_folder = 'Data/Distributor_Master'
    file_name = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')][0]
    file_path = os.path.join(data_folder, file_name)
    df2 = pd.read_excel(file_path)
    matching_row = df2[df2['Distr Code'] == distributor_code]

    if not matching_row.empty:
        longitude = matching_row['Distributor Longitude'].values[0]
        latitude = matching_row['Distributor Latitude'].values[0]
        print(f"Longitude: {longitude}, Latitude: {latitude}")
        return latitude, longitude
    else:
        print("No matching row found in File 2.")
        return 0,0


def existing_beat_main():
    data_folder = 'Data'
    file_name = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')][0]
    file_path = os.path.join(data_folder, file_name)

    regular_data, split_data,distributor_code = load_and_filter_data(file_path)

    # Combine route codes from both regular and split data
    all_route_codes = pd.concat([regular_data['Route Code'], split_data['Route Code']]).unique()

    # Define colors and create color map
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_map = generate_color_map(all_route_codes, colors)

    # Create convex hulls for regular and split routes
    regular_hulls = create_convex_hulls(regular_data, color_map)
    split_hulls = create_split_convex_hulls(split_data, color_map)

    # Initialize the map figure
    fig = go.Figure()

    # Add regular and split routes to the figure
    fig = plot_regular_routes(fig, regular_data, regular_hulls, colors)
    fig = plot_split_routes(fig, split_data, split_hulls)

    # Get distributor coordinates
    distributor_lat, distributor_lon = get_distributor_coordinates(distributor_code)

    # Add distributor marker with unique design
    fig.add_trace(go.Scattermapbox(
        lat=[distributor_lat],
        lon=[distributor_lon],
        mode='markers+text',
        marker=dict(
            size=15,
            color='red',
            symbol='circle'
        ),
        text=[f"Distributor Location<br>Coordinates: ({distributor_lat}, {distributor_lon})"],
        hoverinfo='text',
        name="Distributor Location",
        showlegend=True
    ))

    # Adjust map center and zoom to ensure distributor is visible
    fig.update_layout(
        mapbox=dict(
            zoom=12,
            center=dict(lat=distributor_lat, lon=distributor_lon),
            style="carto-positron"
        )
    )

    # Save map as HTML file
    fig.write_html("existing_beat.html")
    print("Map created! Open existing_beat.html to view.")

    # Ask user if they want to view the map
    view_map = input(f"{TextColor.BLUE}Do you want to see the map now? (Yes/No): {TextColor.RESET}")
    if view_map.lower() == 'yes':
        fig.show()


# if __name__ == "__main_":
#     existing_beat_main()