import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

def load_and_filter_data(file_path):
    """Load and filter data based on coverage type and location bounds."""
    data = pd.read_excel(file_path)
    data = data.dropna(subset=['Latitude', 'Longitude'])

    # Filter coordinates within Mumbai's approximate bounds
    lat_min, lat_max = 18.87, 19.27
    lon_min, lon_max = 72.77, 72.97
    data = data[(data['Latitude'] >= lat_min) & (data['Latitude'] <= lat_max) &
                (data['Longitude'] >= lon_min) & (data['Longitude'] <= lon_max)]

    # Filter data for Regular and Split Coverage
    regular_data = data[data['Coverage Type'] == 'Regular']
    split_data = data[data['Coverage Type'] == 'Split Coverage']

    return regular_data, split_data

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

def plot_split_routes(fig, split_data):
    """Add scatter plot and boundaries for split routes."""
    split_traces = []
    for route_code, group in split_data.groupby("Route Code"):
        trace = go.Scattermapbox(
            lat=group['Latitude'],
            lon=group['Longitude'],
            mode='markers',
            marker=dict(size=8, opacity=0.8),
            hoverinfo='text',
            text=group.apply(lambda row: f"{row['Retailer Name']}Route Code: {row['Route Code']}Channel Sub Type: {row['Channel Sub Type']}Coordinates: ({row['Latitude']}, {row['Longitude']})", axis=1),
            name=f'Split Route {route_code}'
        )
        split_traces.append(trace)
        fig.add_trace(trace)
    return fig

def add_distributor_location(fig, distributor_location):
    """Add distributor location to the map."""
    fig.add_trace(go.Scattermapbox(
        lat=[distributor_location[0]],
        lon=[distributor_location[1]],
        hoverinfo='text',
        text=["Distributor Location"],
        marker=dict(size=20, color='orange', symbol='star'),
        name="Distributor Location"
    ))
    return fig

def main():
    file_path = "Data/Samarth_Sales_RR.xlsx"
    regular_data, split_data = load_and_filter_data(file_path)

    # Print unique route codes for debugging
    print("Unique Route Codes (Regular):", regular_data['Route Code'].unique())
    print("Unique Route Codes (Split):", split_data['Route Code'].unique())

    # Define colors and create color map
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_map = generate_color_map(regular_data['Route Code'].unique(), colors)

    # Create convex hulls for regular routes
    hulls = create_convex_hulls(regular_data, color_map)

    # Initialize the map figure
    fig = go.Figure()

    # Add regular and split routes to the figure
    fig = plot_regular_routes(fig, regular_data, hulls, colors)
    fig = plot_split_routes(fig, split_data)

    # Add distributor location
    distributor_location = [19.107342, 72.92511]
    fig = add_distributor_location(fig, distributor_location)

    # Update layout
    fig.update_layout(
        mapbox=dict(
            center={"lat": distributor_location[0], "lon": distributor_location[1]},
            zoom=14
        ),
        title="Retailers Map - Regular and Split Coverage",
        showlegend=True
    )

    # Save map as HTML file
    fig.write_html("existing_beat.html")
    print("Map created! Open existing_beat.html to view.")

    # Ask user if they want to view the map
    view_map = input("Do you want to see the map now? (Yes/No): ")
    if view_map.lower() == 'yes':
        fig.show()
    else:
        print("Thank You !!! ")
        print("For any queries contact: nayal.saket@in.nestle.com")

if __name__ == "__main__":
    main()