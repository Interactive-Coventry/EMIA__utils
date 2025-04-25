from os.path import join as pathjoin

import geopandas as gpd
from geopandas import GeoDataFrame
from matplotlib import pyplot as plt
from shapely.geometry import Point


def split_in_categories(value, range_size, categories=3):
    if categories == 3:
        if value < range_size:
            category = 'blue'
        elif range_size <= value < 2 * range_size:
            category = 'yellow'
        else:
            category = 'red'
        return category
    else:
        raise ValueError("Only 3 categories are supported.")


def categorize_and_color(values, categories=3):
    max_value = max(values)
    range_size = max_value / categories
    colored_values = [split_in_categories(value, range_size, categories) for value in values]
    return colored_values


def assign_heatmap_colors(values, colormap_name='viridis'):
    norm = plt.Normalize(0, max(values))
    cmap = plt.get_cmap(colormap_name)
    colored_values = [cmap(norm(value)) for value in values]
    return colored_values


def print_camera_locations(camera_info, camera_list, longitude_key_name, latitude_key_name, camera_id_key_name, camera_types, show_legend=True):

    df_info = camera_info[camera_info[camera_id_key_name].isin(camera_list)]

    singapore = gpd.read_file(pathjoin("assets", "maps", "SGP_adm0.shp"))
    if "colors" in camera_info.columns:
        colors = camera_info["colors"].values
    else:
        colors = ["red" if x == 0 else "blue" for x in camera_info["camera_type"]]

    if "is_selected" in camera_info.columns:
        markers = ["o" if x == 0 else "d" for x in camera_info["is_selected"]]
    else:
        markers = ["o" for _ in camera_info["camera_type"]]

    # Create a GeoDataFrame
    gdf_points = gpd.GeoDataFrame(
        df_info,
        geometry=gpd.points_from_xy(df_info[longitude_key_name], df_info[latitude_key_name]),
        crs="EPSG:4326"
    )
    # handle only points inside the boundaries of Singapore
    gdf_points = gdf_points[gdf_points.geometry.within(singapore.unary_union)]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    singapore.plot(ax=ax, color='white', edgecolor='black')

    sensor = {longitude_key_name: [103.8501], latitude_key_name: [1.2897], camera_id_key_name: ["Weather station"]}
    geometry = [Point(xy) for xy in zip(sensor[longitude_key_name], sensor[latitude_key_name])]
    gdf_weather = GeoDataFrame(sensor, geometry=geometry)

    last_point = None
    for point, color, marker in zip(gdf_points.geometry, colors, markers):
        if marker == "d": # emphasize the selected camera
            last_point = [point, color, marker]
        else:
            gpd.GeoSeries([point]).plot(ax=ax, color=color, marker=marker, markersize=15)

    if show_legend: # Shows weather station only when legend is enabled
        gdf_weather.plot(ax=ax, marker="s", color="cyan", markersize=15)

    if last_point is not None:
        gpd.GeoSeries([last_point[0]]).plot(ax=ax, color=last_point[1], marker=last_point[2], markersize=30,
                                            edgecolor="black")

    if show_legend:
        # Add a dummy plot for the legend
        dummy_point_expr = plt.Line2D([0], [0], marker="o", color="red", markersize=5, linewidth=0,
                                       label=camera_types[0])
        dummy_point_mobile = plt.Line2D([0], [0], marker="o", color="blue", markersize=5, linewidth=0,
                                        label=camera_types[1])
        dummy_point_selected = plt.Line2D([0], [0], marker="d", color="black", markersize=5, linewidth=0,
                                        label="Selected camera")
        dummy_point_station = plt.Line2D([0], [0], marker="s", color="cyan", markersize=5, linewidth=0,
                                        label="Weather station")
        ax.legend(handles=[dummy_point_expr, dummy_point_mobile, dummy_point_selected, dummy_point_station],
                  loc="lower right", fontsize=8)

    ax.axis("off")

    return ax.get_figure()