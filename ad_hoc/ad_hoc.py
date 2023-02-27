# To plot the most recent x journeys

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import gpxpy
import shapely
from shapely.geometry import Polygon
import numpy as np
import time
import datetime

from utils import (
    which_layers_needed,
    read_in_original_file,
    plotter_functions_dict,
    convert_crop_base_map_layers,
    get_speeds,
    scalarMap,
)
from configs.configs_maps import (
    base_layers_configs,
    map_configs,
    colored_black_end_points,
    colored_not_black_end_points,
    red,
    data_path,
    x_lims,
    y_lims,
)

n_concurrent = 100
journey_plots_for_moving_recents = []
journey_plots_for_by_year = []

# Set the runstring (string of the time of running)
runstr = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M")

maps_dict = {}

# Only want to plot with alpha = 1, so overwrite the values in map_configs
for key in map_configs.keys():
    map_configs[key]["plotting_or_not"] = False
map_configs["overall_alpha_1"]["plotting_or_not"] = True

# Find exactly which layers are needed, read them in, crop them down
layers_to_read = which_layers_needed()
base_layers = read_in_original_file(layers_to_read)
base_layers = convert_crop_base_map_layers(base_layers)

plotting_all = False
number_journeys = [5, 10, 20, 30, 50, 100, 200, 250]


def make_one_off_plots(journeys):
    # Plot the base layers
    for key, value in map_configs.items():
        if value["plotting_or_not"]:
            print("")
            fig, ax = plt.subplots(figsize=(14 * 2, 6 * 2))
            ax.set_xlim(-23000, 1000)
            ax.set_ylim(6667500, 6686000)
            x_axis = ax.axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = ax.axes.get_yaxis()
            y_axis.set_visible(False)
            if key == "total_fig_no_roads":
                print("Plotting " + "tidal_water" + " for " + key)
                ax.set_facecolor(tuple([x / 255 for x in [0, 0, 0]]))
                maps_dict[key] = [fig, ax, x_axis, y_axis]
                base_layers["tidal_water"].plot(
                    categorical=False,
                    legend=False,
                    ax=maps_dict[key][1],
                    color=tuple([x / 255 for x in [0, 0, 60]]),
                    zorder=0,
                )
            else:
                for layer in value["layers"]:
                    print("Plotting " + layer + " for " + key)
                    ax.set_facecolor(tuple([x / 255 for x in [232, 237, 232]]))
                    maps_dict[key] = [fig, ax, x_axis, y_axis]
                    base_layers[layer].plot(
                        categorical=False,
                        legend=False,
                        ax=maps_dict[key][1],
                        color=base_layers_configs[layer][1],
                        zorder=0,
                    )

    n_files_unparsable = 0
    unparsable_files = []
    all_xs = []
    all_ys = []
    n_journeys_plotted = 0
    n_non_cycling_activities = 0
    n_journeys_attempted = 0

    for journey in journeys:
        try:
            # Read in the data, convert to gdf
            gpx = gpxpy.parse(
                open(os.path.join(os.path.join(data_path, "cycling_data"), journey))
            )
            track = gpx.tracks[0]
            lats = []
            lngs = []
            times = []

            if "cycling" in gpx.tracks[0].name.lower():
                for segment in track.segments:
                    for point in segment.points:
                        lats.append(point.latitude)
                        lngs.append(point.longitude)
                        times.append(time.mktime(point.time.timetuple()))

                speeds = get_speeds(lats, lngs, times)

                new_points = [shapely.geometry.Point(xy) for xy in zip(lngs, lats)]
                new_point_gdf = gpd.GeoDataFrame(
                    crs={"init": "epsg:4326"}, geometry=new_points
                )
                new_point_gdf = new_point_gdf.to_crs(crs={"init": "epsg:3395"})

                # Get the coordinates for boundaries
                xs = list(new_point_gdf.geometry.x)
                ys = list(new_point_gdf.geometry.y)

                # Don't plot journeys that weren't in London
                if (
                    (new_point_gdf.geometry[0].x > x_lims[0])
                    and (new_point_gdf.geometry[0].x < x_lims[1])
                    and (new_point_gdf.geometry[0].y > y_lims[0])
                    and (new_point_gdf.geometry[0].y < y_lims[1])
                ):

                    all_xs += xs
                    all_ys += ys

                    # Get colours for points
                    if colored_black_end_points:
                        colors = np.append(
                            np.append(
                                [[0, 0, 0, 1]], scalarMap.to_rgba(speeds[1:-1]), axis=0
                            ),
                            [[0, 0, 0, 1]],
                            axis=0,
                        )
                    if colored_not_black_end_points:
                        colors = scalarMap.to_rgba(speeds)
                    if red:
                        colors = "red"

                    # Plot the journey
                    for key, value in map_configs.items():
                        if value["plotting_or_not"]:
                            # Pass the plotter functions some nonsense value of

                            (
                                journey_plots_for_moving_recents,
                                journey_plots_for_by_year,
                                journey_plots_shrinking,
                                journey_plots_bubbling,
                            ) = plotter_functions_dict[key](
                                journey_year_change=False,
                                journey_year="",
                                set_of_points=new_point_gdf,
                                ax=maps_dict[key][1],
                                journey_plots_for_moving_recents=[],
                                journey_plots_for_by_year=[],
                                n_concurrent=n_concurrent,
                                colors=colors,
                                journey_colour_score=0,
                                journey_plots_shrinking=[],
                                journey_plots_bubbling=[],
                                n_concurrent_shrinking=40,
                            )

                            n_journeys_plotted += 1
                            n_journeys_attempted += 1

                else:
                    print("Journey was outside London")

            # Don't plot non-cycling journeys
            else:
                n_non_cycling_activities += 1
                n_journeys_attempted += 1

        # If plotting fails
        except Exception as e:
            print(
                "Problem parsing "
                + (os.path.join(os.path.join(data_path, "cycling_data"), journey))
            )
            print(e)
            n_files_unparsable += 1
            unparsable_files.append(journey)
            n_journeys_attempted += 1

    # Calculate the boundaries from the coordinates
    xmin = min(all_xs)
    ymin = min(all_ys)
    xmax = max(all_xs)
    ymax = max(all_ys)

    x_range = xmax - xmin
    y_range = ymax - ymin

    x_lower = xmin - (x_range * 0.1)
    y_lower = ymin - (y_range * 0.1)
    x_upper = xmax + (x_range * 0.1)
    y_upper = ymax + (y_range * 0.1)

    # Add notes on the dates
    for key, value in map_configs.items():
        if value["plotting_or_not"]:
            journey_str_list = (journeys[0] + "-" + journeys[-1]).split("-")
            journey_str_list2 = [journey_str_list[i] for i in [0, 1, 2, 4, 5, 6]]
            timestr = (
                journey_str_list2[0]
                + r"/"
                + journey_str_list2[1]
                + r"/"
                + journey_str_list2[2]
                + " to "
                + journey_str_list2[3]
                + r"/"
                + journey_str_list2[4]
                + r"/"
                + journey_str_list2[5]
            )
            maps_dict[key][0].text(
                0.0125,
                0.025,
                timestr,
                horizontalalignment="left",
                verticalalignment="top",
                transform=maps_dict[key][1].transAxes,
                color=map_configs[key]["year_text"],
            )

    # Set the boundaries and save the figures
    for key, value in map_configs.items():
        if value["plotting_or_not"]:
            ax = maps_dict[key][1]
            ax.set_xlim(x_lower, x_upper)
            ax.set_ylim(y_lower, y_upper)

            # Save the final images
            print("Saving the final image...")
            filename = os.path.join(
                os.path.join(data_path, "results"),
                runstr
                + "_ad_hoc_cycling__"
                + key
                + "__"
                + timestr.replace(r"/", "_").replace(" ", "_")
                + ".png",
            )
            if (value["final_figure_output"] == "final_output") and value[
                "plotting_or_not"
            ]:
                maps_dict[key][0].savefig(
                    filename, bbox_inches="tight", ax=maps_dict[key][1]
                )


# Last x journeys
for this_no_journeys in number_journeys:
    journeys = sorted(
        [x for x in os.listdir(os.path.join(data_path, "cycling_data")) if ".gpx" in x]
    )[-(this_no_journeys):-1]
    make_one_off_plots(journeys)

# All journeys
if plotting_all:
    journeys = sorted([x for x in os.listdir("cycling_data") if ".gpx" in x])
    make_one_off_plots(journeys)

"""
# Random collection of journeys
all_journeys = sorted([x for x in os.listdir('cycling_data') if '.gpx' in x])
journey_refs = [1000, -1001, 1500, -500]
journeys = [all_journeys[i] for i in journey_refs]
make_one_off_plots(journeys)
"""
