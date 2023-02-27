import logging
import os
import time

import geopandas as gpd
import gpxpy
import numpy as np
import pandas as pd
import shapely

from configs.configs_maps import x_lims, y_lims

logger = logging.getLogger(__name__)


def convert_crop_base_map_layers(layer):
    """Crop the base_map_layers to the London area"""

    layer = crop(layer, x_lims, y_lims)

    layer = layer.to_crs("epsg:4326")  # Mercator

    return layer


def get_speeds(lats, lngs, times):
    """From a gpx file with name journey_name, get a numpy array of estimated speeds at each of those points"""
    try:
        df = pd.DataFrame({"x": lngs, "y": lats, "times": times})

        # needed to prevent df columns being in different order on linux vs windows?
        df = df[["x", "y", "times"]]

        df = pd.concat([df.shift(2), df.shift(-2)], axis=1)
        df.columns = ["x1", "y1", "time1", "x2", "y2", "time2"]

        for coordinate_variable in ["x1", "y1", "x2", "y2"]:
            df[coordinate_variable + "_rad"] = np.deg2rad(df[coordinate_variable])

        # Haversine formula for distance

        R = 3959.87433  # Radius of the Earth in miles

        df["distance"] = (
            2
            * R
            * np.arcsin(
                (
                    (np.sin((df["y2_rad"] - df["y1_rad"]) / 2) ** 2)
                    + (
                        np.cos(df["y1_rad"])
                        * np.cos(df["y2_rad"])
                        * ((np.sin((df["x2_rad"] - df["x1_rad"]) / 2)) ** 2)
                    )
                )
                ** 0.5
            )
        )
        df["speed"] = df["distance"] / (df["time2"] - df["time1"])

        # Remove values at the ends, replace them with uniform acceleration to and from 0
        speeds = df["speed"][2:-2].values
        speeds = np.append(speeds[0] / 2, np.append(speeds, speeds[-1] / 2))
        speeds = np.append(0, np.append(speeds, 0))

        speeds = speeds * 3600  # Convert from miles per second into miles per hours

        return speeds

    except Exception as e:
        logger.info(f"Exception: {e}")


def get_track(data_path, journey):
    gpx = gpxpy.parse(
        open(os.path.join(os.path.join(data_path, "cycling_data"), journey))
    )
    track = gpx.tracks[0]
    return track


def convert_to_gdf(track):
    lats = []
    lngs = []
    times = []

    for segment in track.segments:
        for point in segment.points:
            lats.append(point.latitude)
            lngs.append(point.longitude)
            times.append(time.mktime(point.time.timetuple()))

    speeds = get_speeds(lats, lngs, times)

    new_points = [shapely.geometry.Point(xy) for xy in zip(lngs, lats)]
    new_point_gdf = gpd.GeoDataFrame(crs="epsg:4326", geometry=new_points)
    return new_point_gdf, speeds


def starts_or_finishes_in_area_of_interest(new_point_gdf):
    return (
        (new_point_gdf.geometry[0].x > x_lims[0])
        and (new_point_gdf.geometry[0].x < x_lims[1])
        and (new_point_gdf.geometry[0].y > y_lims[0])
        and (new_point_gdf.geometry[0].y < y_lims[1])
    ) or (
        (new_point_gdf.geometry[len(new_point_gdf.geometry) - 1].x > x_lims[0])
        and (new_point_gdf.geometry[len(new_point_gdf.geometry) - 1].x < x_lims[1])
        and (new_point_gdf.geometry[len(new_point_gdf.geometry) - 1].y > y_lims[0])
        and (new_point_gdf.geometry[len(new_point_gdf.geometry) - 1].y < y_lims[1])
    )


def initial_checks_on_activity_type_and_location(plotter_inputs, track):
    cycling_activity = "cycling" in track.name.lower()
    plotter_inputs["set_of_points"], speeds = convert_to_gdf(track)
    activity_in_area_of_interest = starts_or_finishes_in_area_of_interest(
        plotter_inputs["set_of_points"]
    )
    return cycling_activity, activity_in_area_of_interest, plotter_inputs, speeds


def increment_counters_for_unplotted_journeys(
    activity_in_area_of_interest, cycling_activity, counters
):
    if not activity_in_area_of_interest:
        logger.info("Activity outside of London limits so not plotted")
        counters["n_journeys_outside_London"] += 1
        counters["n_journeys_attempted"] += 1

    if not cycling_activity:
        counters["n_non_cycling_activities"] += 1
        counters["n_journeys_attempted"] += 1

    return counters


def handle_problem_parsing_gpx_file(e, counters, journey):
    logger.info("Problem parsing " + (os.path.join("cycling_data", journey)))
    logger.info(e)
    counters["n_files_unparsable"] += 1
    counters["unparsable_files"].append(journey)
    counters["n_journeys_attempted"] += 1
    return counters


def crop(layer, x_lims, y_lims):
    lims_gs = gpd.GeoSeries(
        [
            shapely.geometry.Point(x_lims[0], y_lims[0]),
            shapely.geometry.Point(x_lims[0], y_lims[1]),
            shapely.geometry.Point(x_lims[1], y_lims[0]),
            shapely.geometry.Point(x_lims[1], y_lims[1]),
        ]
    )
    lims_gs = lims_gs.set_crs("epsg:4326")
    lims_gs = lims_gs.to_crs("epsg:27700")
    # Convert to maps' CRS to save time (takes more time to convert full maps)

    lims = list(lims_gs.total_bounds)

    x_len = lims[2] - lims[0]
    y_len = lims[3] - lims[1]

    layer = layer.cx[
        lims[0] - (x_len * 0.1) : lims[2] + (x_len * 0.1),
        lims[1] - (y_len * 0.1) : lims[3] + (y_len * 0.1),
    ]
    return layer
