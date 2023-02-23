import argparse
import datetime
import logging
import math
import os
import shutil
import time

import cv2
import geopandas as gpd
import gpxpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from tqdm import tqdm

from configs.configs_deployment import data_path
from configs.configs_maps import (
    x_lims,
    y_lims,
    scalarMap_time,
    base_layers_configs,
    scalarMap,
)
from configs.configs_run import (
    colored_black_end_points,
    colored_not_black_end_points,
    red,
    making_videos,
    image_interval,
    n_concurrent,
    n_concurrent_bubbling,
    n_concurrent_bubbling_end_points,
)

runstr = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(data_path, "results", f"{runstr}run.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


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
        lims[0]-(x_len*0.1) : lims[2]+(x_len*0.1),
        lims[1]-(y_len*0.1) : lims[3]+(y_len*0.1),
    ]
    return layer


def make_video(which_set_of_images, runstr, n_journeys_plotted):
    """Code adapted from http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    Turns a set of images into an .mp4 video"""
    logger.info("Making video of " + which_set_of_images + " images...")

    ext = "png"

    dir_path = os.path.join(
        os.path.join(data_path, "images_for_video"), which_set_of_images
    )
    output = os.path.join(
        os.path.join(data_path, "results"),
        runstr
        + "_"
        + which_set_of_images
        + "__first_"
        + str(n_journeys_plotted)
        + "_journeys.mp4",
    )

    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)

    # Make sure images are in right order
    images = sorted(images)

    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    # cv2.imshow("video", frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:

        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)

        out.write(frame)  # Write out frame to video

        # cv2.imshow("video", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):  # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    logger.info("The output video is {}".format(output))

    # Clear images folder
    folder = os.path.join(
        os.path.join(data_path, "images_for_video"), which_set_of_images
    )

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.info(f"Exception: {e}")


def pack_up_returns(
    journey_plots_for_moving_recents,
    journey_plots_for_by_year,
    journey_plots_shrinking,
    journey_plots_bubbling,
    end_points_bubbling,
    end_points_shrinking,
):
    return {
        "journey_plots_for_moving_recents": journey_plots_for_moving_recents,
        "journey_plots_for_by_year": journey_plots_for_by_year,
        "journey_plots_shrinking": journey_plots_shrinking,
        "journey_plots_bubbling": journey_plots_bubbling,
        "end_points_bubbling": end_points_bubbling,
        "end_points_shrinking": end_points_shrinking,
    }


def plotter_total(inputs):
    """For plotting the points on the total_fig map"""

    # Unpack inputs
    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]
    colors = inputs["colors"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]

    set_of_points.plot(
        categorical=False, legend=False, ax=ax, color=colors, alpha=0.1, markersize=0.2
    )

    # Pack up returns

    returns = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return returns


def plotter_moving_recents(inputs):
    """For plotting the points on the moving_recents_fig map"""

    # Unpack inputs
    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]
    n_concurrent = inputs["n_concurrent"]
    colors = inputs["colors"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]

    lats = []
    lngs = []

    for index, row in set_of_points.iterrows():
        lngs.append(row.geometry.x)
        lats.append(row.geometry.y)

    journey_plots_for_moving_recents.append(
        ax.scatter(lngs, lats, color=colors, s=3, marker="s", alpha=0.05)
    )

    if len(journey_plots_for_moving_recents) > n_concurrent:
        n_to_remove = len(journey_plots_for_moving_recents) - n_concurrent
        for i in range(n_to_remove):
            journey_plots_for_moving_recents[i].remove()
        for j in range(n_to_remove):
            del journey_plots_for_moving_recents[j]

    # Pack up returns

    returns = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return returns


def plotter_by_year(inputs):
    """For plotting the points on the by_year map"""

    # Unpack inputs
    journey_year_change = inputs["journey_year_change"]
    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]
    colors = inputs["colors"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]

    lats = []
    lngs = []

    for index, row in set_of_points.iterrows():
        lngs.append(row.geometry.x)
        lats.append(row.geometry.y)

    if journey_year_change:
        for i in range(len(journey_plots_for_by_year)):
            journey_plots_for_by_year[i].remove()
        journey_plots_for_by_year = []

    journey_plots_for_by_year.append(
        ax.scatter(lngs, lats, color=colors, s=2, marker="s", alpha=1)
    )

    # Pack up returns

    returns = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return returns


def plotter_all_journeys_thick(inputs):
    """For plotting the points on the by_year map"""

    # Unpack inputs
    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]
    colors = inputs["colors"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]

    lats = []
    lngs = []

    for index, row in set_of_points.iterrows():
        lngs.append(row.geometry.x)
        lats.append(row.geometry.y)

    ax.scatter(lngs, lats, color=colors, s=2, marker="s", alpha=1)

    # Pack up returns

    returns = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return returns


def plotter_no_roads(inputs):
    """For plotting the points on the total_fig_no_roads map"""

    # Unpack inputs
    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]

    set_of_points.plot(
        categorical=False, legend=False, ax=ax, color="white", alpha=0.1, markersize=0.1
    )
    # Pack up returns

    returns = {
        "journey_plots_for_moving_recents": journey_plots_for_moving_recents,
        "journey_plots_for_by_year": journey_plots_for_by_year,
        "journey_plots_shrinking": journey_plots_shrinking,
        "journey_plots_bubbling": journey_plots_bubbling,
        "end_points_bubbling": end_points_bubbling,
        "end_points_shrinking": end_points_shrinking,
    }

    return returns


def plotter_dark_colours_by_time(inputs):
    """For plotting the points on the dark_colours_by_time map"""

    # Unpack inputs

    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]

    journey_colour_score = inputs["journey_colour_score"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]

    colors = scalarMap_time.to_rgba(journey_colour_score * 0.85)

    set_of_points.plot(
        categorical=False, legend=False, ax=ax, color=colors, alpha=0.06, markersize=0.3
    )
    # Pack up returns

    returns = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return returns


def plotter_alpha_one(inputs):
    """For plotting the points on the total_fig_alpha_1 map"""

    # Unpack inputs

    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]
    colors = inputs["colors"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]

    set_of_points.plot(
        categorical=False, legend=False, ax=ax, color=colors, alpha=1, markersize=0.05
    )
    # Pack up returns

    returns = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return returns


def plotter_all_journeys_shrinking(inputs):
    """For plotting the points on the moving_recents_fig map"""

    # Unpack inputs

    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]
    colors = inputs["colors"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]

    lats = []
    lngs = []

    for index, row in set_of_points.iterrows():
        lngs.append(row.geometry.x)
        lats.append(row.geometry.y)

    journey_plots_shrinking.append(
        ax.scatter(
            lngs,
            lats,
            color=colors,
            s=50,
            # marker="s",
            alpha=0.4,
        )
    )

    if len(journey_plots_shrinking) > 0:
        for points in journey_plots_shrinking:
            if points._sizes > 1:
                points.set_sizes(points._sizes * 0.95)
                points.set_alpha(points._alpha * 0.99)

    # Pack up returns

    returns = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return returns


def plotter_all_journeys_bubbling_off(inputs):
    """For plotting the points on the moving_recents_fig map"""

    # Unpack inputs

    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]
    colors = inputs["colors"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]
    n_concurrent_bubbling = inputs["n_concurrent_bubbling"]

    lats = []
    lngs = []

    for index, row in set_of_points.iterrows():
        lngs.append(row.geometry.x)
        lats.append(row.geometry.y)

    journey_plots_bubbling.append(
        ax.scatter(
            lngs,
            lats,
            color=colors,
            s=10,
            # marker="s",
            alpha=1,
        )
    )

    for i in range(len(journey_plots_bubbling)):
        journey_plots_bubbling[i].set_sizes(journey_plots_bubbling[i]._sizes * 1.2)
        if journey_plots_bubbling[i]._alpha > 0.0005:
            journey_plots_bubbling[i].set_alpha(journey_plots_bubbling[i]._alpha * 0.82)

    if len(journey_plots_bubbling) > n_concurrent_bubbling:
        n_to_remove = len(journey_plots_bubbling) - n_concurrent_bubbling
        for i in range(n_to_remove):
            journey_plots_bubbling[i].remove()
        for j in range(n_to_remove):
            del journey_plots_bubbling[j]

    # Pack up returns

    returns = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return returns


def plotter_end_points(inputs):
    """For plotting the points on the total_fig_alpha_1 map"""

    # Unpack inputs

    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]

    # Get first and last points
    lngs = [
        set_of_points.geometry[0].x,
        set_of_points.geometry[len(set_of_points.geometry) - 1].x,
    ]
    lats = [
        set_of_points.geometry[0].y,
        set_of_points.geometry[len(set_of_points.geometry) - 1].y,
    ]

    new_points = [shapely.geometry.Point(xy) for xy in zip(lngs, lats)]

    ax.scatter(lngs, lats, color=["green", "blue"], s=6, alpha=0.4)

    # Pack up returns

    journey_plots = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return journey_plots


def plotter_end_points_bubbling_off(inputs):
    """For plotting the points on the moving_recents_fig map"""

    # Unpack inputs

    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]
    n_concurrent_bubbling_end_points = inputs["n_concurrent_bubbling_end_points"]

    # Get first and last points
    lngs = [
        set_of_points.geometry[0].x,
        set_of_points.geometry[len(set_of_points.geometry) - 1].x,
    ]
    lats = [
        set_of_points.geometry[0].y,
        set_of_points.geometry[len(set_of_points.geometry) - 1].y,
    ]

    end_points_bubbling.append(
        ax.scatter(
            lngs,
            lats,
            color=["green", "blue"],
            s=10,
            # marker="s",
            alpha=1,
        )
    )

    for i in range(len(end_points_bubbling)):
        end_points_bubbling[i].set_sizes(end_points_bubbling[i]._sizes * 1.1)
        if end_points_bubbling[i]._alpha > 0.0005:
            end_points_bubbling[i].set_alpha(end_points_bubbling[i]._alpha * 0.95)

    if len(end_points_bubbling) > n_concurrent_bubbling_end_points:
        n_to_remove = len(end_points_bubbling) - n_concurrent_bubbling_end_points
        for i in range(n_to_remove):
            end_points_bubbling[i].remove()
        for j in range(n_to_remove):
            del end_points_bubbling[j]

    # Pack up returns

    returns = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return returns


def plotter_end_points_shrinking(inputs):
    """For plotting the points on the moving_recents_fig map"""

    # Unpack inputs
    set_of_points = inputs["set_of_points"]
    ax = inputs["ax"]
    journey_plots_for_moving_recents = inputs["journey_plots_for_moving_recents"]
    journey_plots_for_by_year = inputs["journey_plots_for_by_year"]
    journey_plots_shrinking = inputs["journey_plots_shrinking"]
    journey_plots_bubbling = inputs["journey_plots_bubbling"]
    end_points_bubbling = inputs["end_points_bubbling"]
    end_points_shrinking = inputs["end_points_shrinking"]

    # Get first and last points
    lngs = [
        set_of_points.geometry[0].x,
        set_of_points.geometry[len(set_of_points.geometry) - 1].x,
    ]
    lats = [
        set_of_points.geometry[0].y,
        set_of_points.geometry[len(set_of_points.geometry) - 1].y,
    ]

    end_points_shrinking.append(
        ax.scatter(
            lngs,
            lats,
            color=["green", "blue"],
            s=400,
            # marker="s",
            alpha=0.4,
        )
    )

    if len(end_points_shrinking) > 0:
        for points in end_points_shrinking:
            if points._sizes > 10:
                points.set_sizes(points._sizes * 0.90)
                points.set_alpha(points._alpha * 0.985)

    # Pack up returns

    returns = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return returns


def which_layers_needed(map_configs):
    all_layers = []
    for key, value in map_configs.items():
        for layer in value["layers"]:
            all_layers.append(layer)
    return list(set(all_layers))


def read_in_original_file(layer_name):
    if layer_name == "parks":
        map_path = os.path.join(
            data_path,
            r"OS_base_maps",
            "OS Open Greenspace (ESRI Shape File) TQ",
            "data",
            base_layers_configs[layer_name][0],
        )

    else:
        map_path = os.path.join(
            data_path,
            "OS_base_maps",
            "OS OpenMap Local (ESRI Shape File) TQ",
            "data",
            base_layers_configs[layer_name][0],
        )
    layer = gpd.read_file(map_path)
    return layer


# Put plotter functions in a dictionary to call them more easily
plotter_functions_dict = {
    "overall": plotter_total,
    "running_recents": plotter_moving_recents,
    "dark": plotter_no_roads,
    "dark_colours_by_time": plotter_dark_colours_by_time,
    "overall_alpha_1": plotter_alpha_one,
    "by_year": plotter_by_year,
    "overall_thick": plotter_all_journeys_thick,
    "overall_shrinking": plotter_all_journeys_shrinking,
    "overall_bubbling_off": plotter_all_journeys_bubbling_off,
    "end_points": plotter_end_points,
    "end_points_bubbling": plotter_end_points_bubbling_off,
    "end_points_shrinking": plotter_end_points_shrinking,
}


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


def read_in_convert_base_maps(map_configs):

    #
    # CONSTRUCT THE BASE layers - read the data in, set the colours and convert to Mercator projections
    #

    base_layers = {}

    # Find exactly which layers are needed, read them in
    layers_to_read = which_layers_needed(map_configs)

    for layer_name in layers_to_read:
        if f"{layer_name}.shp" not in os.listdir(
            os.path.join(data_path, "cropped_maps")
        ):
            logger.info(f"Reading in original file for {layer_name} map")
            layer = read_in_original_file(layer_name)
            logger.info(f"Cropping and converting original map for {layer_name} map")
            layer = convert_crop_base_map_layers(layer)
            logger.info(f"Saving processed {layer_name} map")
            layer.to_file(os.path.join(data_path, "cropped_maps", f"{layer_name}.shp"))
            base_layers[layer_name] = layer
        else:
            logger.info(f"Reading in processed {layer_name} map")
            base_layers[layer_name] = gpd.read_file(
                os.path.join(data_path, "cropped_maps", f"{layer_name}.shp")
            )

    return base_layers


def clear_out_old_folders_and_make_new(map_configs):

    #
    # CLEAR OUT THE images_for_video FOLDER
    #

    # Remove old folders
    if len(os.listdir(os.path.join(data_path, "images_for_video"))) > 0:
        for folder in os.listdir(os.path.join(data_path, "images_for_video")):
            shutil.rmtree(
                os.path.join(os.path.join(data_path, "images_for_video"), folder)
            )

    # Generate new folders
    for key, item in map_configs.items():
        os.mkdir(os.path.join(os.path.join(data_path, "images_for_video"), key))


def plot_base_map_layers(base_layers, map_configs):

    #
    # PLOT THE BASE LAYERS
    #

    # Create maps_dict for storing the map objects in
    maps_dict = {}
    # Plot the base layers for all maps being made
    for key, value in map_configs.items():
        fig, ax = plt.subplots(figsize=(14 * 2, 6 * 2))
        ax.set_xlim(x_lims[0], x_lims[1])
        ax.set_ylim(y_lims[0], y_lims[1])
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)
        y_axis = ax.axes.get_yaxis()
        y_axis.set_visible(False)
        if key in ["dark", "dark_colours_by_time"] and (
            "tidal_water" in value["layers"]
        ):
            logger.info(f"Plotting tidal_water for {key}")
            ax.set_facecolor(tuple([x / 255 for x in [0, 0, 0]]))
            maps_dict[key] = [fig, ax, x_axis, y_axis]
            maps_dict[key][1] = base_layers["tidal_water"].plot(
                categorical=False,
                legend=False,
                ax=maps_dict[key][1],
                color=tuple([x / 255 for x in [0, 0, 60]]),
                zorder=0,
            )
        else:
            for layer in value["layers"]:
                logger.info(f"Plotting {layer} for {key}")
                ax.set_facecolor(tuple([x / 255 for x in [232, 237, 232]]))
                maps_dict[key] = [fig, ax, x_axis, y_axis]
                maps_dict[key][1] = base_layers[layer].plot(
                    categorical=False,
                    legend=False,
                    ax=maps_dict[key][1],
                    color=base_layers_configs[layer][1],
                    zorder=0,
                )
    return maps_dict


def parse_the_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--no_journeys", type=str, default="", help="How many journeys to include?"
    )
    parser.add_argument(
        "--is_debug",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="Running a faster version of code for debugging?",
    )
    return parser.parse_args()


def get_journey_files(no_journeys):

    logger.info("Getting the journey files")

    journey_files = sorted(
        [x for x in os.listdir(os.path.join(data_path, "cycling_data")) if ".gpx" in x]
    )

    if no_journeys == "":
        no_journeys = len(journey_files)
        attempting_all = True
    else:
        no_journeys = int(no_journeys)
        attempting_all = False

    return journey_files, no_journeys, attempting_all


def make_first_frames(counters, journeys, text_vars, maps_dict, map_configs):
    for map_scheme_name, map_scheme_configs in map_configs.items():
        filename = os.path.join(
            os.path.join(os.path.join(data_path, "images_for_video"), map_scheme_name),
            "first_" + str(counters["n_journeys_plotted"]).zfill(4) + "_journeys.png",
        )

        timestr = counters["first_year"] + " - " + journeys[0].split("-")[0]
        timestr_moving_recents = journeys[0].split("-")[0]
        if map_scheme_name in ["running_recents", "by_year"]:
            text_vars[map_scheme_name] = maps_dict[map_scheme_name][0].text(
                0.9650,
                0.985,
                timestr_moving_recents,
                horizontalalignment="left",
                verticalalignment="top",
                transform=maps_dict[map_scheme_name][1].transAxes,
                color=map_configs[map_scheme_name]["year_text"],
            )
        else:
            text_vars[map_scheme_name] = maps_dict[map_scheme_name][0].text(
                0.9250,
                0.985,
                timestr,
                horizontalalignment="left",
                verticalalignment="top",
                transform=maps_dict[map_scheme_name][1].transAxes,
                color=map_configs[map_scheme_name]["year_text"],
            )

        maps_dict[map_scheme_name][0].savefig(
            filename,
            bbox_inches="tight",
        )

    return timestr, timestr_moving_recents, text_vars


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


def save_by_year_plot_if_new_year(
    new_journey_year, counters, maps_dict, map_configs, runstr
):
    journey_year_change = False
    if new_journey_year != counters["journey_year"]:
        journey_year_change = True
        if "by_year" in map_configs.keys():
            filename = os.path.join(
                os.path.join(data_path, "results"),
                runstr + "_cycling_in_year_" + counters["journey_year"] + ".png",
            )
            maps_dict["by_year"][0].savefig(
                filename,
                bbox_inches="tight",
                ax=maps_dict["by_year"][1],
            )  # Output the by_year maps at the end of the year
    return journey_year_change


def set_colors(colored_black_end_points, colored_not_black_end_points, red, speeds):
    if colored_black_end_points:
        colors = np.append(
            np.append([[0, 0, 0, 1]], scalarMap.to_rgba(speeds[1:-1]), axis=0),
            [[0, 0, 0, 1]],
            axis=0,
        )
    if colored_not_black_end_points:
        colors = scalarMap.to_rgba(speeds)
    if red:
        colors = "red"
    return colors


def update_timestr_if_necessary(
    timestr,
    new_timestr,
    text_vars,
    map_scheme_name,
    maps_dict,
    new_timestr_moving_recents,
    map_configs,
):
    if timestr != new_timestr:
        text_vars[map_scheme_name].set_visible(False)
        if map_scheme_name in ["running_recents", "by_year"]:
            text_vars[map_scheme_name] = maps_dict[map_scheme_name][0].text(
                0.9650,
                0.985,
                new_timestr_moving_recents,
                horizontalalignment="left",
                verticalalignment="top",
                transform=maps_dict[map_scheme_name][1].transAxes,
                color=map_configs[map_scheme_name]["year_text"],
            )
        else:
            text_vars[map_scheme_name] = maps_dict[map_scheme_name][0].text(
                0.9250,
                0.985,
                new_timestr,
                horizontalalignment="left",
                verticalalignment="top",
                transform=maps_dict[map_scheme_name][1].transAxes,
                color=map_configs[map_scheme_name]["year_text"],
            )
    return text_vars


def save_frames_for_video(making_videos, counters, maps_dict, map_scheme_name):
    if making_videos:
        if (counters["n_journeys_plotted"] + 1) % image_interval == 0:
            filename = os.path.join(
                os.path.join(
                    os.path.join(data_path, "images_for_video"),
                    map_scheme_name,
                ),
                "first_"
                + str(counters["n_journeys_plotted"] + 1).zfill(4)
                + "_journeys.png",
            )
            maps_dict[map_scheme_name][0].savefig(
                filename,
                bbox_inches="tight",
            )


def increment_counters(counters, new_timestr, new_timestr_moving_recents):
    counters["n_journeys_plotted"] += 1
    counters["n_journeys_attempted"] += 1

    timestr = new_timestr
    timestr_moving_recents = new_timestr_moving_recents
    counters["journey_year"] = counters["new_journey_year"]
    return counters, timestr, timestr_moving_recents


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


def make_all_other_frames(
    journey_files,
    attempting_all,
    no_journeys,
    start_time,
    maps_dict,
    text_vars,
    timestr,
    journey_plots,
    counters,
    map_configs,
):
    counters["journey_year"] = counters["first_year"]

    plotter_inputs = {
        "n_concurrent": n_concurrent,
        "n_concurrent_bubbling": n_concurrent_bubbling,
        "n_concurrent_bubbling_end_points": n_concurrent_bubbling_end_points,
    }
    plotter_inputs.update(journey_plots)

    for journey in tqdm(journey_files):
        # for the dark_colours_by_time plot
        journey_colour_score = counters["n_journeys_plotted"] / no_journeys

        try:
            track = get_track(data_path, journey)
        except Exception as e:
            counters = handle_problem_parsing_gpx_file(e, counters, journey)
            pass

        (
            cycling_activity,
            activity_in_area_of_interest,
            plotter_inputs,
            speeds,
        ) = initial_checks_on_activity_type_and_location(plotter_inputs, track)

        if len(plotter_inputs["set_of_points"]) < 5:
            counters["n_files_too_few_points"] += 1
            counters["files_too_few_points"].append(journey)
            continue

        counters = increment_counters_for_unplotted_journeys(
            activity_in_area_of_interest, cycling_activity, counters
        )

        if cycling_activity and activity_in_area_of_interest:

            new_timestr = counters["first_year"] + " - " + journey.split("-")[0]
            new_timestr_moving_recents = journey.split("-")[0]
            counters["new_journey_year"] = journey.split("-")[0]

            plotter_inputs["journey_year_change"] = save_by_year_plot_if_new_year(
                counters["new_journey_year"],
                counters,
                maps_dict,
                map_configs,
                runstr,
            )

            plotter_inputs["colors"] = set_colors(
                colored_black_end_points,
                colored_not_black_end_points,
                red,
                speeds,
            )

            for map_scheme_name, map_scheme_configs in map_configs.items():

                plotter_inputs.update(
                    {
                        "journey_year": counters["journey_year"],
                        "ax": maps_dict[map_scheme_name][1],
                        "journey_colour_score": journey_colour_score,
                    }
                )
                plotter_inputs.update(
                    plotter_functions_dict[map_scheme_name](plotter_inputs)
                )

                text_vars = update_timestr_if_necessary(
                    timestr,
                    new_timestr,
                    text_vars,
                    map_scheme_name,
                    maps_dict,
                    new_timestr_moving_recents,
                    map_configs,
                )

                save_frames_for_video(
                    making_videos, counters, maps_dict, map_scheme_name
                )

            counters, timestr, timestr_moving_recents = increment_counters(
                counters, new_timestr, new_timestr_moving_recents
            )

            if counters["n_journeys_plotted"] >= no_journeys:
                break

    return counters


def make_final_by_year_image(counters, maps_dict, map_configs):

    # Save the final by_year image - done separately to the other image formats below,
    # due to the difference in filename
    if "by_year" in map_configs.keys():
        filename = os.path.join(
            os.path.join(data_path, "results"),
            runstr + "_cycling_in_year_" + counters["journey_year"] + "_to_date.png",
        )
        maps_dict["by_year"][0].savefig(
            filename, bbox_inches="tight", ax=maps_dict["by_year"][1]
        )  # Output the by_year maps at the end of the year

    # Save the final images
    logger.info("Saving the final images...")
    for map_scheme_name, map_scheme_configs in map_configs.items():
        filename = os.path.join(
            os.path.join(data_path, "results"),
            runstr
            + "_"
            + map_scheme_name
            + "__first_"
            + str(counters["n_journeys_plotted"])
            + "_journeys.png",
        )
        maps_dict[map_scheme_name][0].savefig(
            filename,
            bbox_inches="tight",
        )


def additional_frames_journeys_fading_out(
    journey_files, maps_dict, journey_plots, counters, map_configs
):
    # Add on some more frames to running_recents, where older journey_files disappear one by one
    # todo: refactor function
    if making_videos:
        index = 0
        if "running_recents" in map_configs.keys():
            timestr = counters["first_year"] + " - " + journey_files[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            number_plots_to_do = len(journey_plots["journey_plots_for_moving_recents"])
            while len(journey_plots["journey_plots_for_moving_recents"]) > 0:
                logger.info(
                    f"HA! Additional figure generation: making additional image {index + 1} of "
                    f"{number_plots_to_do} for the running_recents video - fading out the old journey_files"
                )
                # Remove line from plot and journey_plots_for_moving_recents list
                journey_plots["journey_plots_for_moving_recents"][0].remove()
                del journey_plots["journey_plots_for_moving_recents"][0]
                # Save the images to files
                if index % image_interval == 0:
                    filename = os.path.join(
                        os.path.join(
                            os.path.join(data_path, "images_for_video"),
                            "running_recents",
                        ),
                        "first_"
                        + str(counters["n_journeys_plotted"] + index + 1).zfill(4)
                        + "_journeys.png",
                    )
                    maps_dict["running_recents"][0].savefig(
                        filename,
                        bbox_inches="tight",
                    )
                index += 1

    # Add on some more frames to bubbling, where older journey_files disappear one by one
    if making_videos:
        index = 0
        if "overall_bubbling_off" in map_configs.keys():
            timestr = counters["first_year"] + " - " + journey_files[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            number_plots_to_do = len(journey_plots["journey_plots_bubbling"])

            while len(journey_plots["journey_plots_bubbling"]) > 0:
                logger.info(
                    f"HA! Additional figure generation: making additional image {index + 1} of "
                    f"{number_plots_to_do} for the bubbling video - fading out the old journey_files"
                )
                # Remove line from plot and journey_plots_for_moving_recents list
                journey_plots["journey_plots_bubbling"][0].remove()
                del journey_plots["journey_plots_bubbling"][0]
                for i in range(len(journey_plots["journey_plots_bubbling"])):
                    journey_plots["journey_plots_bubbling"][i].set_sizes(
                        journey_plots["journey_plots_bubbling"][i]._sizes * 1.2
                    )
                    if journey_plots["journey_plots_bubbling"][i]._alpha > 0.0005:
                        journey_plots["journey_plots_bubbling"][i].set_alpha(
                            journey_plots["journey_plots_bubbling"][i]._alpha * 0.82
                        )
                # Save the images to files
                if index % image_interval == 0:
                    filename = os.path.join(
                        os.path.join(
                            os.path.join(data_path, "images_for_video"),
                            "overall_bubbling_off",
                        ),
                        "first_"
                        + str(counters["n_journeys_plotted"] + index + 1).zfill(4)
                        + "_journeys.png",
                    )
                    maps_dict["overall_bubbling_off"][0].savefig(
                        filename,
                        bbox_inches="tight",
                    )
                index += 1

    # Add on some more frames to journey_plots['end_points_bubbling'], where older journey_files disappear one by one
    if making_videos:
        index = 0
        if "end_points_bubbling" in map_configs.keys():
            timestr = counters["first_year"] + " - " + journey_files[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            number_plots_to_do = len(journey_plots["end_points_bubbling"])

            while len(journey_plots["end_points_bubbling"]) > 0:
                logger.info(
                    f"HA! Additional figure generation: making additional image {index + 1} of "
                    f"{number_plots_to_do} for the end_points_bubbling video - fading out the old journey_files"
                )
                # Remove line from plot and journey_plots_for_moving_recents list
                journey_plots["end_points_bubbling"][0].remove()
                del journey_plots["end_points_bubbling"][0]
                for i in range(len(journey_plots["end_points_bubbling"])):
                    journey_plots["end_points_bubbling"][i].set_sizes(
                        journey_plots["end_points_bubbling"][i]._sizes * 1.1
                    )
                    if journey_plots["end_points_bubbling"][i]._alpha > 0.0005:
                        journey_plots["end_points_bubbling"][i].set_alpha(
                            journey_plots["end_points_bubbling"][i]._alpha * 0.95
                        )
                # Save the images to files
                if index % image_interval == 0:
                    filename = os.path.join(
                        os.path.join(
                            os.path.join(data_path, "images_for_video"),
                            "end_points_bubbling",
                        ),
                        "first_"
                        + str(counters["n_journeys_plotted"] + index + 1).zfill(4)
                        + "_journeys.png",
                    )
                    maps_dict["end_points_bubbling"][0].savefig(
                        filename,
                        bbox_inches="tight",
                    )
                index += 1

    # Add on some more frames to shrinking, where older gradually shrink
    if making_videos:
        index = 0
        if "overall_shrinking" in map_configs.keys():
            timestr = counters["first_year"] + " - " + journey_files[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function

            sizes = [
                points._sizes for points in journey_plots["journey_plots_shrinking"]
            ]
            sizes = [size[0] for size in sizes]
            max_sizes = max(sizes)

            rat = 0.95
            min_size = 1

            number_plots_to_do = math.ceil(
                (np.log(min_size) - np.log(max_sizes)) / np.log(rat)
            )

            for i in range(number_plots_to_do):
                logger.info(
                    f"HA! Additional figure generation: making additional image {index + 1} of "
                    f"{number_plots_to_do} for the shrinking video - fading out the old journey_files"
                )

                for points in journey_plots["journey_plots_shrinking"]:
                    if points._sizes > min_size:
                        points.set_sizes(points._sizes * 0.95)
                        points.set_alpha(points._alpha * 0.99)

                # Save the images to files
                if index % image_interval == 0:
                    filename = os.path.join(
                        os.path.join(
                            os.path.join(data_path, "images_for_video"),
                            "overall_shrinking",
                        ),
                        "first_"
                        + str(counters["n_journeys_plotted"] + index + 1).zfill(4)
                        + "_journeys.png",
                    )
                    maps_dict["overall_shrinking"][0].savefig(
                        filename,
                        bbox_inches="tight",
                    )
                index += 1

    # Add on some more frames to journey_plots['end_points_shrinking'], where older gradually shrink
    if making_videos:
        index = 0
        if "end_points_shrinking" in map_configs.keys():
            timestr = counters["first_year"] + " - " + journey_files[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            sizes = [points._sizes for points in journey_plots["end_points_shrinking"]]
            sizes = [size[0] for size in sizes]
            max_sizes = max(sizes)

            rat = 0.9
            min_size = 10

            number_plots_to_do = math.ceil(
                (np.log(min_size) - np.log(max_sizes)) / np.log(rat)
            )

            for i in range(number_plots_to_do):
                logger.info(
                    f"HA! Additional figure generation: making additional image {index + 1} of "
                    f"{number_plots_to_do} for the end_points_shrinking video - fading out the old journey_files"
                )


                for points in journey_plots["end_points_shrinking"]:
                    if points._sizes > min_size:
                        points.set_sizes(points._sizes * rat)
                        points.set_alpha(points._alpha * 0.985)

                # Save the images to files
                if index % image_interval == 0:
                    filename = os.path.join(
                        os.path.join(
                            os.path.join(data_path, "images_for_video"),
                            "end_points_shrinking",
                        ),
                        "first_"
                        + str(counters["n_journeys_plotted"] + index + 1).zfill(4)
                        + "_journeys.png",
                    )
                    maps_dict["end_points_shrinking"][0].savefig(
                        filename,
                        bbox_inches="tight",
                    )
                index += 1


def make_all_videos(counters, map_configs):
    # Make the videos
    if making_videos:
        for map_scheme_name, map_scheme_configs in map_configs.items():
            make_video(map_scheme_name, runstr, counters["n_journeys_plotted"])


def clear_out_images_for_video_folder(making_videos):
    # Clear out the images_for_video folder
    if making_videos:
        if len(os.listdir(os.path.join(data_path, "images_for_video"))) > 0:
            for folder in os.listdir(os.path.join(data_path, "images_for_video")):
                shutil.rmtree(
                    os.path.join(os.path.join(data_path, "images_for_video"), folder)
                )


def set_up_plot_lists_and_counters(journeys):

    start_time = datetime.datetime.now()

    journey_plots = {
        "journey_plots_for_moving_recents": [],
        "journey_plots_for_by_year": [],
        "journey_plots_shrinking": [],
        "journey_plots_bubbling": [],
        "end_points_shrinking": [],
        "end_points_bubbling": [],
    }

    counters = {
        "n_journeys_attempted": 0,
        "n_journeys_plotted": 0,
        "n_journeys_outside_London": 0,
        "n_non_cycling_activities": 0,
        "n_files_unparsable": 0,
        "n_files_too_few_points": 0,
        "files_too_few_points": [],
        "unparsable_files": [],
        "first_year": journeys[0].split("-")[0],
    }

    text_vars = {}

    return (
        start_time,
        journey_plots,
        counters,
        text_vars,
    )


def reduce_map_configs_to_maps_being_made(map_configs, which_maps_to_make):

    maps_to_make = [k for k, v in which_maps_to_make.items() if v]

    map_configs = {k: v for k, v in map_configs.items() if k in maps_to_make}

    return map_configs
