import argparse
import datetime
import logging
import math
import os
import shutil

import cv2
import geopandas as gpd
import gpxpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

from configs import (
    x_lims,
    y_lims,
    data_path,
    x_lims_broader,
    y_lims_broader,
    scalarMap_time,
    base_layers_configs,
    image_interval,
    n_concurrent,
    making_videos,
    colored_black_end_points,
    colored_not_black_end_points,
    red,
    n_concurrent_bubbling,
    n_concurrent_bubbling_end_points,
    scalarMap,
)

logger = logging.getLogger(__name__)


def timesince(time, percent_done):
    """Time since is used for measuring time since this same function was called, and for estimating time remaining"""
    diff = datetime.datetime.now() - time
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    try:
        remaining_diff = (diff / percent_done) * 100 - diff
        remaining_days, remaining_seconds = remaining_diff.days, remaining_diff.seconds
        remaining_hours = remaining_days * 24 + remaining_seconds // 3600
        remaining_minutes = (remaining_seconds % 3600) // 60
        remaining_seconds = remaining_seconds % 60
        return (
            str(hours)
            + " hours, "
            + str(minutes)
            + " minutes, "
            + str(seconds)
            + " seconds taken so far"
            + "\n"
            + "Estimated "
            + str(remaining_hours)
            + " hours, "
            + str(remaining_minutes)
            + " minutes, "
            + str(remaining_seconds)
            + " seconds to completion"
            + "\n"
            + "Estimated completion time: "
            + (datetime.datetime.now() + remaining_diff).strftime("%H:%M:%S")
        )
    except:
        return "Cannot calculate times elapsed and remaining at this time"


def time_diff(time1, time2):
    """Returns a sting of duration between two times"""
    diff = time2 - time1
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return (
        str(hours)
        + " hours, "
        + str(minutes)
        + " minutes, "
        + str(seconds)
        + " seconds taken"
    )


def make_video(which_set_of_images, runstr, n_journeys_plotted):
    """Code adapted from http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    Turns a set of images into an .mp4 video"""
    print("\nMaking video of " + which_set_of_images + " images...")

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

    print("The output video is {}".format(output))

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
            print(e)


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

    returns = pack_up_returns(
        journey_plots_for_moving_recents,
        journey_plots_for_by_year,
        journey_plots_shrinking,
        journey_plots_bubbling,
        end_points_bubbling,
        end_points_shrinking,
    )

    return returns


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
        if value["plotting_or_not"]:
            for layer in value["layers"]:
                all_layers.append(layer)
    return list(set(all_layers))


def read_in_required_layers(layers_to_read):
    base_layers = {}
    for layer_to_read in layers_to_read:
        print("Reading in " + layer_to_read + " map")
        try:
            base_layers[layer_to_read] = gpd.read_file(
                os.path.join(
                    data_path,
                    os.path.join(
                        r"OS_base_maps/OS OpenMap Local (ESRI Shape File) TQ/data/",
                        base_layers_configs[layer_to_read][0],
                    ),
                )
            )
        except:
            base_layers[layer_to_read] = gpd.read_file(
                os.path.join(
                    data_path,
                    os.path.join(
                        r"OS_base_maps/OS Open Greenspace (ESRI Shape File) TQ/data/",
                        base_layers_configs[layer_to_read][0],
                    ),
                )
            )
    return base_layers


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


def convert_crop_base_map_layers(base_layers):
    """Crop the base_map_layers to the London area"""
    print("")
    for key, value in base_layers.items():
        print("Converting coordinates in " + key)
        value["geometry"] = (
            value["geometry"]
            .to_crs({"init": "epsg:3395"})
            .cx[
                x_lims_broader[0] : x_lims_broader[1],
                y_lims_broader[0] : y_lims_broader[1],
            ]
        )  # Mercator
        print("Cropping " + key + " layer to London area")
    return base_layers


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
        print(e)


def read_in_convert_base_maps(map_configs):

    #
    # CONSTRUCT THE BASE layers - read the data in, set the colours and convert to Mercator projections
    #

    # Find exactly which layers are needed, read them in
    layers_to_read = which_layers_needed(map_configs)
    base_layers = read_in_required_layers(layers_to_read)
    # Convert required layers to Mercator projection, crop layers down
    base_layers = convert_crop_base_map_layers(base_layers)
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
        if item["plotting_or_not"]:
            os.mkdir(os.path.join(os.path.join(data_path, "images_for_video"), key))


def plot_base_map_layers(base_layers, map_configs):

    #
    # PLOT THE BASE LAYERS
    #

    # Create maps_dict for storing the map objects in
    maps_dict = {}
    # Plot the base layers for all maps being made
    for key, value in map_configs.items():
        if value["plotting_or_not"]:
            print("")
            fig, ax = plt.subplots(figsize=(14 * 2, 6 * 2))
            ax.set_xlim(x_lims[0], x_lims[1])
            ax.set_ylim(y_lims[0], y_lims[1])
            x_axis = ax.axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = ax.axes.get_yaxis()
            y_axis.set_visible(False)
            if key in ["dark", "dark_colours_by_time"] and ("tidal" in value["layers"]):
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


def get_start_time():
    overall_start_time = datetime.datetime.now()
    runstr = overall_start_time.strftime("%Y_%m_%d__%H_%M_")
    return overall_start_time, runstr


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
        if map_scheme_configs["plotting_or_not"]:
            filename = os.path.join(
                os.path.join(
                    os.path.join(data_path, "images_for_video"), map_scheme_name
                ),
                "first_"
                + str(counters["n_journeys_plotted"]).zfill(4)
                + "_journeys.png",
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
                fig=maps_dict[map_scheme_name][0],
                ax=maps_dict[map_scheme_name][1],
            )

            return timestr


def make_all_other_frames(
    journey_files,
    attempting_all,
    no_journeys,
    start_time,
    time,
    maps_dict,
    runstr,
    text_vars,
    timestr,
    journey_plots,
    counters,
    map_configs,
):
    # Plot the maps with journeys
    # todo: refactor
    counters["journey_year"] = counters["first_year"]

    for journey in journey_files:
        if attempting_all:
            print(
                "\nAttempting to plot journey number {} of {} ({}% done)".format(
                    str(counters["n_journeys_attempted"] + 1),
                    str(no_journeys),
                    str(round(100 * counters["n_journeys_attempted"] / no_journeys, 2)),
                )
            )
            print(
                "{} journeys were plotted successfully, {} journeys were unparsable, {} journeys were not cycling, {} "
                "journeys started outside London".format(
                    str(counters["n_journeys_plotted"]),
                    str(counters["n_files_unparsable"]),
                    str(counters["n_non_cycling_activities"]),
                    str(counters["n_journeys_outside_London"]),
                )
            )
            print(
                timesince(
                    start_time, 100 * counters["n_journeys_attempted"] / no_journeys
                )
            )
        else:
            print(
                "\nAttempting to plot journey number "
                + str(counters["n_journeys_plotted"] + 1)
                + " of "
                + str(no_journeys)
                + " ("
                + str(round(100 * counters["n_journeys_plotted"] / no_journeys, 2))
                + "% done)"
            )
            print(
                str(counters["n_journeys_plotted"])
                + " journeys were plotted successfully, "
                + str(counters["n_files_unparsable"])
                + " journeys were unparsable, "
                + str(counters["n_non_cycling_activities"])
                + " journeys were not cycling, "
                + str(counters["n_journeys_outside_London"])
                + " journeys started outside London"
            )
            print(
                timesince(
                    start_time, 100 * counters["n_journeys_plotted"] / no_journeys
                )
            )

        journey_colour_score = counters["n_journeys_plotted"] / no_journeys

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

                # todo: also consider final point
                if (
                    (new_point_gdf.geometry[0].x > x_lims[0])
                    and (new_point_gdf.geometry[0].x < x_lims[1])
                    and (new_point_gdf.geometry[0].y > y_lims[0])
                    and (new_point_gdf.geometry[0].y < y_lims[1])
                ):

                    new_timestr = counters["first_year"] + " - " + journey.split("-")[0]
                    new_timestr_moving_recents = journey.split("-")[0]
                    new_journey_year = journey.split("-")[0]

                    journey_year_change = False
                    if new_journey_year != counters["journey_year"]:
                        journey_year_change = True
                        if map_configs["by_year"]["plotting_or_not"]:
                            filename = os.path.join(
                                os.path.join(data_path, "results"),
                                runstr
                                + "_cycling_in_year_"
                                + counters["journey_year"]
                                + ".png",
                            )
                            maps_dict["by_year"][0].savefig(
                                filename,
                                bbox_inches="tight",
                                ax=maps_dict["by_year"][1],
                            )  # Output the by_year maps at the end of the year

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

                    for map_scheme_name, map_scheme_configs in map_configs.items():
                        if map_scheme_configs["plotting_or_not"]:

                            # Pack up inputs

                            inputs = {
                                "journey_year_change": journey_year_change,
                                "journey_year": counters["journey_year"],
                                "set_of_points": new_point_gdf,
                                "ax": maps_dict[map_scheme_name][1],
                                "journey_plots_for_moving_recents": journey_plots[
                                    "journey_plots_for_moving_recents"
                                ],
                                "journey_plots_for_by_year": journey_plots[
                                    "journey_plots_for_by_year"
                                ],
                                "n_concurrent": n_concurrent,
                                "colors": colors,
                                "journey_colour_score": journey_colour_score,
                                "journey_plots_shrinking": journey_plots[
                                    "journey_plots_shrinking"
                                ],
                                "journey_plots_bubbling": journey_plots[
                                    "journey_plots_bubbling"
                                ],
                                "end_points_bubbling": journey_plots[
                                    "end_points_bubbling"
                                ],
                                "end_points_shrinking": journey_plots[
                                    "end_points_shrinking"
                                ],
                                "n_concurrent_bubbling": n_concurrent_bubbling,
                                "n_concurrent_bubbling_end_points": n_concurrent_bubbling_end_points,
                            }

                            returns = plotter_functions_dict[map_scheme_name](inputs)

                            # Unpack returns

                            journey_plots["journey_plots_for_moving_recents"] = returns[
                                "journey_plots_for_moving_recents"
                            ]
                            journey_plots["journey_plots_for_by_year"] = returns[
                                "journey_plots_for_by_year"
                            ]
                            journey_plots["journey_plots_shrinking"] = returns[
                                "journey_plots_shrinking"
                            ]
                            journey_plots["journey_plots_shrinking"] = returns[
                                "journey_plots_shrinking"
                            ]
                            journey_plots["journey_plots_bubbling"] = returns[
                                "journey_plots_bubbling"
                            ]
                            journey_plots["end_points_bubbling"] = returns[
                                "end_points_bubbling"
                            ]
                            journey_plots["end_points_shrinking"] = returns[
                                "end_points_shrinking"
                            ]

                            if timestr != new_timestr:
                                text_vars[map_scheme_name].set_visible(False)
                                if map_scheme_name in ["running_recents", "by_year"]:
                                    text_vars[map_scheme_name] = maps_dict[
                                        map_scheme_name
                                    ][0].text(
                                        0.9650,
                                        0.985,
                                        new_timestr_moving_recents,
                                        horizontalalignment="left",
                                        verticalalignment="top",
                                        transform=maps_dict[map_scheme_name][
                                            1
                                        ].transAxes,
                                        color=map_configs[map_scheme_name]["year_text"],
                                    )
                                else:
                                    text_vars[map_scheme_name] = maps_dict[
                                        map_scheme_name
                                    ][0].text(
                                        0.9250,
                                        0.985,
                                        new_timestr,
                                        horizontalalignment="left",
                                        verticalalignment="top",
                                        transform=maps_dict[map_scheme_name][
                                            1
                                        ].transAxes,
                                        color=map_configs[map_scheme_name]["year_text"],
                                    )

                            if making_videos:
                                if (
                                    counters["n_journeys_plotted"] + 1
                                ) % image_interval == 0:
                                    filename = os.path.join(
                                        os.path.join(
                                            os.path.join(data_path, "images_for_video"),
                                            map_scheme_name,
                                        ),
                                        "first_"
                                        + str(counters["n_journeys_plotted"] + 1).zfill(
                                            4
                                        )
                                        + "_journeys.png",
                                    )
                                    maps_dict[map_scheme_name][0].savefig(
                                        filename,
                                        bbox_inches="tight",
                                        fig=maps_dict[map_scheme_name][0],
                                        ax=maps_dict[map_scheme_name][1],
                                    )

                    counters["n_journeys_plotted"] += 1
                    counters["n_journeys_attempted"] += 1

                    timestr = new_timestr
                    timestr_moving_recents = new_timestr_moving_recents
                    counters["journey_year"] = new_journey_year

                    if counters["n_journeys_plotted"] >= no_journeys:
                        break

                else:
                    print("Activity outside of London limits so not plotted")
                    counters["n_journeys_outside_London"] += 1
                    counters["n_journeys_attempted"] += 1

            else:
                counters["n_non_cycling_activities"] += 1
                counters["n_journeys_attempted"] += 1

        except Exception as e:
            print("Problem parsing " + (os.path.join("cycling_data", journey)))
            print(e)
            counters["n_files_unparsable"] += 1
            counters["unparsable_files"].append(journey)
            counters["n_journeys_attempted"] += 1
    return counters


def make_final_by_year_image(runstr, counters, maps_dict, map_configs):
    # Save the final by_year image - done separately to the other image formats below, due to the difference in filename

    if map_configs["by_year"]["plotting_or_not"]:
        filename = os.path.join(
            os.path.join(data_path, "results"),
            runstr + "_cycling_in_year_" + counters["journey_year"] + "_to_date.png",
        )
        maps_dict["by_year"][0].savefig(
            filename, bbox_inches="tight", ax=maps_dict["by_year"][1]
        )  # Output the by_year maps at the end of the year

    # Save the final images
    print("")
    print("Saving the final images...")
    for map_scheme_name, map_scheme_configs in map_configs.items():
        if (
            map_scheme_configs["final_figure_output"]
            and map_scheme_configs["plotting_or_not"]
        ):
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
                filename, bbox_inches="tight", ax=maps_dict[map_scheme_name][1]
            )


def additional_frames_journeys_fading_out(
    journey_files, maps_dict, journey_plots, counters, map_configs
):
    # Add on some more frames to running_recents, where older journey_files disappear one by one
    # todo: refactor function
    if making_videos:
        index = 0
        if map_configs["running_recents"]["plotting_or_not"]:
            timestr = counters["first_year"] + " - " + journey_files[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            number_plots_to_do = len(journey_plots["journey_plots_for_moving_recents"])
            print("")
            while len(journey_plots["journey_plots_for_moving_recents"]) > 0:
                print(
                    "HA! Additional figure generation: making additional image "
                    + str(index + 1)
                    + " of "
                    + str(number_plots_to_do)
                    + " for the running_recents video - fading out the old journey_files"
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
                        fig=maps_dict["running_recents"][0],
                        ax=maps_dict["running_recents"][1],
                    )
                index += 1

    # Add on some more frames to bubbling, where older journey_files disappear one by one
    if making_videos:
        index = 0
        if map_configs["overall_bubbling_off"]["plotting_or_not"]:
            timestr = counters["first_year"] + " - " + journey_files[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            number_plots_to_do = len(journey_plots["journey_plots_bubbling"])
            print("")
            while len(journey_plots["journey_plots_bubbling"]) > 0:
                print(
                    "HA! Additional figure generation: making additional image "
                    + str(index + 1)
                    + " of "
                    + str(number_plots_to_do)
                    + " for the bubbling video - fading out the old journey_files"
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
                        fig=maps_dict["overall_bubbling_off"][0],
                        ax=maps_dict["overall_bubbling_off"][1],
                    )
                index += 1

    # Add on some more frames to journey_plots['end_points_bubbling'], where older journey_files disappear one by one
    if making_videos:
        index = 0
        if map_configs["end_points_bubbling"]["plotting_or_not"]:
            timestr = counters["first_year"] + " - " + journey_files[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            number_plots_to_do = len(journey_plots["end_points_bubbling"])
            print("")
            while len(journey_plots["end_points_bubbling"]) > 0:
                print(
                    "HA! Additional figure generation: making additional image "
                    + str(index + 1)
                    + " of "
                    + str(number_plots_to_do)
                    + " for the end_pounts_bubbling video - fading out the old journey_files"
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
                        fig=maps_dict["end_points_bubbling"][0],
                        ax=maps_dict["end_points_bubbling"][1],
                    )
                index += 1

    # Add on some more frames to shrinking, where older gradually shrink
    if making_videos:
        index = 0
        if map_configs["overall_shrinking"]["plotting_or_not"]:
            timestr = counters["first_year"] + " - " + journey_files[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            print("")
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
                print(
                    "HA! Additional figure generation: making additional image "
                    + str(index + 1)
                    + " of "
                    + str(number_plots_to_do)
                    + " for the shrinking video - shrinking the old journey_files"
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
                        fig=maps_dict["overall_shrinking"][0],
                        ax=maps_dict["overall_shrinking"][1],
                    )
                index += 1

    # Add on some more frames to journey_plots['end_points_shrinking'], where older gradually shrink
    if making_videos:
        index = 0
        if map_configs["end_points_shrinking"]["plotting_or_not"]:
            timestr = counters["first_year"] + " - " + journey_files[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            print("")
            sizes = [points._sizes for points in journey_plots["end_points_shrinking"]]
            sizes = [size[0] for size in sizes]
            max_sizes = max(sizes)

            rat = 0.9
            min_size = 10

            number_plots_to_do = math.ceil(
                (np.log(min_size) - np.log(max_sizes)) / np.log(rat)
            )

            for i in range(number_plots_to_do):
                print(
                    "HA! Additional figure generation: making additional image "
                    + str(index + 1)
                    + " of "
                    + str(number_plots_to_do)
                    + " for the end_points_shrinking video - shrinking the old journey_files"
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
                        fig=maps_dict["end_points_shrinking"][0],
                        ax=maps_dict["end_points_shrinking"][1],
                    )
                index += 1


def make_all_videos(runstr, counters, map_configs):
    # Make the videos
    if making_videos:
        for map_scheme_name, map_scheme_configs in map_configs.items():
            if map_scheme_configs["plotting_or_not"]:
                make_video(map_scheme_name, runstr, counters["n_journeys_plotted"])


def clear_out_images_for_video_folder(making_videos):
    # Clear out the images_for_video folder
    if making_videos:
        if len(os.listdir(os.path.join(data_path, "images_for_video"))) > 0:
            for folder in os.listdir(os.path.join(data_path, "images_for_video")):
                shutil.rmtree(
                    os.path.join(os.path.join(data_path, "images_for_video"), folder)
                )


def overall_run_notes(
    runstr,
    attempting_all,
    no_journeys,
    overall_start_time,
    overall_finish_time,
    counters,
    map_configs,
):

    outputs_str = ""
    outputs_str += "Journeys plotted at " + runstr[0:-1] + "\n\n"
    outputs_str += "Making videos was " + str(making_videos) + "\n\n"
    if attempting_all:
        outputs_str += "Attempted all the images\n\n"
    else:
        outputs_str += (
            "Attempted a subset of " + str(no_journeys) + " of the journeys\n\n"
        )

    outputs_str += "Plots made:\n"

    for map_scheme_name, map_scheme_configs in map_configs.items():
        if map_scheme_configs["plotting_or_not"]:
            outputs_str += map_scheme_name
            outputs_str += "\n"

    outputs_str += (
        "\n" + str(counters["n_journeys_attempted"]) + " journey plots attempted"
    )
    outputs_str += (
        "\n" + str(counters["n_journeys_plotted"]) + " journeys successfully plotted"
    )
    outputs_str += (
        "\n" + str(counters["n_non_cycling_activities"]) + " journeys were not cycling"
    )
    outputs_str += (
        "\n"
        + str(counters["n_journeys_outside_London"])
        + " journeys were outside of London"
    )
    outputs_str += (
        "\n" + str(counters["n_files_unparsable"]) + " journeys would not parse\n\n"
    )

    if counters["n_files_unparsable"] > 0:
        outputs_str += (
            "Journeys that would not parse:\n"
            + "\n".join(counters["unparsable_files"])
            + "\n\n"
        )
    outputs_str += (
        "Summary of time taken:\n"
        + time_diff(overall_start_time, overall_finish_time)
        + " on everything\n"
    )

    print("Info about the run\n")
    print(outputs_str)

    logger.info(outputs_str)
    print("All done!")


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


def configure_logger(runstr):
    """
    Sets up a logger that prints to the console and to a file
    Returns:
        logger object
    """
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    logger = logging.getLogger()
    fileHandler = logging.FileHandler(
        os.path.join(os.path.join(data_path, "results"), "{}run.log".format(runstr))
    )
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    return logger
