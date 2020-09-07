import os
import geopandas as gpd
import cv2
import datetime
import numpy as np
import pandas as pd
import shapely
import matplotlib.pyplot as plt
import shutil

from configs import (
    base_layers_configs,
    map_configs,
    x_lims_broader,
    y_lims_broader,
    data_path,
    scalarMap,
    scalarMap_time,
    base_layers_configs,
    x_lims,
    y_lims,
)


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

    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    #cv2.imshow("video", frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:

        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)

        out.write(frame)  # Write out frame to video

        #cv2.imshow("video", frame)
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
        df = pd.concat([df.shift(2), df.shift(-2)], axis=1)
        df.columns = ["time1", "x1", "y1", "time2", "x2", "y2"]

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


def plot_base_map_layers(map_configs, base_layers):

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
            if key in ["dark", "dark_colours_by_time"]:
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
