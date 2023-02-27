import datetime
import logging
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from configs.configs_deployment import data_path
from configs.configs_maps import (
    x_lims,
    y_lims,
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
from utils.geo import (
    get_track,
    handle_problem_parsing_gpx_file,
    initial_checks_on_activity_type_and_location,
    increment_counters_for_unplotted_journeys,
)
from utils.plotters import plotter_functions_dict

logger = logging.getLogger(__name__)


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


def which_layers_needed(map_configs):
    all_layers = []
    for key, value in map_configs.items():
        for layer in value["layers"]:
            all_layers.append(layer)
    return list(set(all_layers))


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


def make_all_other_frames(
    journey_files,
    start_time,
    maps_dict,
    text_vars,
    timestr,
    journey_plots,
    counters,
    map_configs, runstr
):
    counters["journey_year"] = counters["first_year"]

    plotter_inputs = {
        "n_concurrent": n_concurrent,
        "n_concurrent_bubbling": n_concurrent_bubbling,
        "n_concurrent_bubbling_end_points": n_concurrent_bubbling_end_points,
    }
    plotter_inputs.update(journey_plots)

    n_journey_files = len(journey_files)

    for i, journey in tqdm(enumerate(journey_files), total=len(journey_files)):
        # for the dark_colours_by_time plot
        journey_colour_score = i / n_journey_files

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

    return counters


def make_final_by_year_image(counters, maps_dict, map_configs, runstr):

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


def make_all_videos(counters, map_configs, runstr):
    # Make the videos
    if making_videos:
        for map_scheme_name, map_scheme_configs in map_configs.items():
            make_video(map_scheme_name, runstr, counters["n_journeys_plotted"])


def set_up_plot_lists_and_counters(journeys):

    start_time = datetime.datetime.now()

    journey_plots = {
        "running_recents": [],
        "by_year": [],
        "overall_shrinking": [],
        "overall_bubbling_off": [],
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
