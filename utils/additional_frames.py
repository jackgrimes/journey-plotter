import logging
import os

from configs.configs_deployment import data_path
from configs.configs_run import making_videos, image_interval

logger = logging.getLogger(__name__)

def running_recents_additional_frames(journey_plots, map_configs):
    journey_plots["running_recents"][0].remove()
    del journey_plots["running_recents"][0]

    if len(journey_plots["running_recents"]) == 0:
        map_configs["running_recents"]["additional_frames_needed"] = False
    return journey_plots, map_configs


def overall_bubbling_off_additional_frames(journey_plots, map_configs):
    journey_plots["overall_bubbling_off"][0].remove()
    del journey_plots["overall_bubbling_off"][0]
    for i in range(len(journey_plots["overall_bubbling_off"])):
        journey_plots["overall_bubbling_off"][i].set_sizes(
            journey_plots["overall_bubbling_off"][i]._sizes * 1.2
        )
        if journey_plots["overall_bubbling_off"][i]._alpha > 0.0005:
            journey_plots["overall_bubbling_off"][i].set_alpha(
                journey_plots["overall_bubbling_off"][i]._alpha * 0.82
            )
    if len(journey_plots["overall_bubbling_off"]) == 0:
        map_configs["overall_bubbling_off"]["additional_frames_needed"] = False
    return journey_plots, map_configs


def end_points_bubbling_additional_frames(journey_plots, map_configs):
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
    if len(journey_plots["end_points_bubbling"]) == 0:
        map_configs["end_points_bubbling"]["additional_frames_needed"] = False
    return journey_plots, map_configs


def overall_shrinking_additional_frames(journey_plots, map_configs):
    sizes = [points._sizes for points in journey_plots["overall_shrinking"]]
    sizes = [size[0] for size in sizes]
    max_sizes = max(sizes)

    rat = 0.95
    min_size = 1

    for points in journey_plots["overall_shrinking"]:
        if points._sizes > min_size:
            points.set_sizes(points._sizes * 0.95)
            points.set_alpha(points._alpha * 0.99)

    if all(
        [points._sizes <= min_size for points in journey_plots["overall_shrinking"]]
    ):
        map_configs["overall_shrinking"]["additional_frames_needed"] = False
    return journey_plots, map_configs


def end_points_shrinking_additional_frames(journey_plots, map_configs):
    sizes = [points._sizes for points in journey_plots["end_points_shrinking"]]
    sizes = [size[0] for size in sizes]
    max_sizes = max(sizes)

    rat = 0.9
    min_size = 10

    for points in journey_plots["end_points_shrinking"]:
        if points._sizes > min_size:
            points.set_sizes(points._sizes * rat)
            points.set_alpha(points._alpha * 0.985)

    if all(
        [points._sizes <= min_size for points in journey_plots["end_points_shrinking"]]
    ):
        map_configs["end_points_shrinking"]["additional_frames_needed"] = False
    return journey_plots, map_configs


def additional_frames_journeys_fading_out(
    journey_files, maps_dict, journey_plots, counters, map_configs
):
    if making_videos:
        # Index for saving result e.g. after every 10 journeys, etc
        index = 1

        # Final year should always remain the year of the last journey file - no longer actually plotting new data
        timestr = counters["first_year"] + " - " + journey_files[-1].split("-")[0]

        # Until we've marked all of these additional frames as finished
        while (
            sum(
                [
                    map_configs[key]["additional_frames_needed"]
                    for key in map_configs.keys()
                ]
            )
            > 0
        ):
            for map_scheme_name in map_configs.keys():

                if map_configs[map_scheme_name]["additional_frames_needed"]:

                    logger.info(
                        f"HA! Additional figure generation: to tidy up effects after all journeys are plotted, "
                        f"frame number {index} for {map_scheme_name}"
                    )

                    if (
                        map_scheme_name == "running_recents"
                        and map_configs["running_recents"]["additional_frames_needed"]
                    ):
                        journey_plots, map_configs = running_recents_additional_frames(
                            journey_plots, map_configs
                        )

                    if (
                        map_scheme_name == "overall_bubbling_off"
                        and map_configs["overall_bubbling_off"][
                            "additional_frames_needed"
                        ]
                    ):
                        (
                            journey_plots,
                            map_configs,
                        ) = overall_bubbling_off_additional_frames(
                            journey_plots, map_configs
                        )

                    if (
                        map_scheme_name == "end_points_bubbling"
                        and map_configs["end_points_bubbling"][
                            "additional_frames_needed"
                        ]
                    ):
                        (
                            journey_plots,
                            map_configs,
                        ) = end_points_bubbling_additional_frames(
                            journey_plots, map_configs
                        )

                    # Add on some more frames to shrinking, where older gradually shrink
                    if (
                        map_scheme_name == "overall_shrinking"
                        and map_configs["overall_shrinking"]["additional_frames_needed"]
                    ):
                        (
                            journey_plots,
                            map_configs,
                        ) = overall_shrinking_additional_frames(
                            journey_plots, map_configs
                        )

                    if (
                        map_scheme_name == "end_points_shrinking"
                        and map_configs["end_points_shrinking"][
                            "additional_frames_needed"
                        ]
                    ):
                        (
                            journey_plots,
                            map_configs,
                        ) = end_points_shrinking_additional_frames(
                            journey_plots, map_configs
                        )

                    # Save the images to files
                    if index % image_interval == 0:
                        filename = os.path.join(
                            os.path.join(
                                os.path.join(data_path, "images_for_video"),
                                map_scheme_name,
                            ),
                            "first_"
                            + str(counters["n_journeys_plotted"] + index + 1).zfill(4)
                            + "_journeys.png",
                        )
                        maps_dict[map_scheme_name][0].savefig(
                            filename,
                            bbox_inches="tight",
                        )
            index += 1
