"""This script plots the first n or all journeys iteratively, building up frames for a video, producing the video,
and keeping the final images

example debug usage python main.py --debug=True
example normal usage python main.py
"""

import argparse
import datetime
import logging
import os
import sys

from configs.configs_deployment import data_path
from configs.configs_maps import MAP_CONFIGS
from configs.configs_run import which_maps_to_make
from utils.additional_frames import additional_frames_journeys_fading_out
from utils.file_system import (
    read_in_convert_base_maps,
    clear_out_old_folders_and_make_new,
    get_journey_files,
    clear_out_images_for_video_folder,
)
from utils.image_and_video import (
    plot_base_map_layers,
    make_first_frames,
    make_all_other_frames,
    make_final_by_year_image,
    make_all_videos,
    set_up_plot_lists_and_counters,
    reduce_map_configs_to_maps_being_made,
)

os.environ["PROJ_LIB"] = os.path.join(
    os.path.dirname(os.path.dirname(sys.executable)), "share", "proj"
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


def run(map_configs, n_journeys=None):

    journey_files = get_journey_files(n_journeys)

    map_configs = reduce_map_configs_to_maps_being_made(map_configs, which_maps_to_make)

    base_layers = read_in_convert_base_maps(map_configs)

    clear_out_old_folders_and_make_new(map_configs)

    maps_dict = plot_base_map_layers(base_layers, map_configs)

    (
        start_time,
        journey_plots,
        counters,
        text_vars,
    ) = set_up_plot_lists_and_counters(journey_files)

    timestr, timestr_moving_recents, text_vars = make_first_frames(
        counters, journey_files, text_vars, maps_dict, map_configs
    )

    counters = make_all_other_frames(
        journey_files,
        start_time,
        maps_dict,
        text_vars,
        timestr,
        journey_plots,
        counters,
        map_configs,
        runstr,
    )

    make_final_by_year_image(counters, maps_dict, map_configs, runstr)

    additional_frames_journeys_fading_out(
        journey_files, maps_dict, journey_plots, counters, map_configs
    )

    make_all_videos(counters, map_configs, runstr)

    clear_out_images_for_video_folder(map_configs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Journey plotter argparser")
    parser.add_argument(
        "--debug",
        "-d",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="Running a faster version of code for debugging?",
    )
    args = parser.parse_args()

    if args.debug:
        for map in MAP_CONFIGS.keys():
            MAP_CONFIGS[map]["layers"] = ["tidal_water"]
        n_journeys = 5
    else:
        n_journeys = None

    run(map_configs=MAP_CONFIGS, n_journeys=n_journeys)
