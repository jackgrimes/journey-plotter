# This script plots the first n or all journeys iteratively, building up frames for a video, producing the video,
# and keeping the final images

# example debug usage python main.py --debug=True
# example normal usage python main.py


import os
import sys

from configs.configs_maps import MAP_CONFIGS
from configs.configs_run import which_maps_to_make
from utils import (
    read_in_convert_base_maps,
    clear_out_old_folders_and_make_new,
    plot_base_map_layers,
    parse_the_args,
    get_journey_files,
    make_first_frames,
    make_all_other_frames,
    make_final_by_year_image,
    additional_frames_journeys_fading_out,
    make_all_videos,
    clear_out_images_for_video_folder,
    set_up_plot_lists_and_counters,
    reduce_map_configs_to_maps_being_made,
)

os.environ["PROJ_LIB"] = os.path.join(
    os.path.dirname(os.path.dirname(sys.executable)), "share", "proj"
)


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
    )

    make_final_by_year_image(counters, maps_dict, map_configs)

    additional_frames_journeys_fading_out(
        journey_files, maps_dict, journey_plots, counters, map_configs
    )

    make_all_videos(counters, map_configs)

    clear_out_images_for_video_folder(map_configs)


if __name__ == "__main__":
    args = parse_the_args()

    if args.debug:
        for map in MAP_CONFIGS.keys():
            MAP_CONFIGS[map]["layers"] = ["tidal_water"]
        n_journeys = 5
    else:
        n_journeys = None

    run(map_configs=MAP_CONFIGS, n_journeys=n_journeys)
