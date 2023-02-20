# This script plots the first n or all journeys interatively, building up frames for a video, producing the video,
# and keeping the final images

# example debug usage python main.py --debug=True --no_journeys=20
# example normal usage python main.py


import datetime
from configs.configs_run import which_maps_to_make
from configs.configs_maps import MAP_CONFIGS
from utils import (
    read_in_convert_base_maps,
    clear_out_old_folders_and_make_new,
    plot_base_map_layers,
    parse_the_args,
    get_start_time,
    get_journey_files,
    make_first_frames,
    make_all_other_frames,
    make_final_by_year_image,
    additional_frames_journeys_fading_out,
    make_all_videos,
    clear_out_images_for_video_folder,
    overall_run_notes,
    set_up_plot_lists_and_counters, reduce_map_configs_to_maps_being_made
)
import os
import sys


os.environ["PROJ_LIB"] = os.path.join( os.path.dirname(os.path.dirname(sys.executable)), "share", "proj")


def run(no_journeys, map_configs):

    overall_start_time, runstr = get_start_time()

    journey_files, no_journeys, attempting_all = get_journey_files(no_journeys)

    map_configs = reduce_map_configs_to_maps_being_made(map_configs, which_maps_to_make)

    base_layers = read_in_convert_base_maps(map_configs)

    clear_out_old_folders_and_make_new(map_configs)

    maps_dict = plot_base_map_layers(base_layers, map_configs)

    (start_time, journey_plots, counters, text_vars,) = set_up_plot_lists_and_counters(
        journey_files
    )

    timestr, timestr_moving_recents, text_vars = make_first_frames(
        counters, journey_files, text_vars, maps_dict, map_configs
    )

    counters = make_all_other_frames(
        journey_files,
        attempting_all,
        no_journeys,
        start_time,
        maps_dict,
        runstr,
        text_vars,
        timestr,
        journey_plots,
        counters,
        map_configs,
    )

    make_final_by_year_image(runstr, counters, maps_dict, map_configs)

    additional_frames_journeys_fading_out(
        journey_files, maps_dict, journey_plots, counters, map_configs
    )

    make_all_videos(runstr, counters, map_configs)

    clear_out_images_for_video_folder(map_configs)

    overall_finish_time = datetime.datetime.now()

    overall_run_notes(
        runstr,
        attempting_all,
        no_journeys,
        overall_start_time,
        overall_finish_time,
        counters,
        map_configs,
    )


if __name__ == "__main__":
    # tqdm.pandas()

    args = parse_the_args()



    map_configs = MAP_CONFIGS

    if args.is_debug:
        for k, v in map_configs.items():
            if k != "dark":
                map_configs[k]["plotting_or_not"] = False

    run(args.no_journeys, map_configs)
