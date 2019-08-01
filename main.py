# This script plots the first n or all journeys interatively, building up frames for a video, producing the video, and keeping the final images

import os
import geopandas as gpd
import gpxpy
import shapely
import datetime
import numpy as np
import time
import math
import shutil

from utils import timesince, time_diff, make_video, which_layers_needed, read_in_required_layers, \
    plotter_functions_dict, convert_crop_base_map_layers, get_speeds, scalarMap, read_in_convert_base_maps, \
    clear_out_old_folders_and_make_new, plot_base_map_layers
from configs import base_layers_configs, map_configs, image_interval, n_concurrent, x_lims, y_lims, making_videos, \
    colored_black_end_points, colored_not_black_end_points, red, n_concurrent_bubbling, data_path, \
    n_concurrent_bubbling_end_points


def run():
    overall_start_time = datetime.datetime.now()

    # Set the runstring (string of the time of running)
    runstr = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_")

    #
    # GET USER INPUT IF ONLY PLOTTING THE FIRST FEW JOURNEYS
    #

    journeys = sorted([x for x in os.listdir(r'C:\dev\data\journey_plotter\cycling_data') if '.gpx' in x])

    no_journeys = input("How many journeys to include? Simply ENTER to attempt all")
    if no_journeys == "":
        no_journeys = len(journeys)
        attempting_all = True
    else:
        no_journeys = int(no_journeys)
        attempting_all = False

    #
    # CONSTRUCT THE BASE layers - read the data in, set the colours and convert to Mercator projections
    #

    base_layers = read_in_convert_base_maps(map_configs)

    clear_out_old_folders_and_make_new(map_configs)

    maps_dict = plot_base_map_layers(map_configs, base_layers)

    #
    # LOOP OVER JOURNEYS, PLOTTING THEM TO THE RELEVANT MAPS
    #

    journey_plots_for_moving_recents = []
    journey_plots_for_by_year = []
    journey_plots_shrinking = []
    journey_plots_bubbling = []
    end_points_shrinking = []
    end_points_bubbling = []

    start_time = datetime.datetime.now()

    n_journeys_attempted = 0
    n_journeys_plotted = 0
    n_journeys_outside_London = 0
    n_non_cycling_activities = 0
    n_files_unparsable = 0
    unparsable_files = []

    text_vars = {}

    # Plot the base maps
    first_year = journeys[0].split("-")[0]
    journey_year = journeys[0].split("-")[0]
    for key, value in map_configs.items():
        if value['plotting_or_not']:
            filename = os.path.join(os.path.join(os.path.join(data_path, 'images_for_video'), key),
                                    'cycling_locations_first_' + str(n_journeys_plotted).zfill(4) + '_journeys.png')

            timestr = first_year + " - " + journeys[0].split("-")[0]
            timestr_moving_recents = journeys[0].split("-")[0]
            if key in ['running_recents', 'by_year']:
                text_vars[key] = maps_dict[key][0].text(0.9650, 0.985, timestr_moving_recents,
                                                        horizontalalignment='left',
                                                        verticalalignment='top',
                                                        transform=maps_dict[key][1].transAxes,
                                                        color=map_configs[key]['year_text'])
            else:
                text_vars[key] = maps_dict[key][0].text(0.9250, 0.985, timestr,
                                                        horizontalalignment='left',
                                                        verticalalignment='top',
                                                        transform=maps_dict[key][1].transAxes,
                                                        color=map_configs[key]['year_text'])

            maps_dict[key][0].savefig(filename, bbox_inches='tight', fig=maps_dict[key][0], ax=maps_dict[key][1])

    # Plot the maps with journeys
    for journey in journeys:
        if attempting_all:
            print(
                "\nAttempting to plot journey number " + str(n_journeys_attempted + 1) + " of " + str(
                    no_journeys) + " (" +
                str(round(100 * n_journeys_attempted / no_journeys, 2)) + "% done)")
            print(str(n_journeys_plotted) + " journeys were plotted successfully, " + str(
                n_files_unparsable) + " journeys were unparsable, " +
                  str(n_non_cycling_activities) + " journeys were not cycling, " + str(
                n_journeys_outside_London) + " journeys started outside London")
            print(timesince(start_time, 100 * n_journeys_attempted / no_journeys))
        else:
            print("\nAttempting to plot journey number " + str(n_journeys_plotted + 1) + " of " + str(
                no_journeys) + " (" +
                  str(round(100 * n_journeys_plotted / no_journeys, 2)) + "% done)")
            print(str(n_journeys_plotted) + " journeys were plotted successfully, " + str(
                n_files_unparsable) + " journeys were unparsable, " +
                  str(n_non_cycling_activities) + " journeys were not cycling, " + str(
                n_journeys_outside_London) + " journeys started outside London")
            print(timesince(start_time, 100 * n_journeys_plotted / no_journeys))

        journey_colour_score = n_journeys_plotted / no_journeys

        try:
            # Read in the data, convert to gdf
            gpx = gpxpy.parse(open(os.path.join(os.path.join(data_path, 'cycling_data'), journey)))
            track = gpx.tracks[0]
            lats = []
            lngs = []
            times = []

            if 'cycling' in gpx.tracks[0].name.lower():

                for segment in track.segments:
                    for point in segment.points:
                        lats.append(point.latitude)
                        lngs.append(point.longitude)
                        times.append(time.mktime(point.time.timetuple()))

                speeds = get_speeds(lats, lngs, times)

                new_points = [shapely.geometry.Point(xy) for xy in zip(lngs, lats)]
                new_point_gdf = gpd.GeoDataFrame(crs={'init': 'epsg:4326'}, geometry=new_points)
                new_point_gdf = new_point_gdf.to_crs(crs={'init': 'epsg:3395'})

                if ((new_point_gdf.geometry[0].x > x_lims[0]) and
                        (new_point_gdf.geometry[0].x < x_lims[1]) and
                        (new_point_gdf.geometry[0].y > y_lims[0]) and
                        (new_point_gdf.geometry[0].y < y_lims[1])):

                    new_timestr = first_year + " - " + journey.split("-")[0]
                    new_timestr_moving_recents = journey.split("-")[0]
                    new_journey_year = journey.split("-")[0]

                    journey_year_change = False
                    if new_journey_year != journey_year:
                        journey_year_change = True
                        if map_configs['by_year']['plotting_or_not']:
                            filename = os.path.join('results', runstr + '_cycling_in_year_' + journey_year + '.png')
                            maps_dict['by_year'][0].savefig(filename, bbox_inches='tight', ax=maps_dict['by_year'][
                                1])  # Output the by_year maps at the end of the year

                    if colored_black_end_points:
                        colors = np.append(np.append([[0, 0, 0, 1]], scalarMap.to_rgba(speeds[1:-1]), axis=0),
                                           [[0, 0, 0, 1]],
                                           axis=0)
                    if colored_not_black_end_points:
                        colors = scalarMap.to_rgba(speeds)
                    if red:
                        colors = "red"

                    for key, value in map_configs.items():
                        if value['plotting_or_not']:

                            # Pack up inputs

                            inputs = {
                                'journey_year_change': journey_year_change,
                                'journey_year': journey_year,
                                'set_of_points': new_point_gdf,
                                'ax': maps_dict[key][1],
                                'journey_plots_for_moving_recents': journey_plots_for_moving_recents,
                                'journey_plots_for_by_year': journey_plots_for_by_year,
                                'n_concurrent': n_concurrent,
                                'colors': colors,
                                'journey_colour_score': journey_colour_score,
                                'journey_plots_shrinking': journey_plots_shrinking,
                                'journey_plots_bubbling': journey_plots_bubbling,
                                'end_points_bubbling': end_points_bubbling,
                                'end_points_shrinking': end_points_shrinking,
                                'n_concurrent_bubbling': n_concurrent_bubbling,
                                'n_concurrent_bubbling_end_points': n_concurrent_bubbling_end_points,
                            }

                            returns = plotter_functions_dict[key](inputs)

                            # Unpack returns

                            journey_plots_for_moving_recents = returns['journey_plots_for_moving_recents']
                            journey_plots_for_by_year = returns['journey_plots_for_by_year']
                            journey_plots_shrinking = returns['journey_plots_shrinking']
                            journey_plots_shrinking = returns['journey_plots_shrinking']
                            journey_plots_bubbling = returns['journey_plots_bubbling']
                            end_points_bubbling = returns['end_points_bubbling']
                            end_points_shrinking = returns['end_points_shrinking']

                            if timestr != new_timestr:
                                text_vars[key].set_visible(False)
                                if key in ['running_recents', 'by_year']:
                                    text_vars[key] = maps_dict[key][0].text(0.9650, 0.985, new_timestr_moving_recents,
                                                                            horizontalalignment='left',
                                                                            verticalalignment='top',
                                                                            transform=maps_dict[key][1].transAxes,
                                                                            color=map_configs[key]['year_text'])
                                else:
                                    text_vars[key] = maps_dict[key][0].text(0.9250, 0.985, new_timestr,
                                                                            horizontalalignment='left',
                                                                            verticalalignment='top',
                                                                            transform=maps_dict[key][1].transAxes,
                                                                            color=map_configs[key]['year_text'])

                            if making_videos:
                                if (n_journeys_plotted + 1) % image_interval == 0:
                                    filename = os.path.join(
                                        os.path.join(os.path.join(data_path, 'images_for_video'), key),
                                        'first_' + str(n_journeys_plotted + 1).zfill(
                                            4) + '_journeys.png')
                                    maps_dict[key][0].savefig(filename, bbox_inches='tight', fig=maps_dict[key][0],
                                                              ax=maps_dict[key][1])

                    n_journeys_plotted += 1
                    n_journeys_attempted += 1

                    timestr = new_timestr
                    timestr_moving_recents = new_timestr_moving_recents
                    journey_year = new_journey_year

                    if n_journeys_plotted >= no_journeys:
                        break

                else:
                    print("Activity outside of London limits so not plotted")
                    n_journeys_outside_London += 1
                    n_journeys_attempted += 1

            else:
                n_non_cycling_activities += 1
                n_journeys_attempted += 1

        except Exception as e:
            print("Problem parsing " + (os.path.join('cycling_data', journey)))
            print(e)
            n_files_unparsable += 1
            unparsable_files.append(journey)
            n_journeys_attempted += 1

    # Save the final by_year image - done separately to the other image formats below, due to the difference in filename
    if map_configs['by_year']['plotting_or_not']:
        filename = os.path.join(os.path.join(data_path, 'results'), runstr + '_cycling_in_year_' + journey_year + '_to_date.png')
        maps_dict['by_year'][0].savefig(filename, bbox_inches='tight', ax=maps_dict['by_year'][
            1])  # Output the by_year maps at the end of the year

    # Save the final images
    print("")
    print("Saving the final images...")
    for key, value in map_configs.items():
        if value['final_figure_output'] and value['plotting_or_not']:
            filename = os.path.join(os.path.join(data_path, 'results'),
                                    runstr + '_' + key + '__first_' + str(n_journeys_plotted) + '_journeys.png')
            maps_dict[key][0].savefig(filename, bbox_inches='tight', ax=maps_dict[key][1])

    # Add on some more frames to running_recents, where older journeys disappear one by one
    if making_videos:
        index = 0
        if map_configs['running_recents']['plotting_or_not']:
            timestr = first_year + " - " + journeys[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            number_plots_to_do = len(journey_plots_for_moving_recents)
            print("")
            while len(journey_plots_for_moving_recents) > 0:
                print('HA! Additional figure generation: making additional image ' + str(index + 1) + ' of ' + str(
                    number_plots_to_do) + ' for the running_recents video - fading out the old journeys')
                # Remove line from plot and journey_plots_for_moving_recents list
                journey_plots_for_moving_recents[0].remove()
                del journey_plots_for_moving_recents[0]
                # Save the images to files
                if index % image_interval == 0:
                    filename = os.path.join(
                        os.path.join(os.path.join(data_path, 'images_for_video'), 'running_recents'),
                        'first_' + str(n_journeys_plotted + index + 1).zfill(4) + '_journeys.png')
                    maps_dict['running_recents'][0].savefig(filename, bbox_inches='tight',
                                                            fig=maps_dict['running_recents'][0],
                                                            ax=maps_dict['running_recents'][1])
                index += 1

    # Add on some more frames to bubbling, where older journeys disappear one by one
    if making_videos:
        index = 0
        if map_configs['overall_bubbling_off']['plotting_or_not']:
            timestr = first_year + " - " + journeys[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            number_plots_to_do = len(journey_plots_bubbling)
            print("")
            while len(journey_plots_bubbling) > 0:
                print('HA! Additional figure generation: making additional image ' + str(index + 1) + ' of ' + str(
                    number_plots_to_do) + ' for the bubbling video - fading out the old journeys')
                # Remove line from plot and journey_plots_for_moving_recents list
                journey_plots_bubbling[0].remove()
                del journey_plots_bubbling[0]
                for i in range(len(journey_plots_bubbling)):
                    journey_plots_bubbling[i].set_sizes(journey_plots_bubbling[i]._sizes * 1.2)
                    if journey_plots_bubbling[i]._alpha > 0.0005:
                        journey_plots_bubbling[i].set_alpha(journey_plots_bubbling[i]._alpha * 0.82)
                # Save the images to files
                if index % image_interval == 0:
                    filename = os.path.join(
                        os.path.join(os.path.join(data_path, 'images_for_video'), 'overall_bubbling_off'),
                        'first_' + str(n_journeys_plotted + index + 1).zfill(4) + '_journeys.png')
                    maps_dict['overall_bubbling_off'][0].savefig(filename, bbox_inches='tight',
                                                                 fig=maps_dict['overall_bubbling_off'][0],
                                                                 ax=maps_dict['overall_bubbling_off'][1])
                index += 1

    # Add on some more frames to end_points_bubbling, where older journeys disappear one by one
    if making_videos:
        index = 0
        if map_configs['end_points_bubbling']['plotting_or_not']:
            timestr = first_year + " - " + journeys[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            number_plots_to_do = len(end_points_bubbling)
            print("")
            while len(end_points_bubbling) > 0:
                print('HA! Additional figure generation: making additional image ' + str(index + 1) + ' of ' + str(
                    number_plots_to_do) + ' for the end_pounts_bubbling video - fading out the old journeys')
                # Remove line from plot and journey_plots_for_moving_recents list
                end_points_bubbling[0].remove()
                del end_points_bubbling[0]
                for i in range(len(end_points_bubbling)):
                    end_points_bubbling[i].set_sizes(end_points_bubbling[i]._sizes * 1.1)
                    if end_points_bubbling[i]._alpha > 0.0005:
                        end_points_bubbling[i].set_alpha(end_points_bubbling[i]._alpha * 0.95)
                # Save the images to files
                if index % image_interval == 0:
                    filename = os.path.join(
                        os.path.join(os.path.join(data_path, 'images_for_video'), 'end_points_bubbling'),
                        'first_' + str(n_journeys_plotted + index + 1).zfill(4) + '_journeys.png')
                    maps_dict['end_points_bubbling'][0].savefig(filename, bbox_inches='tight',
                                                                fig=maps_dict['end_points_bubbling'][0],
                                                                ax=maps_dict['end_points_bubbling'][1])
                index += 1

    # Add on some more frames to shrinking, where older gradually shrink
    if making_videos:
        index = 0
        if map_configs['overall_shrinking']['plotting_or_not']:
            timestr = first_year + " - " + journeys[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            print("")
            sizes = [points._sizes for points in journey_plots_shrinking]
            sizes = [size[0] for size in sizes]
            max_sizes = max(sizes)

            rat = 0.95
            min_size = 1

            number_plots_to_do = math.ceil((np.log(min_size) - np.log(max_sizes)) / np.log(rat))

            for i in range(number_plots_to_do):
                print('HA! Additional figure generation: making additional image ' + str(index + 1) + ' of ' + str(
                    number_plots_to_do) + ' for the shrinking video - shrinking the old journeys')

                for points in journey_plots_shrinking:
                    if points._sizes > min_size:
                        points.set_sizes(points._sizes * 0.95)
                        points.set_alpha(points._alpha * 0.99)

                # Save the images to files
                if index % image_interval == 0:
                    filename = os.path.join(
                        os.path.join(os.path.join(data_path, 'images_for_video'), 'overall_shrinking'),
                        'first_' + str(n_journeys_plotted + index + 1).zfill(4) + '_journeys.png')
                    maps_dict['overall_shrinking'][0].savefig(filename, bbox_inches='tight',
                                                              fig=maps_dict['overall_shrinking'][0],
                                                              ax=maps_dict['overall_shrinking'][1])
                index += 1

    # Add on some more frames to end_points_shrinking, where older gradually shrink
    if making_videos:
        index = 0
        if map_configs['end_points_shrinking']['plotting_or_not']:
            timestr = first_year + " - " + journeys[-1].split("-")[0]
            # journey_plots_for_moving_recents defined above, and written to in the plotter_moving_recents function
            print("")
            sizes = [points._sizes for points in end_points_shrinking]
            sizes = [size[0] for size in sizes]
            max_sizes = max(sizes)

            rat = 0.9
            min_size = 10

            number_plots_to_do = math.ceil((np.log(min_size) - np.log(max_sizes)) / np.log(rat))

            for i in range(number_plots_to_do):
                print('HA! Additional figure generation: making additional image ' + str(index + 1) + ' of ' + str(
                    number_plots_to_do) + ' for the end_points_shrinking video - shrinking the old journeys')

                for points in end_points_shrinking:
                    if points._sizes > min_size:
                        points.set_sizes(points._sizes * rat)
                        points.set_alpha(points._alpha * 0.985)

                # Save the images to files
                if index % image_interval == 0:
                    filename = os.path.join(
                        os.path.join(os.path.join(data_path, 'images_for_video'), 'end_points_shrinking'),
                        'first_' + str(n_journeys_plotted + index + 1).zfill(4) + '_journeys.png')
                    maps_dict['end_points_shrinking'][0].savefig(filename, bbox_inches='tight',
                                                                 fig=maps_dict['end_points_shrinking'][0],
                                                                 ax=maps_dict['end_points_shrinking'][1])
                index += 1

    # Make the videos
    if making_videos:
        for key, value in map_configs.items():
            if value['plotting_or_not']:
                make_video(key, runstr, n_journeys_plotted)

    # Clear out the images_for_video folder
    if making_videos:
        if len(os.listdir(os.path.join(data_path, 'images_for_video'))) > 0:
            for folder in os.listdir(os.path.join(data_path, 'images_for_video')):
                shutil.rmtree(os.path.join(os.path.join(data_path, 'images_for_video'), folder))

    overall_finish_time = datetime.datetime.now()

    #
    # OUTPUT THE RUN NOTES
    #

    outputs_str = ""
    outputs_str += "Journeys plotted at " + runstr[0:-1] + "\n\n"
    outputs_str += "Making videos was " + str(making_videos) + "\n\n"
    if attempting_all:
        outputs_str += "Attempted all the images\n\n"
    else:
        outputs_str += "Attempted a subset of " + str(no_journeys) + " of the journeys\n\n"

    outputs_str += "Plots made:\n"

    for key, value in map_configs.items():
        if value['plotting_or_not']:
            outputs_str += key
            outputs_str += "\n"

    outputs_str += ("\n" + str(n_journeys_attempted) + " journey plots attempted")
    outputs_str += ("\n" + str(n_journeys_plotted) + " journeys successfully plotted")
    outputs_str += ("\n" + str(n_non_cycling_activities) + " journeys were not cycling")
    outputs_str += ("\n" + str(n_journeys_outside_London) + " journeys were outside of London")
    outputs_str += ("\n" + str(n_files_unparsable) + " journeys would not parse\n\n")

    if n_files_unparsable > 0:
        outputs_str += ("Journeys that would not parse:\n" +
                        "\n".join(unparsable_files) +
                        "\n\n")
    outputs_str += ("Summary of time taken:\n" +
                    time_diff(overall_start_time, overall_finish_time) + ' on everything\n')

    print("Info about the run\n")
    print(outputs_str)

    file = open(os.path.join(os.path.join(data_path, 'results'), runstr + "run_notes.txt"), "w")
    file.write(outputs_str)
    file.close()

    print("All done!")


if __name__ == "__main__":
    run()
