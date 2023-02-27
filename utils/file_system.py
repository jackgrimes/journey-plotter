import datetime
import datetime
import logging
import os
import shutil

import geopandas as gpd

from configs.configs_deployment import data_path
from configs.configs_maps import (
    base_layers_configs,
)
from utils.image_and_video import which_layers_needed
from utils.geo import convert_crop_base_map_layers

logger = logging.getLogger(__name__)


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


def get_journey_files(n_journeys=None):

    logger.info("Getting the journey files")

    journey_files = sorted(
        [x for x in os.listdir(os.path.join(data_path, "cycling_data")) if ".gpx" in x]
    )

    if n_journeys is not None:
        journey_files = journey_files[:n_journeys]

    return journey_files


def clear_out_images_for_video_folder(making_videos):
    # Clear out the images_for_video folder
    if making_videos:
        if len(os.listdir(os.path.join(data_path, "images_for_video"))) > 0:
            for folder in os.listdir(os.path.join(data_path, "images_for_video")):
                shutil.rmtree(
                    os.path.join(os.path.join(data_path, "images_for_video"), folder)
                )
