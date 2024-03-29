import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

# Set area of interest - currently set to London - is WGS 84/epsg:4326 format
x_lims = (-0.23161763999482177, 0.023482985785112587)
y_lims = (51.449560897899204, 51.566634677359474)

x_lims_broader = tuple(np.mean(x_lims) + ((x_lims - np.mean(x_lims)) * 2))
y_lims_broader = tuple(np.mean(y_lims) + ((y_lims - np.mean(y_lims)) * 2))

# Color scheme for journeys
cmap = matplotlib.cm.get_cmap("autumn")
cmap_rainbow = matplotlib.cm.get_cmap("gist_rainbow")

# Configs for the colours

cNorm = colors.Normalize(vmin=0, vmax=18)  # Speed in mph for completely yellow dots
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

cNorm_time = colors.Normalize(vmin=0, vmax=1)  # Because no_journeys/journeys_to_ plot goes from 0 to 1
scalarMap_time = cmx.ScalarMappable(norm=cNorm_time, cmap=cmap_rainbow)


# Define base layers and their colours, use this info to read them in
base_layers_configs = {
    "roads": ["TQ_Road.shp", "white"],
    "water": ["TQ_SurfaceWater_Area.shp", tuple([x / 255 for x in [221, 237, 255]])],
    "tidal_water": ["TQ_TidalWater.shp", tuple([x / 255 for x in [221, 237, 255]])],
    "building": ["TQ_Building.shp", "lightgray"],
    "parks": ["TQ_GreenspaceSite.shp", tuple([x / 255 for x in [139, 224, 147]])],
}

# Define which layers appear on each map, whether we are plotting that map, and whether its final image should be kept
MAP_CONFIGS = {
    "overall": {
        "layers": [
            "roads",
            "water",
            "tidal_water",
            "building",
            "parks",
        ],
        "which_journeys": "all_journeys",
        "final_figure_output": True,
        "year_text": "black",
        "additional_frames_needed": False
    },
    "running_recents": {
        "layers": [
            "roads",
            "water",
            "tidal_water",
            "building",
            "parks",
        ],
        "which_journeys": "most_recent_journeys",
        "final_figure_output": False,
        "year_text": "black",
        "additional_frames_needed": True
    },
    "dark": {
        "layers": ["tidal_water"],
        "which_journeys": "all_journeys",
        "final_figure_output": True,
        "year_text": "dimgrey",
        "additional_frames_needed": False
    },
    "dark_colours_by_time": {
        "layers": ["tidal_water"],
        "which_journeys": "all_journeys",
        "final_figure_output": True,
        "year_text": "dimgrey",
        "additional_frames_needed": False
    },
    "overall_alpha_1": {
        "layers": [
            "roads",
            "water",
            "tidal_water",
            "building",
            "parks",
        ],
        "which_journeys": "all_journeys",
        "final_figure_output": True,
        "year_text": "black",
        "additional_frames_needed": False
    },
    "by_year": {
        "layers": [
            "roads",
            "water",
            "tidal_water",
            "building",
            "parks",
        ],
        "which_journeys": "all_journeys",
        "final_figure_output": False,
        "year_text": "black",
        "additional_frames_needed": False
    },
    "overall_thick": {
        "layers": [
            "roads",
            "water",
            "tidal_water",
            "building",
            "parks",
        ],
        "which_journeys": "all_journeys",
        "final_figure_output": True,
        "year_text": "black",
        "additional_frames_needed": False
    },
    "overall_shrinking": {
        "layers": [
            "roads",
            "water",
            "tidal_water",
            "building",
            "parks",
        ],
        "which_journeys": "all_journeys",
        "final_figure_output": True,
        "year_text": "black",
        "additional_frames_needed": True
    },
    "overall_bubbling_off": {
        "layers": [
            "roads",
            "water",
            "tidal_water",
            "building",
            "parks",
        ],
        "which_journeys": "all_journeys",
        "final_figure_output": True,
        "year_text": "black",
        "additional_frames_needed": True
    },
    "end_points": {
        "layers": [
            "roads",
            "water",
            "tidal_water",
            "building",
            "parks",
        ],
        "which_journeys": "all_journeys",
        "final_figure_output": True,
        "year_text": "black",
        "additional_frames_needed": False
    },
    "end_points_shrinking": {
        "layers": [
            "roads",
            "water",
            "tidal_water",
            "building",
            "parks",
        ],
        "which_journeys": "all_journeys",
        "final_figure_output": True,
        "year_text": "black",
        "additional_frames_needed": True
    },
    "end_points_bubbling": {
        "layers": [
            "roads",
            "water",
            "tidal_water",
            "building",
            "parks",
        ],
        "which_journeys": "all_journeys",
        "final_figure_output": True,
        "year_text": "black",
        "additional_frames_needed": True
    },
}
