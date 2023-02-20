import os
import sys

import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

# needed to add this line when using conda environments
os.environ["PROJ_LIB"] = os.path.join( os.path.dirname(os.path.dirname(sys.executable)), "share", "proj")

# Set which maps you want to create
overall = False
running_recents = False
dark = False
overall_alpha_1 = True
by_year = False
overall_thick = False
dark_colours_by_time = False
overall_shrinking = True
overall_bubbling_off = False
end_points = False
end_points_shrinking = False
end_points_bubbling = False

# Paths
data_path = "/home/jack/dev/data/journey-plotter"

# Colour scheme for journeys?

colored_black_end_points = True
colored_not_black_end_points = False
red = False

# Making the videos or only the final images?
making_videos = True

# Set area of interest - currently set to London
x_lims = (-25400, 3400)
y_lims = (6667992, 6689192)

x_lims_broader = tuple(np.mean(x_lims) + ((x_lims - np.mean(x_lims)) * 2))
y_lims_broader = tuple(np.mean(y_lims) + ((y_lims - np.mean(y_lims)) * 2))

# Color scheme for journeys
cmap = matplotlib.cm.get_cmap("autumn")
cmap_rainbow = matplotlib.cm.get_cmap("gist_rainbow")

# Configs for the colours

cNorm = colors.Normalize(vmin=0, vmax=18)  # Speed in mph for completely yellow dots
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

cNorm_time = colors.Normalize(
    vmin=0, vmax=1
)  # Because no_journeys/journeys_to_ plot goes from 0 to 1
scalarMap_time = cmx.ScalarMappable(norm=cNorm_time, cmap=cmap_rainbow)

# Choose intervals to save images (after how many journeys?), and the number of trips to show on a given map, for the moving_recents_fig

image_interval = 1
n_concurrent = 100
n_concurrent_bubbling = 40
n_concurrent_bubbling_end_points = 100

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
        "layers": ["roads", "water", "tidal_water", "building", "parks",],
        "which_journeys": "all_journeys",
        "plotting_or_not": overall,
        "final_figure_output": True,
        "year_text": "black",
    },
    "running_recents": {
        "layers": ["roads", "water", "tidal_water", "building", "parks",],
        "which_journeys": "most_recent_journeys",
        "plotting_or_not": running_recents,
        "final_figure_output": False,
        "year_text": "black",
    },
    "dark": {
        "layers": ["tidal_water"],
        "which_journeys": "all_journeys",
        "plotting_or_not": dark,
        "final_figure_output": True,
        "year_text": "dimgrey",
    },
    "dark_colours_by_time": {
        "layers": ["tidal_water"],
        "which_journeys": "all_journeys",
        "plotting_or_not": dark_colours_by_time,
        "final_figure_output": True,
        "year_text": "dimgrey",
    },
    "overall_alpha_1": {
        "layers": ["roads", "water", "tidal_water", "building", "parks",],
        "which_journeys": "all_journeys",
        "plotting_or_not": overall_alpha_1,
        "final_figure_output": True,
        "year_text": "black",
    },
    "by_year": {
        "layers": ["roads", "water", "tidal_water", "building", "parks",],
        "which_journeys": "all_journeys",
        "plotting_or_not": by_year,
        "final_figure_output": False,
        "year_text": "black",
    },
    "overall_thick": {
        "layers": ["roads", "water", "tidal_water", "building", "parks",],
        "which_journeys": "all_journeys",
        "plotting_or_not": overall_thick,
        "final_figure_output": True,
        "year_text": "black",
    },
    "overall_shrinking": {
        "layers": ["roads", "water", "tidal_water", "building", "parks",],
        "which_journeys": "all_journeys",
        "plotting_or_not": overall_shrinking,
        "final_figure_output": True,
        "year_text": "black",
    },
    "overall_bubbling_off": {
        "layers": ["roads", "water", "tidal_water", "building", "parks",],
        "which_journeys": "all_journeys",
        "plotting_or_not": overall_bubbling_off,
        "final_figure_output": True,
        "year_text": "black",
    },
    "end_points": {
        "layers": ["roads", "water", "tidal_water", "building", "parks",],
        "which_journeys": "all_journeys",
        "plotting_or_not": end_points,
        "final_figure_output": True,
        "year_text": "black",
    },
    "end_points_shrinking": {
        "layers": ["roads", "water", "tidal_water", "building", "parks",],
        "which_journeys": "all_journeys",
        "plotting_or_not": end_points_shrinking,
        "final_figure_output": True,
        "year_text": "black",
    },
    "end_points_bubbling": {
        "layers": ["roads", "water", "tidal_water", "building", "parks",],
        "which_journeys": "all_journeys",
        "plotting_or_not": end_points_bubbling,
        "final_figure_output": True,
        "year_text": "black",
    },
}

MAP_CONFIGS_NO_BASE_LAYERS = {
    "overall": {
        "layers": ["tidal_water"],
        "which_journeys": "all_journeys",
        "plotting_or_not": overall,
        "final_figure_output": True,
        "year_text": "black",
    },
    "running_recents": {
        "layers": ["tidal_water"],
        "which_journeys": "most_recent_journeys",
        "plotting_or_not": running_recents,
        "final_figure_output": False,
        "year_text": "black",
    },
    "dark": {
        "layers": ["tidal_water"],
        "which_journeys": "all_journeys",
        "plotting_or_not": dark,
        "final_figure_output": True,
        "year_text": "dimgrey",
    },
    "dark_colours_by_time": {
        "layers": [],
        "which_journeys": "all_journeys",
        "plotting_or_not": dark_colours_by_time,
        "final_figure_output": True,
        "year_text": "dimgrey",
    },
    "overall_alpha_1": {
        "layers": [],
        "which_journeys": "all_journeys",
        "plotting_or_not": overall_alpha_1,
        "final_figure_output": True,
        "year_text": "black",
    },
    "by_year": {
        "layers": [],
        "which_journeys": "all_journeys",
        "plotting_or_not": by_year,
        "final_figure_output": False,
        "year_text": "black",
    },
    "overall_thick": {
        "layers": [],
        "which_journeys": "all_journeys",
        "plotting_or_not": overall_thick,
        "final_figure_output": True,
        "year_text": "black",
    },
    "overall_shrinking": {
        "layers": [],
        "which_journeys": "all_journeys",
        "plotting_or_not": overall_shrinking,
        "final_figure_output": True,
        "year_text": "black",
    },
    "overall_bubbling_off": {
        "layers": [],
        "which_journeys": "all_journeys",
        "plotting_or_not": overall_bubbling_off,
        "final_figure_output": True,
        "year_text": "black",
    },
    "end_points": {
        "layers": [],
        "which_journeys": "all_journeys",
        "plotting_or_not": end_points,
        "final_figure_output": True,
        "year_text": "black",
    },
    "end_points_shrinking": {
        "layers": [],
        "which_journeys": "all_journeys",
        "plotting_or_not": end_points_shrinking,
        "final_figure_output": True,
        "year_text": "black",
    },
    "end_points_bubbling": {
        "layers": [],
        "which_journeys": "all_journeys",
        "plotting_or_not": end_points_bubbling,
        "final_figure_output": True,
        "year_text": "black",
    },
}
