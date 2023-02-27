import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

# Set which maps you want to create
which_maps_to_make = {
    "overall": True,
    "running_recents": False,
    "dark": False,
    "overall_alpha_1": False,
    "by_year": False,
    "overall_thick": False,
    "dark_colours_by_time": False,
    "overall_shrinking": False,
    "overall_bubbling_off": False,
    "end_points": False,
    "end_points_shrinking": False,
    "end_points_bubbling": False,
}

# Colour scheme for journeys?

colored_black_end_points = True
colored_not_black_end_points = False
red = False

# Making the videos or only the final images?
making_videos = True


# Choose intervals to save images (after how many journeys?), and the number of trips to show on a given map, for the moving_recents_fig

image_interval = 1
n_concurrent = 100
n_concurrent_bubbling = 40
n_concurrent_bubbling_end_points = 100
