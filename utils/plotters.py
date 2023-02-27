import datetime
import datetime
import logging
import os

import shapely

from configs.configs_deployment import data_path
from configs.configs_maps import (
    scalarMap_time,
)

logger = logging.getLogger(__name__)


def plotter_total(inputs):
    """For plotting the points on the total_fig map"""

    inputs["set_of_points"].plot(
        categorical=False,
        legend=False,
        ax=inputs["ax"],
        color=inputs["colors"],
        alpha=0.1,
        markersize=0.2,
    )
    return inputs


def plotter_moving_recents(inputs):
    """For plotting the points on the moving_recents_fig map"""

    lats = []
    lngs = []

    for index, row in inputs["set_of_points"].iterrows():
        lngs.append(row.geometry.x)
        lats.append(row.geometry.y)

    inputs["running_recents"].append(
        inputs["ax"].scatter(
            lngs, lats, color=inputs["colors"], s=3, marker="s", alpha=0.05
        )
    )

    if len(inputs["running_recents"]) > inputs["n_concurrent"]:
        n_to_remove = len(inputs["running_recents"]) - inputs["n_concurrent"]
        for i in range(n_to_remove):
            inputs["running_recents"][i].remove()
            del inputs["running_recents"][i]

    return inputs


def plotter_by_year(inputs):
    """For plotting the points on the by_year map"""

    lats = []
    lngs = []

    for index, row in inputs["set_of_points"].iterrows():
        lngs.append(row.geometry.x)
        lats.append(row.geometry.y)

    if inputs["journey_year_change"]:
        for i in range(len(inputs["by_year"])):
            inputs["by_year"][i].remove()
        inputs["by_year"] = []

    inputs["by_year"].append(
        inputs["ax"].scatter(
            lngs, lats, color=inputs["colors"], s=2, marker="s", alpha=1
        )
    )

    return inputs


def plotter_all_journeys_thick(inputs):
    """For plotting the points on the by_year map"""

    lats = []
    lngs = []

    for index, row in inputs["set_of_points"].iterrows():
        lngs.append(row.geometry.x)
        lats.append(row.geometry.y)

    inputs["ax"].scatter(lngs, lats, color=inputs["colors"], s=2, marker="s", alpha=1)

    return inputs


def plotter_no_roads(inputs):
    """For plotting the points on the total_fig_no_roads map"""
    inputs["set_of_points"].plot(
        categorical=False,
        legend=False,
        ax=inputs["ax"],
        color="white",
        alpha=0.1,
        markersize=0.1,
    )
    return inputs


def plotter_dark_colours_by_time(inputs):
    """For plotting the points on the dark_colours_by_time map"""

    colors = scalarMap_time.to_rgba(inputs["journey_colour_score"] * 0.85)

    inputs["set_of_points"].plot(
        categorical=False,
        legend=False,
        ax=inputs["ax"],
        color=scalarMap_time.to_rgba(inputs["journey_colour_score"] * 0.85),
        alpha=0.06,
        markersize=0.3,
    )

    return inputs


def plotter_alpha_one(inputs):
    """For plotting the points on the total_fig_alpha_1 map"""

    inputs["set_of_points"].plot(
        categorical=False,
        legend=False,
        ax=inputs["ax"],
        color=inputs["colors"],
        alpha=1,
        markersize=0.05,
    )
    return inputs


def plotter_all_journeys_shrinking(inputs):
    """For plotting the points on the moving_recents_fig map"""

    lats = []
    lngs = []

    for index, row in inputs["set_of_points"].iterrows():
        lngs.append(row.geometry.x)
        lats.append(row.geometry.y)

    inputs["overall_shrinking"].append(
        inputs["ax"].scatter(
            lngs,
            lats,
            color=inputs["colors"],
            s=50,
            # marker="s",
            alpha=0.4,
        )
    )

    if len(inputs["overall_shrinking"]) > 0:
        for points in inputs["overall_shrinking"]:
            if points._sizes > 1:
                points.set_sizes(points._sizes * 0.95)
                points.set_alpha(points._alpha * 0.99)

    return inputs


def plotter_all_journeys_bubbling_off(inputs):
    """For plotting the points on the moving_recents_fig map"""

    lats = []
    lngs = []

    for index, row in inputs["set_of_points"].iterrows():
        lngs.append(row.geometry.x)
        lats.append(row.geometry.y)

    inputs["overall_bubbling_off"].append(
        inputs["ax"].scatter(
            lngs,
            lats,
            color=inputs["colors"],
            s=10,
            # marker="s",
            alpha=1,
        )
    )

    for i in range(len(inputs["overall_bubbling_off"])):
        inputs["overall_bubbling_off"][i].set_sizes(
            inputs["overall_bubbling_off"][i]._sizes * 1.2
        )
        if inputs["overall_bubbling_off"][i]._alpha > 0.0005:
            inputs["overall_bubbling_off"][i].set_alpha(
                inputs["overall_bubbling_off"][i]._alpha * 0.82
            )

    if len(inputs["overall_bubbling_off"]) > inputs["n_concurrent_bubbling"]:
        n_to_remove = (
            len(inputs["overall_bubbling_off"]) - inputs["n_concurrent_bubbling"]
        )
        for i in range(n_to_remove):
            inputs["overall_bubbling_off"][i].remove()
        for j in range(n_to_remove):
            del inputs["overall_bubbling_off"][j]

    return inputs


def plotter_end_points(inputs):
    """For plotting the points on the total_fig_alpha_1 map"""

    # Get first and last points
    lngs = [
        inputs["set_of_points"].geometry[0].x,
        inputs["set_of_points"].geometry[len(inputs["set_of_points"].geometry) - 1].x,
    ]
    lats = [
        inputs["set_of_points"].geometry[0].y,
        inputs["set_of_points"].geometry[len(inputs["set_of_points"].geometry) - 1].y,
    ]

    new_points = [shapely.geometry.Point(xy) for xy in zip(lngs, lats)]

    inputs["ax"].scatter(lngs, lats, color=["green", "blue"], s=6, alpha=0.4)

    return inputs


def plotter_end_points_bubbling_off(inputs):
    """For plotting the points on the moving_recents_fig map"""

    # Get first and last points
    lngs = [
        inputs["set_of_points"].geometry[0].x,
        inputs["set_of_points"].geometry[len(inputs["set_of_points"].geometry) - 1].x,
    ]
    lats = [
        inputs["set_of_points"].geometry[0].y,
        inputs["set_of_points"].geometry[len(inputs["set_of_points"].geometry) - 1].y,
    ]

    inputs["end_points_bubbling"].append(
        inputs["ax"].scatter(
            lngs,
            lats,
            color=["green", "blue"],
            s=10,
            # marker="s",
            alpha=1,
        )
    )

    for i in range(len(inputs["end_points_bubbling"])):
        inputs["end_points_bubbling"][i].set_sizes(
            inputs["end_points_bubbling"][i]._sizes * 1.1
        )
        if inputs["end_points_bubbling"][i]._alpha > 0.0005:
            inputs["end_points_bubbling"][i].set_alpha(
                inputs["end_points_bubbling"][i]._alpha * 0.95
            )

    if len(inputs["end_points_bubbling"]) > inputs["n_concurrent_bubbling_end_points"]:
        n_to_remove = (
            len(inputs["end_points_bubbling"])
            - inputs["n_concurrent_bubbling_end_points"]
        )
        for i in range(n_to_remove):
            inputs["end_points_bubbling"][i].remove()
        for j in range(n_to_remove):
            del inputs["end_points_bubbling"][j]

    return inputs


def plotter_end_points_shrinking(inputs):
    """For plotting the points on the moving_recents_fig map"""

    # Get first and last points
    lngs = [
        inputs["set_of_points"].geometry[0].x,
        inputs["set_of_points"].geometry[len(inputs["set_of_points"].geometry) - 1].x,
    ]
    lats = [
        inputs["set_of_points"].geometry[0].y,
        inputs["set_of_points"].geometry[len(inputs["set_of_points"].geometry) - 1].y,
    ]

    inputs["end_points_shrinking"].append(
        inputs["ax"].scatter(
            lngs,
            lats,
            color=["green", "blue"],
            s=400,
            # marker="s",
            alpha=0.4,
        )
    )

    if len(inputs["end_points_shrinking"]) > 0:
        for points in inputs["end_points_shrinking"]:
            if points._sizes > 10:
                points.set_sizes(points._sizes * 0.90)
                points.set_alpha(points._alpha * 0.985)

    return inputs


plotter_functions_dict = {
    "overall": plotter_total,
    "running_recents": plotter_moving_recents,
    "dark": plotter_no_roads,
    "dark_colours_by_time": plotter_dark_colours_by_time,
    "overall_alpha_1": plotter_alpha_one,
    "by_year": plotter_by_year,
    "overall_thick": plotter_all_journeys_thick,
    "overall_shrinking": plotter_all_journeys_shrinking,
    "overall_bubbling_off": plotter_all_journeys_bubbling_off,
    "end_points": plotter_end_points,
    "end_points_bubbling": plotter_end_points_bubbling_off,
    "end_points_shrinking": plotter_end_points_shrinking,
}
