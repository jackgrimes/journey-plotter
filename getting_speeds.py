import gpxpy
import matplotlib.pyplot as plt
import os
import numpy as np
import time

from utils import get_speeds
from configs import data_path

all_speeds = []
journeys = [
    x for x in os.listdir(os.path.join(data_path, "cycling_data")) if ".gpx" in x
]
number_to_parse = len(journeys)

for idx, journey in enumerate(journeys):
    print(
        "Parsing journey "
        + journey
        + ", number "
        + str(idx)
        + " of "
        + str(number_to_parse)
        + " ("
        + str(round(100 * (idx - 1) / number_to_parse))
        + "% complete)"
    )
    try:

        gpx = gpxpy.parse(
            open(os.path.join(os.path.join(data_path, "cycling_data"), journey))
        )
        points = []
        track = gpx.tracks[0]

        if "cycling" in gpx.tracks[0].name.lower():
            for segment in track.segments:
                for point in segment.points:
                    points.append(
                        {
                            "lat": point.latitude,
                            "lng": point.longitude,
                            "alt": point.elevation,
                            "location": point.name,
                            "time": point.time,
                        }
                    )

            lats = [x["lat"] for x in points]
            lngs = [x["lng"] for x in points]
            times = np.array(
                [time.mktime(x["time"].timetuple()) for x in points]
            )  # Extract datetimes in Unix time
            all_speeds += get_speeds(lats, lngs, times).tolist()

    except Exception as e:
        print(e)

plt.hist(x=all_speeds, bins=100)
plt.show()

# so a high speed is 18 mph
