import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider

def latlon_to_local(lat, lon, lat0, lon0):
    lat = np.radians(lat)
    lon = np.radians(lon)

    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    EARTH_RADIUS = 6378137.0
    north = (lat - lat0) * EARTH_RADIUS
    east = (lon - lon0) * EARTH_RADIUS * np.cos(lat0)

    return east, north

if __name__ == "__main__":
    gnss = pd.read_csv("gnssinfo.csv")
    filtered = pd.read_csv("filteredDataInfo.csv")

    if "satellites" in gnss.columns:
        # tutaj usuwam bledy gps
        # prosta heurystyka; im mniej satelit tym gorsza pozycja
        gnss = gnss[gnss["satellites"] >= 6].copy()

    lat0 = gnss.iloc[0]["latitude"]
    lon0 = gnss.iloc[0]["longitude"]

    x, y = latlon_to_local(gnss["latitude"].values, gnss["longitude"].values, lat0, lon0)

    gnss["x"] = x
    gnss["y"] = y

    # ze wzgledu na rozne probkowanie to trzeba interpolowac wysokosc ale jest to tak jakby ok
    gnss["z"] = np.interp(gnss["ts"], filtered["ts"], filtered["filteredAltitudeAGL"])

    gnss["x"] -= gnss["x"].iloc[0]
    gnss["y"] -= gnss["y"].iloc[0]
    gnss["z"] -= gnss["z"].iloc[0]

    trajectory = gnss[["ts", "x", "y", "z"]]

    trajectory.to_csv("trajectory.csv", index=False)

    # dalsza czesc to taki lady wykresik obrotowy

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    plt.subplots_adjust(bottom=0.18)

    ax.plot(
        trajectory["x"],
        trajectory["y"],
        trajectory["z"],
        color="green",
        linewidth=2
    )

    ax.scatter(
        trajectory["x"].iloc[0],
        trajectory["y"].iloc[0],
        trajectory["z"].iloc[0],
        color="green",
        s=50
    )

    ax.scatter(
        trajectory["x"].iloc[-1],
        trajectory["y"].iloc[-1],
        trajectory["z"].iloc[-1],
        color="red",
        s=50
    )

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Altitude [m]")

    ax.set_title("3D Trajectory")

    ax.view_init(elev=30, azim=-60)

    ax_elev = plt.axes((0.15, 0.08, 0.7, 0.03))
    ax_azim = plt.axes((0.15, 0.03, 0.7, 0.03))

    slider_elev = Slider(
        ax=ax_elev,
        label="Elevation",
        valmin=0,
        valmax=90,
        valinit=30
    )

    slider_azim = Slider(
        ax=ax_azim,
        label="Azimuth",
        valmin=-180,
        valmax=180,
        valinit=-60
    )

    def update(val):
        ax.view_init(
            elev=slider_elev.val,
            azim=slider_azim.val
        )
        fig.canvas.draw_idle()

    slider_elev.on_changed(update)
    slider_azim.on_changed(update)

    plt.show()