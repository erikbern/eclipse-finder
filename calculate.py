import io
from datetime import datetime, timedelta, timezone

import modal

stub = modal.Stub()

image = modal.Image.debian_slim().pip_install(
    "scipy",
    "astropy",
    "async-timeout",
    "jplephem",
    "matplotlib",
    "basemap",
    "timezonefinder",
    "pytz",
)

with image.imports():
    from scipy.optimize import minimize
    from astropy.coordinates import AltAz, EarthLocation, get_body
    from astropy.time import Time
    from astropy.units import deg, m

    # from astropy.utils import iers

    from mpl_toolkits.basemap import Basemap
    from matplotlib import pyplot

    from timezonefinder import TimezoneFinder
    import pytz


def sun_moon_separation(lat: float, lon: float, t: float) -> float:
    loc = EarthLocation(lat=lat * deg, lon=lon * deg, height=0 * m)
    time = Time(t, format="unix")
    moon = get_body("moon", time, loc)
    sun = get_body("sun", time, loc)

    # Check that the sun and moon aren visible
    az = AltAz(obstime=time, location=loc)
    sun_az = sun.transform_to(az)
    moon_az = sun.transform_to(az)
    if sun_az.alt < 0 or moon_az.alt < 0:
        return 180

    # They are visible, return the separation
    sep = moon.separation(sun)
    return sep.deg


@stub.function(image=image)
def find_eclipse_location(dt: datetime) -> tuple[datetime, float, float] | None:
    """Given a timestamp, return the location on earth of an eclipse, or None."""
    t = datetime.timestamp(dt)
    fun = lambda x: sun_moon_separation(x[0], x[1], t)

    # Pick a starting point through a simple grid search
    x0s = [
        (lat, lon)
        for lat in [-75, -45, -15, 15, 45, 75]
        for lon in [-150, -90, -30, 30, 90, 150]
    ]
    x0 = min(x0s, key=fun)

    # Search
    ret = minimize(fun, bounds=[(-90, 90), (-180, 180)], x0=x0)

    if ret.fun < 1e-3:
        lat, lon = ret.x
        return (dt, lat, lon)
    else:
        return None


def gen_dts(dt_a: datetime, dt_b: datetime, sec_delta: float) -> list[datetime]:
    dt = dt_a
    dts = []
    while dt < dt_b:
        dts.append(dt)
        dt = dt + timedelta(seconds=sec_delta)
    return dts


@stub.function(image=image)
def plot_path(dts: list[datetime], lats: list[float], lons: list[float]) -> bytes:
    # Set up a world map
    pyplot.figure(figsize=(6, 6))
    lat_0, lon_0 = lats[len(lats) // 2], lons[len(lons) // 2]
    bm = Basemap(projection="ortho", lat_0=lat_0, lon_0=lon_0)
    bm.drawmapboundary(fill_color="navy")
    bm.fillcontinents(color="forestgreen", lake_color="blue")
    bm.drawcoastlines()

    # Plot eclipse path
    x, y = bm(lons, lats)
    bm.plot(x, y, color="red")

    # Title
    dt = dts[len(dts) // 2]
    pyplot.title(f"Eclipse on {dt.date()} (local and UTC times)")

    # Annotate with times
    tzf = TimezoneFinder()
    n_times = 15
    for step in range(n_times):
        i = int((len(dts) - 1) * step / (n_times - 1))
        dt, lat, lon = dts[i], lats[i], lons[i]

        # Local time
        tz_str = tzf.timezone_at(lng=lon, lat=lat)
        tz = pytz.timezone(tz_str)
        dt_local = dt.astimezone(tz)
        pyplot.annotate(
            dt_local.strftime("%H:%M"),
            xy=bm(lon, lat),
            ha="center",
            va="bottom",
            color="yellow",
            fontsize=5,
        )

        # UTC time
        pyplot.annotate(
            dt.strftime("%H:%M"),
            xy=bm(lon, lat),
            ha="center",
            va="top",
            color="orange",
            fontsize=5,
        )

    pyplot.tight_layout()
    buf = io.BytesIO()
    pyplot.savefig(buf, dpi=300)
    return buf.getvalue()


@stub.function(image=image)
def plot_eclipse(dt_min: datetime, dt_max: datetime) -> tuple[datetime, bytes]:
    # Generate minute-level timestamps
    print(f"Finding path of eclipse from {dt_min} to {dt_max}")
    dt_a = dt_min - timedelta(seconds=3600)
    dt_b = dt_max + timedelta(seconds=3600)
    dts, lats, lons = [], [], []
    for tup in find_eclipse_location.map(gen_dts(dt_a, dt_b, 60)):
        if tup is not None:
            dt, lat, lon = tup
            dts.append(dt)
            lats.append(lat)
            lons.append(lon)

    # Plot the path
    print(f"Plotting eclipse from {dt_min} to {dt_max}")
    png_data = plot_path.remote(dts, lats, lons)
    return dts[0], png_data


@stub.local_entrypoint()
def run():
    dt_a = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dt_b = datetime(2030, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Generate even-hour spaced intervals
    dts = gen_dts(dt_a, dt_b, 3600)

    # Find eclipses by mapping over all hours
    eclipses = []
    for tup in find_eclipse_location.map(dts):
        if tup is not None:
            dt, _, _ = tup
            if len(eclipses) == 0 or dt - eclipses[-1][-1] > timedelta(seconds=3601):
                eclipses.append([])
            eclipses[-1].append(dt)

    # Pick the min, max of each eclipse
    eclipses = [(min(e), max(e)) for e in eclipses]

    # For each eclipse, plot the path
    for dt, png_data in plot_eclipse.starmap(eclipses):
        with open(f"output/eclipse-{dt.date()}.png", "wb") as f:
            f.write(png_data)
