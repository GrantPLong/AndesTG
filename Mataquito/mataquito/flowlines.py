"""D8 flowline tracing and stream distance calculations.

Extracted from NB07 (downstream confluence mixing).
"""

import geopandas as gpd
import rasterio
from shapely.geometry import LineString, Point

# Standard D8 flow direction codes
FLOW_DIRECTIONS = {
    1: (0, 1),       # East
    2: (1, 1),       # Southeast
    4: (1, 0),       # South
    8: (1, -1),      # Southwest
    16: (0, -1),     # West
    32: (-1, -1),    # Northwest
    64: (-1, 0),     # North
    128: (-1, 1),    # Northeast
}


def create_flowlines_from_raster(flow_direction_path, area_path, samples_gdf,
                                 min_drainage_area=50):
    """Trace D8 flowlines downstream from each sample point.

    Parameters
    ----------
    flow_direction_path : str or path-like
        Path to the flow direction raster (D8 encoding).
    area_path : str or path-like
        Path to the drainage area raster.
    samples_gdf : GeoDataFrame
        Sample points with a ``Name`` column and point geometries.
    min_drainage_area : float
        Minimum drainage area threshold to continue tracing.

    Returns
    -------
    GeoDataFrame
        Flowlines with columns ``sample_name``, ``geometry``, ``length_km``.
    """
    with rasterio.open(flow_direction_path) as fd_src:
        flow_direction = fd_src.read(1)
        transform = fd_src.transform

    with rasterio.open(area_path) as area_src:
        drainage_area = area_src.read(1)

    def coords_to_pixel(x, y):
        col, row = ~transform * (x, y)
        return int(row), int(col)

    def pixel_to_coords(row, col):
        x, y = transform * (col, row)
        return x, y

    def trace_downstream(start_row, start_col, max_length=10000):
        path = []
        current_row, current_col = start_row, start_col
        for _ in range(max_length):
            if (current_row < 0 or current_row >= flow_direction.shape[0] or
                    current_col < 0 or current_col >= flow_direction.shape[1]):
                break
            x, y = pixel_to_coords(current_row, current_col)
            path.append((x, y))
            fd_value = flow_direction[current_row, current_col]
            if fd_value == 0 or fd_value not in FLOW_DIRECTIONS:
                break
            if drainage_area[current_row, current_col] < min_drainage_area:
                break
            row_offset, col_offset = FLOW_DIRECTIONS[fd_value]
            current_row += row_offset
            current_col += col_offset
        return path

    flowlines = []
    for idx, sample in samples_gdf.iterrows():
        sample_name = sample["Name"]
        sample_point = sample.geometry
        start_row, start_col = coords_to_pixel(sample_point.x, sample_point.y)
        path = trace_downstream(start_row, start_col)
        if len(path) > 1:
            line = LineString(path)
            flowlines.append({
                "sample_name": sample_name,
                "geometry": line,
                "length_km": line.length / 1000,
            })
        else:
            print(f"Warning: Could not create flowline for {sample_name}")

    if flowlines:
        return gpd.GeoDataFrame(flowlines, crs=samples_gdf.crs)
    print("Warning: No flowlines created")
    return gpd.GeoDataFrame(
        columns=["sample_name", "geometry", "length_km"], crs=samples_gdf.crs
    )


def calculate_distance_along_line(line, point):
    """Distance (km) from the start of *line* to the projection of *point*."""
    return line.project(point) / 1000


def substring_line(line, start_distance, end_distance):
    """Extract a portion of a LineString between two distances (in CRS units)."""
    if start_distance <= 0 and end_distance >= line.length:
        return line

    start_point = line.interpolate(start_distance)
    end_point = line.interpolate(end_distance)
    coords = list(line.coords)

    new_coords = [(start_point.x, start_point.y)]
    current_distance = 0

    for i in range(len(coords) - 1):
        segment_start = Point(coords[i])
        segment_end = Point(coords[i + 1])
        segment_length = segment_start.distance(segment_end)

        if current_distance + segment_length > start_distance and current_distance < end_distance:
            if current_distance > start_distance:
                new_coords.append(coords[i])
            if current_distance + segment_length < end_distance:
                new_coords.append(coords[i + 1])

        current_distance += segment_length
        if current_distance >= end_distance:
            break

    new_coords.append((end_point.x, end_point.y))

    unique_coords = []
    for coord in new_coords:
        if not unique_coords or coord != unique_coords[-1]:
            unique_coords.append(coord)

    if len(unique_coords) >= 2:
        return LineString(unique_coords)
    return LineString([new_coords[0], new_coords[-1]])


def calculate_stream_distances_from_confluence(samples_gdf, flowlines_gdf,
                                               target_samples=None,
                                               tributary_samples=None,
                                               confluence_coords=(282071, 6126393)):
    """Calculate stream distances from each sample to the CT-5/CT-6 confluence.

    Parameters
    ----------
    samples_gdf : GeoDataFrame
        Sample points with ``Name`` column.
    flowlines_gdf : GeoDataFrame
        Flowlines from :func:`create_flowlines_from_raster`.
    target_samples : list of str
        Downstream sample IDs.
    tributary_samples : list of str
        Upstream tributary sample IDs.
    confluence_coords : tuple
        (easting, northing) in the CRS of the data.

    Returns
    -------
    distances : dict
        ``{Sample_ID: distance_km}``
    confluence_point : Point
    """
    if target_samples is None:
        target_samples = ["CT-4", "CT-10", "CT-11", "CT-8"]
    if tributary_samples is None:
        tributary_samples = ["CT-5", "CT-6"]

    distances = {}
    confluence_point = Point(confluence_coords[0], confluence_coords[1])

    # Upstream samples
    for sample_name in tributary_samples:
        sample_flowline = flowlines_gdf[flowlines_gdf["sample_name"] == sample_name]
        if sample_flowline.empty:
            continue
        line = sample_flowline.geometry.iloc[0]
        distances[sample_name] = calculate_distance_along_line(line, confluence_point)

    # CT-4 (downstream of confluence along the CT-5 flowline)
    if "CT-4" in target_samples:
        ct5_flowline = flowlines_gdf[flowlines_gdf["sample_name"] == "CT-5"]
        if not ct5_flowline.empty:
            ct5_line = ct5_flowline.geometry.iloc[0]
            confluence_distance = ct5_line.project(confluence_point)
            remaining = substring_line(ct5_line, confluence_distance, ct5_line.length)
            ct4_sample = samples_gdf[samples_gdf["Name"] == "CT-4"].geometry.iloc[0]
            distances["CT-4"] = remaining.project(ct4_sample) / 1000

    # Other downstream samples
    other_samples = [s for s in target_samples if s != "CT-4"]
    for sample_name in other_samples:
        sample_flowline = flowlines_gdf[flowlines_gdf["sample_name"] == sample_name]
        if sample_flowline.empty:
            continue
        line = sample_flowline.geometry.iloc[0]
        projected_distance = line.project(confluence_point)
        point_on_line = line.interpolate(projected_distance)
        if confluence_point.distance(point_on_line) < 100:
            distances[sample_name] = projected_distance / 1000
        else:
            sample_point = samples_gdf[samples_gdf["Name"] == sample_name].geometry.iloc[0]
            distances[sample_name] = sample_point.distance(confluence_point) / 1000

    return distances, confluence_point


def calculate_stream_distances_from_outlet(samples_gdf, flowlines_gdf,
                                           outlet_coords=(208913.7, 6126075.0)):
    """Calculate stream distances from each sample to the watershed outlet.

    Parameters
    ----------
    samples_gdf : GeoDataFrame
        Sample points with ``Name`` column.
    flowlines_gdf : GeoDataFrame
        Flowlines from :func:`create_flowlines_from_raster`.
    outlet_coords : tuple
        (easting, northing) of the outlet in the CRS of the data.

    Returns
    -------
    distances : dict
        ``{Sample_ID: distance_km}``
    outlet_point : Point
    """
    distances = {}
    outlet_point = Point(outlet_coords[0], outlet_coords[1])

    for sample_name in samples_gdf["Name"].unique():
        sample_flowline = flowlines_gdf[flowlines_gdf["sample_name"] == sample_name]
        if sample_flowline.empty:
            continue

        line = sample_flowline.geometry.iloc[0]
        sample_point = samples_gdf[samples_gdf["Name"] == sample_name].geometry.iloc[0]

        projected_distance = line.project(outlet_point)
        point_on_line = line.interpolate(projected_distance)

        if outlet_point.distance(point_on_line) < 500:
            sample_distance = line.project(sample_point)
            if sample_distance < projected_distance:
                distance = (projected_distance - sample_distance) / 1000
            else:
                distance = (sample_distance - projected_distance) / 1000
            distances[sample_name] = distance
        else:
            flowline_length = line.length / 1000
            line_end = Point(line.coords[-1])
            end_to_outlet = line_end.distance(outlet_point) / 1000
            distances[sample_name] = flowline_length + end_to_outlet

    return distances, outlet_point
