"""
Prepare DEM Data for Fluvial Inversion

This module provides a wrapper function to convert data from the dem.py toolkit
into the format required by the fluvial inversion functions.

The fluvial inversion code requires:
- x, y: 1D arrays of pixel coordinates [L]
- z: 1D array of elevations [L]
- rec_array: 1D array of receiver indices (flow network topology)
- area_array: 1D array of drainage areas [L^2]
- slope_array: 1D array of slopes [dimensionless]

This module extracts these from dem.py's grid-based data structures.
"""

import numpy as np
from typing import Tuple, Dict, Optional


def prepare_inversion_data(
    dem,
    area,
    flow_direction,
    outlet_location: Tuple[float, float],
    min_drainage_area: float,
    max_search_length: float = np.inf
) -> Dict[str, np.ndarray]:
    """
    Extract fluvial network data from DEM grids for inversion analysis.

    This function takes dem.py grid objects and extracts the subset of pixels
    that are part of the fluvial network (drainage area >= min_drainage_area)
    and can be reached by flowing downstream from the outlet. It returns
    1D arrays in the format expected by the fluvial inversion functions.

    Parameters
    ----------
    dem : Elevation or GeographicElevation
        Digital elevation model from dem.py. Must have same grid dimensions
        as area and flow_direction.
    area : Area or GeographicArea
        Drainage area grid from dem.py [L^2]. Typically created from
        flow_direction using Area(flow_direction=flow_direction).
    flow_direction : FlowDirectionD8
        D8 flow direction grid from dem.py. Flow codes are:
        1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE.
    outlet_location : tuple of (x, y)
        Real-world coordinates of the outlet/pour point [L]. The function
        will trace upstream from this point to find the fluvial network.
    min_drainage_area : float
        Minimum drainage area threshold [L^2]. Only pixels with
        area >= min_drainage_area will be included in the output.
        This filters out hillslopes and focuses on the channel network.
    max_search_length : float, optional
        Maximum distance to search upstream from outlet [L].
        Default is np.inf (no limit).

    Returns
    -------
    dict
        Dictionary containing 1D arrays for inversion:
        - 'x': x-coordinates of fluvial network pixels [L]
        - 'y': y-coordinates of fluvial network pixels [L]
        - 'z': elevations of fluvial network pixels [L]
        - 'rec_array': receiver array, where rec_array[i] = j means
                       pixel i flows to pixel j. Outlets have rec_array[i] = i.
        - 'area_array': drainage areas [L^2]
        - 'slope_array': local slope magnitudes [dimensionless]
        - 'n_pixels': number of pixels in the network (int)

    Notes
    -----
    The receiver array (rec_array) encodes the flow network topology.
    For each pixel i, rec_array[i] gives the index (in the output arrays)
    of the downstream pixel. For outlet pixels, rec_array[i] = i.

    Slope is calculated using numpy.gradient on the DEM, then taking the
    magnitude: slope = sqrt(dz/dx^2 + dz/dy^2).

    Examples
    --------
    >>> from dem import Elevation, FlowDirectionD8, Area
    >>> # Load or create DEM
    >>> dem = Elevation(gdal_filename='my_dem.tif')
    >>> filled_dem = FilledElevation(elevation=dem)
    >>> flow_dir = FlowDirectionD8(flooded_dem=filled_dem)
    >>> area_grid = Area(flow_direction=flow_dir)
    >>>
    >>> # Extract fluvial network for inversion
    >>> outlet = (523000.0, 4567000.0)  # UTM coordinates
    >>> min_area = 1e6  # 1 km^2 in m^2
    >>> data = prepare_inversion_data(
    ...     dem=filled_dem,
    ...     area=area_grid,
    ...     flow_direction=flow_dir,
    ...     outlet_location=outlet,
    ...     min_drainage_area=min_area
    ... )
    >>>
    >>> # Use with inversion functions
    >>> from fluvial_inversion import calculate_chi, invert_block_uplift
    >>> chi = calculate_chi(
    ...     data['x'], data['y'], data['rec_array'],
    ...     data['area_array'], m=0.45
    ... )
    >>> Ustar, tstar, misfit = invert_block_uplift(
    ...     chi, data['z'], gamma=1.0, q=10, U_pri=0.0
    ... )

    See Also
    --------
    fluvial_inversion.calculate_chi : Calculate chi coordinate for inversion
    fluvial_inversion.invert_block_uplift : Invert for block uplift history
    """

    # Validate inputs
    if dem._georef_info.nx != area._georef_info.nx or \
       dem._georef_info.ny != area._georef_info.ny:
        raise ValueError("DEM and area grids must have same dimensions")

    if dem._georef_info.nx != flow_direction._georef_info.nx or \
       dem._georef_info.ny != flow_direction._georef_info.ny:
        raise ValueError("DEM and flow_direction grids must have same dimensions")

    # Step 1: Get all pixels above drainage area threshold
    # This returns a tuple of (x,y) coordinates
    candidate_pixels_xy = area.areas_greater_than(min_drainage_area)

    if len(candidate_pixels_xy) == 0:
        raise ValueError(
            f"No pixels found with drainage area >= {min_drainage_area}. "
            f"Try reducing min_drainage_area threshold."
        )

    # Step 2: Convert candidate pixels to grid indices (row, col)
    candidate_pixels_ij = flow_direction._xy_to_rowscols(candidate_pixels_xy)

    # Create a set for fast lookup
    candidate_set = set(candidate_pixels_ij)

    # Step 3: Trace upstream from outlet to find connected network
    # Search down from all candidates to see if they reach the outlet
    outlet_ij = flow_direction._xy_to_rowscols((outlet_location,))[0]

    # Build the network by finding all pixels that can reach the outlet
    network_pixels_ij = []

    for ij in candidate_pixels_ij:
        # Trace downstream from this pixel
        path = flow_direction.search_down_flow_direction_from_rowscols_location(
            ij, return_rowscols=True, search_length=max_search_length
        )

        # Check if path reaches outlet or stays within candidate pixels
        # (we want pixels that are part of the main drainage network)
        if outlet_ij in path or len(path) > 1:
            network_pixels_ij.append(ij)

    if len(network_pixels_ij) == 0:
        raise ValueError(
            f"No connected fluvial network found from outlet {outlet_location}. "
            f"Check that outlet location is correct and min_drainage_area is not too large."
        )

    n_pixels = len(network_pixels_ij)

    # Step 4: Create index mapping: (row, col) -> position in output arrays
    ij_to_index = {ij: idx for idx, ij in enumerate(network_pixels_ij)}

    # Step 5: Extract data for each pixel
    x_arr = np.zeros(n_pixels)
    y_arr = np.zeros(n_pixels)
    z_arr = np.zeros(n_pixels)
    area_arr = np.zeros(n_pixels)
    slope_arr = np.zeros(n_pixels)
    rec_array = np.zeros(n_pixels, dtype=int)

    # Calculate gradient (slope) from DEM
    dx = dem._georef_info.dx
    grad_y, grad_x = np.gradient(dem._griddata, dx)
    slope_grid = np.sqrt(grad_x**2 + grad_y**2)

    # Convert network pixels to xy coordinates
    network_pixels_xy = flow_direction._rowscols_to_xy(network_pixels_ij)

    for idx, (ij, xy) in enumerate(zip(network_pixels_ij, network_pixels_xy)):
        i, j = ij
        x, y = xy

        # Extract pixel data
        x_arr[idx] = x
        y_arr[idx] = y
        z_arr[idx] = dem._griddata[i, j]
        area_arr[idx] = area._griddata[i, j]
        slope_arr[idx] = slope_grid[i, j]

        # Find receiver (downstream pixel)
        i_rec, j_rec, is_good = flow_direction.get_flow_to_cell(i, j)

        if is_good and (i_rec, j_rec) in ij_to_index:
            # Receiver is in the network
            rec_array[idx] = ij_to_index[(i_rec, j_rec)]
        else:
            # This pixel is an outlet (flows out of network or off grid)
            rec_array[idx] = idx

    return {
        'x': x_arr,
        'y': y_arr,
        'z': z_arr,
        'rec_array': rec_array,
        'area_array': area_arr,
        'slope_array': slope_arr,
        'n_pixels': n_pixels
    }


def prepare_inversion_data_simple(
    dem,
    area,
    flow_direction,
    min_drainage_area: float
) -> Dict[str, np.ndarray]:
    """
    Simplified version that extracts ALL pixels above drainage area threshold.

    This version does not require an outlet location and simply returns all
    pixels with area >= min_drainage_area. Use this when you want to analyze
    the entire drainage network, not just a specific watershed.

    Parameters
    ----------
    dem : Elevation or GeographicElevation
        Digital elevation model from dem.py
    area : Area or GeographicArea
        Drainage area grid from dem.py [L^2]
    flow_direction : FlowDirectionD8
        D8 flow direction grid from dem.py
    min_drainage_area : float
        Minimum drainage area threshold [L^2]

    Returns
    -------
    dict
        Same structure as prepare_inversion_data()

    See Also
    --------
    prepare_inversion_data : Full version with outlet-based network extraction
    """

    # Validate inputs
    if dem._georef_info.nx != area._georef_info.nx or \
       dem._georef_info.ny != area._georef_info.ny:
        raise ValueError("DEM and area grids must have same dimensions")

    if dem._georef_info.nx != flow_direction._georef_info.nx or \
       dem._georef_info.ny != flow_direction._georef_info.ny:
        raise ValueError("DEM and flow_direction grids must have same dimensions")

    # Get all pixels above drainage area threshold
    network_pixels_xy = area.areas_greater_than(min_drainage_area)

    if len(network_pixels_xy) == 0:
        raise ValueError(
            f"No pixels found with drainage area >= {min_drainage_area}. "
            f"Try reducing min_drainage_area threshold."
        )

    # Convert to grid indices
    network_pixels_ij = flow_direction._xy_to_rowscols(network_pixels_xy)
    n_pixels = len(network_pixels_ij)

    # Create index mapping
    ij_to_index = {ij: idx for idx, ij in enumerate(network_pixels_ij)}

    # Extract data
    x_arr = np.zeros(n_pixels)
    y_arr = np.zeros(n_pixels)
    z_arr = np.zeros(n_pixels)
    area_arr = np.zeros(n_pixels)
    slope_arr = np.zeros(n_pixels)
    rec_array = np.zeros(n_pixels, dtype=int)

    # Calculate gradient (slope) from DEM
    dx = dem._georef_info.dx
    grad_y, grad_x = np.gradient(dem._griddata, dx)
    slope_grid = np.sqrt(grad_x**2 + grad_y**2)

    for idx, (ij, xy) in enumerate(zip(network_pixels_ij, network_pixels_xy)):
        i, j = ij
        x, y = xy

        # Extract pixel data
        x_arr[idx] = x
        y_arr[idx] = y
        z_arr[idx] = dem._griddata[i, j]
        area_arr[idx] = area._griddata[i, j]
        slope_arr[idx] = slope_grid[i, j]

        # Find receiver (downstream pixel)
        i_rec, j_rec, is_good = flow_direction.get_flow_to_cell(i, j)

        if is_good and (i_rec, j_rec) in ij_to_index:
            # Receiver is in the network
            rec_array[idx] = ij_to_index[(i_rec, j_rec)]
        else:
            # This pixel is an outlet
            rec_array[idx] = idx

    return {
        'x': x_arr,
        'y': y_arr,
        'z': z_arr,
        'rec_array': rec_array,
        'area_array': area_arr,
        'slope_array': slope_arr,
        'n_pixels': n_pixels
    }
