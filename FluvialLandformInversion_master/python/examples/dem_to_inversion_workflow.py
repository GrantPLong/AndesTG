"""
Complete workflow: DEM processing to fluvial inversion

This example demonstrates the full workflow from a raw DEM file through
flow routing and drainage area calculation to fluvial inversion for
uplift history.

Dependencies:
- dem.py (included in this repository)
- fluvial_inversion package
- numpy, scipy, matplotlib

Workflow steps:
1. Load DEM from file (GeoTIFF or other GDAL-supported format)
2. Fill depressions
3. Calculate D8 flow directions
4. Calculate drainage area
5. Extract fluvial network above drainage area threshold
6. Run fluvial inversion for uplift history
7. Visualize results
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import dem module
sys.path.insert(0, '../')

# Import dem.py classes
from dem import (
    Elevation, GeographicElevation,
    FilledElevation, GeographicFilledElevation,
    FlowDirectionD8,
    Area, GeographicArea
)

# Import fluvial inversion functions
from fluvial_inversion import (
    prepare_inversion_data,
    prepare_inversion_data_simple,
    findm_slope_area,
    calculate_chi,
    invert_block_uplift
)


def dem_to_inversion_full_workflow(
    dem_filename: str,
    outlet_xy: tuple,
    min_drainage_area: float,
    gamma: float = 1.0,
    q: int = 10,
    m: float = None,
    use_geographic: bool = False
):
    """
    Complete workflow from DEM file to inversion results.

    Parameters
    ----------
    dem_filename : str
        Path to DEM file (GeoTIFF or other GDAL-supported format)
    outlet_xy : tuple of (x, y)
        Outlet location in real-world coordinates [L]
    min_drainage_area : float
        Minimum drainage area for channel network [L^2]
    gamma : float
        Tikhonov regularization parameter (default: 1.0)
    q : int
        Number of time intervals for inversion (default: 10)
    m : float, optional
        Concavity index. If None, will be calculated using slope-area method
    use_geographic : bool
        If True, use geographic coordinate-aware classes (default: False)

    Returns
    -------
    dict
        Results containing all intermediate and final data
    """

    print("=" * 70)
    print("FLUVIAL INVERSION WORKFLOW: DEM TO UPLIFT HISTORY")
    print("=" * 70)

    # Step 1: Load DEM
    print("\n1. Loading DEM...")
    if use_geographic:
        dem = GeographicElevation(gdal_filename=dem_filename)
    else:
        dem = Elevation(gdal_filename=dem_filename)

    print(f"   DEM size: {dem._georef_info.nx} x {dem._georef_info.ny} pixels")
    print(f"   Resolution: {dem._georef_info.dx} m")

    # Step 2: Fill depressions
    print("\n2. Filling depressions...")
    if use_geographic:
        filled_dem = GeographicFilledElevation(elevation=dem)
    else:
        filled_dem = FilledElevation(elevation=dem)

    # Step 3: Calculate flow directions
    print("\n3. Calculating D8 flow directions...")
    flow_dir = FlowDirectionD8(flooded_dem=filled_dem)

    # Step 4: Calculate drainage area
    print("\n4. Calculating drainage area...")
    if use_geographic:
        area_grid = GeographicArea(flow_direction=flow_dir)
    else:
        area_grid = Area(flow_direction=flow_dir)

    max_area = np.nanmax(area_grid._griddata)
    print(f"   Maximum drainage area: {max_area:.2e} m^2 ({max_area/1e6:.2f} km^2)")

    # Step 5: Extract fluvial network
    print("\n5. Extracting fluvial network...")
    print(f"   Outlet location: ({outlet_xy[0]:.1f}, {outlet_xy[1]:.1f})")
    print(f"   Min drainage area: {min_drainage_area:.2e} m^2 ({min_drainage_area/1e6:.2f} km^2)")

    network_data = prepare_inversion_data(
        dem=filled_dem,
        area=area_grid,
        flow_direction=flow_dir,
        outlet_location=outlet_xy,
        min_drainage_area=min_drainage_area
    )

    print(f"   Network size: {network_data['n_pixels']} pixels")

    # Step 6: Calculate concavity index (if not provided)
    if m is None:
        print("\n6. Calculating concavity index (m) using slope-area method...")
        m, m_lb, m_ub, R2 = findm_slope_area(
            network_data['slope_array'],
            network_data['area_array'],
            to_plot=False
        )
        print(f"   m = {m:.4f} (95% CI: [{m_lb:.4f}, {m_ub:.4f}]), R^2 = {R2:.4f}")
    else:
        print(f"\n6. Using provided concavity index: m = {m:.4f}")

    # Step 7: Calculate chi coordinate
    print("\n7. Calculating chi coordinate...")
    chi = calculate_chi(
        network_data['x'],
        network_data['y'],
        network_data['rec_array'],
        network_data['area_array'],
        m=m,
        A0=1.0
    )
    print(f"   Chi range: [{np.min(chi):.1f}, {np.max(chi):.1f}] m")

    # Step 8: Run inversion
    print("\n8. Running block uplift inversion...")
    print(f"   Regularization (gamma): {gamma}")
    print(f"   Number of time intervals (q): {q}")

    Ustar, tstar, misfit = invert_block_uplift(
        chi=chi,
        z=network_data['z'],
        gamma=gamma,
        q=q,
        U_pri=0.0
    )

    print(f"   Misfit: {misfit:.4f} m")
    print(f"   Uplift rate range: [{np.min(Ustar):.4e}, {np.max(Ustar):.4e}] m/m_chi")

    # Step 9: Compile results
    print("\n9. Compiling results...")
    results = {
        'dem': dem,
        'filled_dem': filled_dem,
        'flow_direction': flow_dir,
        'area': area_grid,
        'network': network_data,
        'm': m,
        'chi': chi,
        'Ustar': Ustar,
        'tstar': tstar,
        'misfit': misfit,
        'gamma': gamma,
        'q': q
    }

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)

    return results


def plot_inversion_results(results):
    """
    Create diagnostic plots for inversion results.

    Parameters
    ----------
    results : dict
        Results dictionary from dem_to_inversion_full_workflow()
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Chi-elevation profile
    ax = axes[0, 0]
    ax.scatter(
        results['chi'],
        results['network']['z'],
        c=results['network']['area_array'],
        s=5,
        alpha=0.6,
        cmap='viridis'
    )
    ax.set_xlabel('Chi (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Chi-Elevation Profile')
    ax.grid(True, alpha=0.3)

    # Plot 2: Uplift rate history
    ax = axes[0, 1]
    ax.plot(results['tstar'], results['Ustar'], 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Dimensionless Time (chi/m)')
    ax.set_ylabel('Dimensionless Uplift Rate')
    ax.set_title(f'Uplift History (misfit = {results["misfit"]:.4f} m)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Drainage area distribution
    ax = axes[1, 0]
    ax.hist(np.log10(results['network']['area_array']), bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('log10(Drainage Area) [m^2]')
    ax.set_ylabel('Number of Pixels')
    ax.set_title('Drainage Area Distribution')
    ax.grid(True, alpha=0.3)

    # Plot 4: Slope-area relationship
    ax = axes[1, 1]
    valid = (results['network']['slope_array'] > 0) & (results['network']['area_array'] > 0)
    ax.loglog(
        results['network']['area_array'][valid],
        results['network']['slope_array'][valid],
        'o',
        alpha=0.3,
        markersize=3
    )
    ax.set_xlabel('Drainage Area (m^2)')
    ax.set_ylabel('Slope')
    ax.set_title(f'Slope-Area (m = {results["m"]:.4f})')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    return fig


def simple_workflow_example(dem_filename: str, min_drainage_area: float):
    """
    Simplified workflow using prepare_inversion_data_simple().

    This version doesn't require an outlet location and extracts all
    pixels above the drainage area threshold.

    Parameters
    ----------
    dem_filename : str
        Path to DEM file
    min_drainage_area : float
        Minimum drainage area for channel network [L^2]
    """

    print("\n" + "=" * 70)
    print("SIMPLIFIED WORKFLOW (no outlet required)")
    print("=" * 70)

    # Load and process DEM
    print("\nLoading and processing DEM...")
    dem = Elevation(gdal_filename=dem_filename)
    filled_dem = FilledElevation(elevation=dem)
    flow_dir = FlowDirectionD8(flooded_dem=filled_dem)
    area_grid = Area(flow_direction=flow_dir)

    # Extract all pixels above threshold
    print(f"\nExtracting network (min area = {min_drainage_area:.2e} m^2)...")
    network_data = prepare_inversion_data_simple(
        dem=filled_dem,
        area=area_grid,
        flow_direction=flow_dir,
        min_drainage_area=min_drainage_area
    )

    print(f"Network size: {network_data['n_pixels']} pixels")

    # Calculate m and chi
    m, _, _, R2 = findm_slope_area(
        network_data['slope_array'],
        network_data['area_array'],
        to_plot=False
    )
    print(f"Concavity index: m = {m:.4f} (R^2 = {R2:.4f})")

    chi = calculate_chi(
        network_data['x'],
        network_data['y'],
        network_data['rec_array'],
        network_data['area_array'],
        m=m
    )

    # Run inversion
    Ustar, tstar, misfit = invert_block_uplift(chi, network_data['z'], gamma=1.0, q=10)
    print(f"Inversion misfit: {misfit:.4f} m")

    return {
        'network': network_data,
        'm': m,
        'chi': chi,
        'Ustar': Ustar,
        'tstar': tstar,
        'misfit': misfit
    }


if __name__ == "__main__":
    """
    Example usage with synthetic or real DEM data.

    To run this script, you need:
    1. A DEM file (GeoTIFF or other GDAL format)
    2. Knowledge of outlet location (x, y coordinates)
    3. Appropriate drainage area threshold for your region
    """

    # Example 1: Full workflow with outlet
    # --------------------------------------
    # Uncomment and modify these parameters for your data:
    #
    # dem_file = '/path/to/your/dem.tif'
    # outlet_location = (523000.0, 4567000.0)  # UTM coordinates (m)
    # min_area = 1e6  # 1 km^2 in m^2
    #
    # results = dem_to_inversion_full_workflow(
    #     dem_filename=dem_file,
    #     outlet_xy=outlet_location,
    #     min_drainage_area=min_area,
    #     gamma=1.0,
    #     q=10
    # )
    #
    # # Plot results
    # fig = plot_inversion_results(results)
    # plt.savefig('inversion_results.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # Example 2: Simplified workflow (no outlet required)
    # ---------------------------------------------------
    # Uncomment and modify for your data:
    #
    # dem_file = '/path/to/your/dem.tif'
    # min_area = 1e6  # 1 km^2
    #
    # results = simple_workflow_example(dem_file, min_area)

    print("\n" + "=" * 70)
    print("EXAMPLE SCRIPT")
    print("=" * 70)
    print("\nThis script demonstrates two workflows:\n")
    print("1. Full workflow with outlet location:")
    print("   - Extracts connected network from specified outlet")
    print("   - Useful for analyzing a specific watershed")
    print()
    print("2. Simplified workflow (no outlet):")
    print("   - Extracts all pixels above drainage area threshold")
    print("   - Useful for regional analysis")
    print()
    print("To use this script:")
    print("1. Modify the parameters in the __main__ section above")
    print("2. Uncomment the example you want to run")
    print("3. Run: python dem_to_inversion_workflow.py")
    print()
    print("=" * 70)
