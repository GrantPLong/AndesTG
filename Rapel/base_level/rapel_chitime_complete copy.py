"""
Rapel River: Chi-to-Time Transformation Analysis
=================================================
Complete analysis for AGU 2025 Poster

This script should be run as a Jupyter notebook.
Each section between ### markers is a separate cell.

Author: Grant P. Long
Advisor: George E. Hilley
Date: December 2024
"""

###############################################################################
# CELL 1: Title and Theory
###############################################################################
"""
# Rapel River: Chi-to-Time Transformation Analysis
## Evidence for Drainage Capture from Time-Elevation Mismatches

**Author:** Grant P. Long  
**Advisor:** George E. Hilley  
**Purpose:** AGU 2025 Poster Analysis

---

## Theoretical Framework

For the **linear stream power law** (n=1):

$$\\frac{\\partial z}{\\partial t} = U(x,t) - K \\cdot A(x)^m \\cdot S$$

The **chi coordinate** (Royden & Perron 2013):

$$\\chi = \\int_{x_b}^{x} \\left(\\frac{A_0}{A(x')}\\right)^m dx'$$

At steady state: $z(\\chi) = z_b + k_{sn} \\cdot \\chi$

**Chi-to-time transformation** (Goren et al. 2014):

$$\\boxed{\\tau = \\frac{\\chi \\cdot k_{sn}}{E}}$$

where:
- τ = time (years)
- χ = chi coordinate (m)
- k_sn = dz/dχ (channel steepness)
- E = erosion rate (m/yr)

**Key Insight:** Adjacent catchments should show similar time histories IF no capture occurred.
"""

###############################################################################
# CELL 2: Imports
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Assuming you have darray imported as 'd'
# import darray as d  # Uncomment if needed

# Publication-quality plot settings
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

print("✓ Imports complete")
print(f"NumPy version: {np.__version__}")

###############################################################################
# CELL 3: Configuration
###############################################################################

# =============================================================================
# SAMPLE INFORMATION
# =============================================================================

SAMPLES = ['RP-S2', 'RP-S3', 'RP-S3u', 'RP-S4u']

# Erosion rates from cosmogenic 10Be (m/Myr)
EROSION_RATES = {
    'RP-S2':  {'mean': 84.9, 'std': 1.75,  'elev': 21},
    'RP-S3':  {'mean': 34.0, 'std': 0.642, 'elev': 23},
    'RP-S3u': {'mean': 30.0, 'std': 0.572, 'elev': 213},
    'RP-S4u': {'mean': 72.8, 'std': 2.85,  'elev': 165}
}

# Drainage areas (km²)
DRAINAGE_AREAS = {
    'RP-S2':  34.0,
    'RP-S3':  209.8,
    'RP-S3u': 112.2,
    'RP-S4u': 9.3
}

# RP-S3 sub-watershed areas (km²)
S3_AREAS = {
    'total': 209.8,
    'below_117m': 0.96,
    'below_175m': 6.41,
}
S3_AREAS['above_175m'] = S3_AREAS['total'] - S3_AREAS['below_175m']
S3_AREAS['above_117m'] = S3_AREAS['total'] - S3_AREAS['below_117m']

# Knickpoint elevations (m)
KP_LOWER = 117
KP_UPPER = 175

# Knickpoint status
KNICKPOINT_STATUS = {
    'RP-S2':  None,
    'RP-S3':  'below',
    'RP-S3u': 'above',
    'RP-S4u': None
}

# Chi calculation parameters
THETA = 0.45
A0 = 1.0

# Monte Carlo
N_MC = 100000
RANDOM_SEED = 42

# Stratigraphic constraints
NAVIDAD_AGE_MIN = 6.8
NAVIDAD_AGE_MAX = 11.5
LA_CUEVA_AGE = 4.6

# Outlet coordinates (UTM) for darray
OUTLETS = {
    'RP-S2':  (246249.2076, 6240448.447),
    'RP-S3':  (250212.9354, 6237948.848),
    'RP-S3u': (250603.5211, 6228481.157),
    'RP-S4u': (262281.9748, 6217862.614)
}

# Filtering
MIN_AREA_M2 = 1e6  # 1 km²

# Colors for plotting
COLORS = {
    'RP-S2':  '#e74c3c',
    'RP-S3':  '#3498db',
    'RP-S3u': '#2ecc71',
    'RP-S4u': '#f39c12'
}

print("\n" + "="*80)
print("CONFIGURATION LOADED")
print("="*80)
print(f"\nSamples: {', '.join(SAMPLES)}")
print(f"Knickpoints: {KP_LOWER}m, {KP_UPPER}m")
print(f"Monte Carlo iterations: {N_MC:,}")

###############################################################################
# CELL 4: Monte Carlo Deconvolution Functions
###############################################################################

def monte_carlo_deconvolution(n_iter=N_MC, seed=RANDOM_SEED):
    """
    Deconvolve RP-S3 basin-averaged erosion rate into components
    above and below knickpoints with uncertainty propagation.
    """
    np.random.seed(seed)
    
    print("\n" + "="*80)
    print("MONTE CARLO EROSION RATE DECONVOLUTION")
    print("="*80)
    print(f"Iterations: {n_iter:,}\n")
    
    # Sample from normal distributions
    E_S3 = np.random.normal(
        EROSION_RATES['RP-S3']['mean'],
        EROSION_RATES['RP-S3']['std'],
        n_iter
    )
    E_S3u = np.random.normal(
        EROSION_RATES['RP-S3u']['mean'],
        EROSION_RATES['RP-S3u']['std'],
        n_iter
    )
    
    # Enforce positivity
    E_S3 = np.maximum(E_S3, 0.1)
    E_S3u = np.maximum(E_S3u, 0.1)
    
    # Deconvolve for both knickpoint scenarios
    A_tot = S3_AREAS['total']
    
    # 175m knickpoint
    E_below_175 = (E_S3 * A_tot - E_S3u * S3_AREAS['above_175m']) / S3_AREAS['below_175m']
    delta_E_175 = E_below_175 - E_S3u
    age_175 = KP_UPPER / delta_E_175
    
    # 117m knickpoint
    E_below_117 = (E_S3 * A_tot - E_S3u * S3_AREAS['above_117m']) / S3_AREAS['below_117m']
    delta_E_117 = E_below_117 - E_S3u
    age_117 = KP_LOWER / delta_E_117
    
    # Filter valid results
    valid_175 = (delta_E_175 > 0) & (age_175 > 0) & (age_175 < 20)
    valid_117 = (delta_E_117 > 0) & (age_117 > 0) & (age_117 < 20)
    
    def get_stats(data):
        return {
            'median': np.median(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'p16': np.percentile(data, 16),
            'p84': np.percentile(data, 84)
        }
    
    results = {
        'samples': {
            'E_S3': E_S3,
            'E_S3u': E_S3u,
            'E_below_175': E_below_175[valid_175],
            'E_below_117': E_below_117[valid_117],
            'age_175': age_175[valid_175],
            'age_117': age_117[valid_117]
        },
        'stats_175': {
            'E_below': get_stats(E_below_175[valid_175]),
            'delta_E': get_stats(delta_E_175[valid_175]),
            'age': get_stats(age_175[valid_175])
        },
        'stats_117': {
            'E_below': get_stats(E_below_117[valid_117]),
            'delta_E': get_stats(delta_E_117[valid_117]),
            'age': get_stats(age_117[valid_117])
        }
    }
    
    # Print results
    print("SCENARIO 1: 175m KNICKPOINT")
    print("-" * 80)
    s = results['stats_175']
    print(f"E_below = {s['E_below']['median']:.1f} "
          f"+{s['E_below']['p84']-s['E_below']['median']:.1f}/"
          f"-{s['E_below']['median']-s['E_below']['p16']:.1f} m/Myr")
    print(f"ΔE = {s['delta_E']['median']:.1f} "
          f"+{s['delta_E']['p84']-s['delta_E']['median']:.1f}/"
          f"-{s['delta_E']['median']-s['delta_E']['p16']:.1f} m/Myr")
    print(f"Age = {s['age']['median']:.2f} "
          f"+{s['age']['p84']-s['age']['median']:.2f}/"
          f"-{s['age']['median']-s['age']['p16']:.2f} Ma")
    
    print("\nSCENARIO 2: 117m KNICKPOINT")
    print("-" * 80)
    s = results['stats_117']
    print(f"E_below = {s['E_below']['median']:.1f} "
          f"+{s['E_below']['p84']-s['E_below']['median']:.1f}/"
          f"-{s['E_below']['median']-s['E_below']['p16']:.1f} m/Myr")
    print(f"ΔE = {s['delta_E']['median']:.1f} "
          f"+{s['delta_E']['p84']-s['delta_E']['median']:.1f}/"
          f"-{s['delta_E']['median']-s['delta_E']['p16']:.1f} m/Myr")
    print(f"Age = {s['age']['median']:.3f} "
          f"+{s['age']['p84']-s['age']['median']:.3f}/"
          f"-{s['age']['median']-s['age']['p16']:.3f} Ma "
          f"({s['age']['median']*1000:.0f} ka)")
    
    return results

###############################################################################
# CELL 5: Run Monte Carlo
###############################################################################

# Run the deconvolution
mc_results = monte_carlo_deconvolution()

###############################################################################
# CELL 6: Load DEM Data
###############################################################################

"""
## Load DEM Data with darray

**Your code goes here:**
```python
import darray as d

area = d.Area.load('/Users/Glong1/Desktop/Andes/Andes_watersheds/RapelRiver/rapel_area_utm30m')
print('area loaded')
fd = d.FlowDirectionD8.load('/Users/Glong1/Desktop/Andes/Andes_watersheds/RapelRiver/rapel_fd_utm30m')
print('fd loaded')
elevation = d.Elevation.load('/Users/Glong1/Desktop/Andes/Andes_watersheds/RapelRiver/rapel_SRTMGL130m_dem_utm.tif')
print('elevation loaded')
```
"""

# Uncomment and run with your data:
# import darray as d
# area = d.Area.load('/Users/Glong1/Desktop/Andes/Andes_watersheds/RapelRiver/rapel_area_utm30m')
# fd = d.FlowDirectionD8.load('/Users/Glong1/Desktop/Andes/Andes_watersheds/RapelRiver/rapel_fd_utm30m')
# elevation = d.Elevation.load('/Users/Glong1/Desktop/Andes/Andes_watersheds/RapelRiver/rapel_SRTMGL130m_dem_utm.tif')

###############################################################################
# CELL 7: Extract Chi-Elevation Data
###############################################################################

def extract_chi_elevation_data(area, fd, elevation, outlets_dict, samples_list, 
                                theta=THETA, A0=A0, min_area=MIN_AREA_M2):
    """
    Extract chi-elevation data for all samples using darray.
    
    Returns:
        dict: {sample_name: {'chi': array, 'elev': array, 'area': array}}
    """
    chi_elev_data = {}
    
    print("\n" + "="*80)
    print("EXTRACTING CHI-ELEVATION DATA")
    print("="*80)
    
    for sample in samples_list:
        if sample not in outlets_dict:
            print(f"⚠ Warning: {sample} outlet not defined, skipping")
            continue
        
        outlet = outlets_dict[sample]
        
        print(f"\nProcessing {sample}...")
        print(f"  Outlet: {outlet}")
        
        try:
            # Calculate chi for this outlet
            chi_single = d.Chi(
                flow_direction=fd,
                area=area,
                theta=theta,
                Ao=A0,
                outlets=[outlet]
            )
            
            # Extract data where chi > 0
            chi_data = chi_single._griddata[chi_single._griddata > 0]
            elev_data = elevation._griddata[chi_single._griddata > 0]
            area_data = area._griddata[chi_single._griddata > 0]
            
            # Filter by drainage area
            mask = area_data >= min_area
            
            chi_filtered = chi_data[mask]
            elev_filtered = elev_data[mask]
            area_filtered = area_data[mask]
            
            chi_elev_data[sample] = {
                'chi': chi_filtered,
                'elev': elev_filtered,
                'area': area_filtered
            }
            
            print(f"  ✓ Extracted {len(chi_filtered):,} points")
            print(f"  ✓ Chi range: {chi_filtered.min():.1f} to {chi_filtered.max():.1f} m")
            print(f"  ✓ Elev range: {elev_filtered.min():.1f} to {elev_filtered.max():.1f} m")
            
        except Exception as e:
            print(f"  ✗ Error extracting {sample}: {e}")
    
    return chi_elev_data

# Run extraction (uncomment when you have darray data loaded):
# chi_elev_data = extract_chi_elevation_data(
#     area, fd, elevation,
#     OUTLETS, SAMPLES,
#     theta=THETA, A0=A0, min_area=MIN_AREA_M2
# )

###############################################################################
# CELL 8: Chi-to-Time Transformation Functions
###############################################################################

def calibrate_K_from_erosion(erosion_rate_m_per_myr, chi_gradient):
    """
    Calculate K from erosion rate and chi gradient.
    For n=1: K = E / (dz/dχ)
    """
    E_m_per_yr = erosion_rate_m_per_myr * 1e-6
    K = E_m_per_yr / chi_gradient
    return K

def chi_to_tau_Ma(chi_values, K_value):
    """
    Transform chi to time in Ma.
    τ = K * χ
    """
    tau_years = K_value * chi_values
    tau_Ma = tau_years / 1e6
    return tau_Ma

def process_catchment(chi_data, elev_data, erosion_rate, sample_name, 
                      has_knickpoint=None):
    """
    Process one catchment: fit chi-elevation, calibrate K, transform to time.
    """
    # Sort by chi
    sort_idx = np.argsort(chi_data)
    chi_sorted = chi_data[sort_idx]
    elev_sorted = elev_data[sort_idx]
    
    # Choose section for fitting based on knickpoint status
    if has_knickpoint == 'below':
        # Active incision zone - use lower-middle section
        chi_lb = np.percentile(chi_sorted, 10)
        chi_ub = np.percentile(chi_sorted, 60)
        zone = "Active incision (below KP)"
    elif has_knickpoint == 'above':
        # Relict surface - use upper section
        chi_lb = np.percentile(chi_sorted, 40)
        chi_ub = np.percentile(chi_sorted, 90)
        zone = "Relict surface (above KP)"
    else:
        # No knickpoint - use middle section
        chi_lb = np.percentile(chi_sorted, 30)
        chi_ub = np.percentile(chi_sorted, 70)
        zone = "No knickpoint"
    
    # Fit chi-elevation
    mask = (chi_sorted >= chi_lb) & (chi_sorted <= chi_ub)
    coeffs = np.polyfit(chi_sorted[mask], elev_sorted[mask], 1)
    chi_gradient = coeffs[0]  # k_sn = dz/dχ
    intercept = coeffs[1]
    
    # Calculate R²
    z_pred = chi_gradient * chi_sorted[mask] + intercept
    ss_res = np.sum((elev_sorted[mask] - z_pred)**2)
    ss_tot = np.sum((elev_sorted[mask] - np.mean(elev_sorted[mask]))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calibrate K
    K_val = calibrate_K_from_erosion(erosion_rate, chi_gradient)
    
    # Transform to time
    tau_Ma = chi_to_tau_Ma(chi_sorted, K_val)
    
    print(f"\n{sample_name} ({zone}):")
    print(f"  Erosion rate: {erosion_rate:.1f} m/Myr")
    print(f"  Chi gradient (dz/dχ): {chi_gradient:.4f}")
    print(f"  R² = {r_squared:.3f}")
    print(f"  Calibrated K: {K_val:.2e} m^(1-2m)/yr")
    print(f"  Time range: 0 to {tau_Ma.max():.4f} Ma (max chi: {chi_sorted.max():.1f} m)")
    
    return {
        'K': K_val,
        'tau_Ma': tau_Ma,
        'elev': elev_sorted,
        'chi': chi_sorted,
        'gradient': chi_gradient,
        'intercept': intercept,
        'r_squared': r_squared,
        'erosion_rate': erosion_rate,
        'knickpoint': has_knickpoint
    }

###############################################################################
# CELL 9: Process All Scenarios
###############################################################################

def process_all_scenarios(chi_elev_data_dict, mc_results):
    """
    Process all three scenarios:
    1. Original (basin-averaged erosion rates)
    2. Corrected (175m knickpoint)
    3. Corrected (117m knickpoint)
    """
    
    # Define scenarios
    scenarios = {
        'Original (basin-avg)': {
            s: EROSION_RATES[s]['mean'] for s in SAMPLES
        },
        'Corrected (175m KP)': {
            'RP-S2':  EROSION_RATES['RP-S2']['mean'],
            'RP-S3':  mc_results['stats_175']['E_below']['median'],
            'RP-S3u': EROSION_RATES['RP-S3u']['mean'],
            'RP-S4u': EROSION_RATES['RP-S4u']['mean']
        },
        'Corrected (117m KP)': {
            'RP-S2':  EROSION_RATES['RP-S2']['mean'],
            'RP-S3':  mc_results['stats_117']['E_below']['median'],
            'RP-S3u': EROSION_RATES['RP-S3u']['mean'],
            'RP-S4u': EROSION_RATES['RP-S4u']['mean']
        }
    }
    
    all_results = {}
    
    for scenario_name, erosion_dict in scenarios.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING: {scenario_name}")
        print(f"{'='*80}")
        
        results = {}
        
        for sample in SAMPLES:
            if sample in chi_elev_data_dict:
                data = chi_elev_data_dict[sample]
                erosion_rate = erosion_dict[sample]
                kp_status = KNICKPOINT_STATUS[sample]
                
                result = process_catchment(
                    data['chi'],
                    data['elev'],
                    erosion_rate,
                    sample,
                    has_knickpoint=kp_status
                )
                
                results[sample] = result
        
        all_results[scenario_name] = results
    
    return all_results

# Run processing (uncomment when you have chi_elev_data):
# all_results = process_all_scenarios(chi_elev_data, mc_results)

###############################################################################
# CELL 10: Plotting Functions
###############################################################################

def plot_scenario_comparison(all_results, mc_results, figsize=(20, 12)):
    """
    Plot all three scenarios side-by-side.
    Top row: Chi-Elevation
    Bottom row: Time-Elevation (THE KEY PLOTS!)
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    scenarios = list(all_results.keys())
    
    for col, scenario in enumerate(scenarios):
        results = all_results[scenario]
        
        # =====================================================================
        # CHI-ELEVATION (top row)
        # =====================================================================
        ax_chi = axes[0, col]
        
        for sample, data in results.items():
            color = COLORS[sample]
            
            # Scatter plot
            ax_chi.scatter(data['chi'], data['elev'], 
                          s=2, alpha=0.5, c=color,
                          label=f"{sample} (E={data['erosion_rate']:.1f})")
            
            # Fit line
            chi_fit = np.array([data['chi'].min(), data['chi'].max()])
            elev_fit = data['gradient'] * chi_fit + data['intercept']
            ax_chi.plot(chi_fit, elev_fit, '--', c=color, 
                       linewidth=2, alpha=0.7)
        
        # Knickpoint reference lines
        ax_chi.axhline(KP_LOWER, color='red', linestyle=':', 
                      linewidth=2, alpha=0.7, label=f'KP {KP_LOWER}m')
        ax_chi.axhline(KP_UPPER, color='orange', linestyle=':', 
                      linewidth=2, alpha=0.7, label=f'KP {KP_UPPER}m')
        
        ax_chi.set_xlabel('χ (m)', fontsize=12, fontweight='bold')
        ax_chi.set_ylabel('Elevation (m)', fontsize=12, fontweight='bold')
        ax_chi.set_title(f'{scenario}\\nχ-Elevation', 
                        fontsize=13, fontweight='bold')
        ax_chi.legend(fontsize=9, loc='best')
        ax_chi.grid(alpha=0.3)
        
        # =====================================================================
        # TIME-ELEVATION (bottom row) ★ KEY FIGURE ★
        # =====================================================================
        ax_time = axes[1, col]
        
        for sample, data in results.items():
            color = COLORS[sample]
            
            # Scatter plot
            ax_time.scatter(data['tau_Ma'], data['elev'],
                           s=2, alpha=0.5, c=color, label=sample)
        
        # Knickpoint reference lines
        ax_time.axhline(KP_LOWER, color='red', linestyle=':', 
                       linewidth=2, alpha=0.7)
        ax_time.axhline(KP_UPPER, color='orange', linestyle=':', 
                       linewidth=2, alpha=0.7)
        
        # Stratigraphic constraints
        if col == 2:  # Only on last panel
            ax_time.axvline(LA_CUEVA_AGE, color='blue', linestyle='--',
                           linewidth=1.5, alpha=0.6, label='La Cueva')
        
        ax_time.set_xlabel('Time before present (Ma)', 
                          fontsize=12, fontweight='bold')
        ax_time.set_ylabel('Elevation (m)', fontsize=12, fontweight='bold')
        ax_time.set_title('Time-Elevation', fontsize=13, fontweight='bold')
        ax_time.legend(fontsize=9, loc='best')
        ax_time.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scenario_comparison_complete.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: scenario_comparison_complete.png")
    plt.show()

def plot_monte_carlo_distributions(mc_results, figsize=(18, 10)):
    """
    Plot Monte Carlo uncertainty distributions for both knickpoint scenarios.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # ==========================================================================
    # 175m KNICKPOINT (top row)
    # ==========================================================================
    
    # E_below
    ax1 = axes[0, 0]
    ax1.hist(mc_results['samples']['E_below_175'], bins=100, density=True,
             alpha=0.7, color='steelblue', edgecolor='black')
    med = mc_results['stats_175']['E_below']['median']
    p16 = mc_results['stats_175']['E_below']['p16']
    p84 = mc_results['stats_175']['E_below']['p84']
    ax1.axvline(med, color='red', linestyle='--', linewidth=2,
               label=f"Median: {med:.1f}")
    ax1.axvline(p16, color='orange', linestyle=':', linewidth=1.5)
    ax1.axvline(p84, color='orange', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('E_below (m/Myr)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax1.set_title('175m KP: Erosion Rate Below', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # ΔE
    ax2 = axes[0, 1]
    delta_E = mc_results['samples']['E_below_175'] - 30
    ax2.hist(delta_E, bins=100, density=True,
             alpha=0.7, color='green', edgecolor='black')
    med = mc_results['stats_175']['delta_E']['median']
    ax2.axvline(med, color='red', linestyle='--', linewidth=2,
               label=f"Median: {med:.1f}")
    ax2.set_xlabel('ΔE (m/Myr)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax2.set_title('175m KP: Erosion Rate Contrast', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # Age
    ax3 = axes[0, 2]
    ax3.hist(mc_results['samples']['age_175'], bins=100, density=True,
             alpha=0.7, color='purple', edgecolor='black')
    med = mc_results['stats_175']['age']['median']
    ax3.axvline(med, color='red', linestyle='--', linewidth=2,
               label=f"Median: {med:.2f} Ma")
    ax3.axvline(LA_CUEVA_AGE, color='blue', linestyle='-', linewidth=2,
               alpha=0.7, label='La Cueva')
    ax3.axvline(NAVIDAD_AGE_MIN, color='cyan', linestyle='-', linewidth=2,
               alpha=0.7, label='Navidad')
    ax3.set_xlabel('Knickpoint Age (Ma)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax3.set_title('175m KP: Uplift Timing', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # ==========================================================================
    # 117m KNICKPOINT (bottom row)
    # ==========================================================================
    
    # E_below
    ax4 = axes[1, 0]
    ax4.hist(mc_results['samples']['E_below_117'], bins=100, density=True,
             alpha=0.7, color='steelblue', edgecolor='black')
    med = mc_results['stats_117']['E_below']['median']
    p16 = mc_results['stats_117']['E_below']['p16']
    p84 = mc_results['stats_117']['E_below']['p84']
    ax4.axvline(med, color='red', linestyle='--', linewidth=2,
               label=f"Median: {med:.1f}")
    ax4.axvline(p16, color='orange', linestyle=':', linewidth=1.5)
    ax4.axvline(p84, color='orange', linestyle=':', linewidth=1.5)
    ax4.set_xlabel('E_below (m/Myr)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax4.set_title('117m KP: Erosion Rate Below', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # ΔE
    ax5 = axes[1, 1]
    delta_E = mc_results['samples']['E_below_117'] - 30
    ax5.hist(delta_E, bins=100, density=True,
             alpha=0.7, color='green', edgecolor='black')
    med = mc_results['stats_117']['delta_E']['median']
    ax5.axvline(med, color='red', linestyle='--', linewidth=2,
               label=f"Median: {med:.1f}")
    ax5.set_xlabel('ΔE (m/Myr)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax5.set_title('117m KP: Erosion Rate Contrast', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    
    # Age
    ax6 = axes[1, 2]
    ax6.hist(mc_results['samples']['age_117'], bins=100, density=True,
             alpha=0.7, color='purple', edgecolor='black')
    med = mc_results['stats_117']['age']['median']
    ax6.axvline(med, color='red', linestyle='--', linewidth=2,
               label=f"Median: {med:.3f} Ma ({med*1000:.0f} ka)")
    ax6.set_xlabel('Knickpoint Age (Ma)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax6.set_title('117m KP: Uplift Timing', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monte_carlo_distributions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: monte_carlo_distributions.png")
    plt.show()

###############################################################################
# CELL 11: Generate All Plots
###############################################################################

# Uncomment when you have all_results from processing:
# plot_scenario_comparison(all_results, mc_results)
# plot_monte_carlo_distributions(mc_results)

###############################################################################
# CELL 12: Summary Statistics
###############################################################################

def print_timing_summary(all_results):
    """
    Print timing comparison to quantify capture signal.
    """
    print("\n" + "="*80)
    print("TIMING SUMMARY: EVIDENCE FOR DRAINAGE CAPTURE")
    print("="*80)
    
    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name}:")
        print("-" * 80)
        
        times = {}
        for sample, data in results.items():
            max_time = data['tau_Ma'].max()
            times[sample] = max_time
            print(f"  {sample:8s}: {max_time:.4f} Ma to headwaters")
        
        # Calculate ratios
        if 'RP-S3' in times and 'RP-S4u' in times:
            ratio = times['RP-S3'] / times['RP-S4u']
            diff = times['RP-S3'] - times['RP-S4u']
            print(f"\n  RP-S3 / RP-S4u ratio: {ratio:.2f}×")
            print(f"  Difference: {diff:.4f} Ma")
            if ratio > 2:
                print(f"  → STRONG CAPTURE SIGNAL!")

# Uncomment when you have all_results:
# print_timing_summary(all_results)

###############################################################################
# CELL 13: AGU Poster Key Numbers
###############################################################################

def generate_poster_numbers(all_results, mc_results):
    """
    Generate key numbers for AGU poster.
    """
    print("\n" + "="*80)
    print("KEY NUMBERS FOR AGU POSTER")
    print("="*80)
    
    print("\n1. EROSION RATES:")
    print("-" * 80)
    for sample in SAMPLES:
        E = EROSION_RATES[sample]['mean']
        std = EROSION_RATES[sample]['std']
        elev = EROSION_RATES[sample]['elev']
        print(f"  {sample}: {E:.1f} ± {std:.2f} m/Myr (at {elev} m elevation)")
    
    print("\n2. DECONVOLVED EROSION RATES (RP-S3):")
    print("-" * 80)
    s = mc_results['stats_175']
    print(f"  175m scenario: {s['E_below']['median']:.1f} "
          f"+{s['E_below']['p84']-s['E_below']['median']:.1f}/"
          f"-{s['E_below']['median']-s['E_below']['p16']:.1f} m/Myr below KP")
    
    s = mc_results['stats_117']
    print(f"  117m scenario: {s['E_below']['median']:.1f} "
          f"+{s['E_below']['p84']-s['E_below']['median']:.1f}/"
          f"-{s['E_below']['median']-s['E_below']['p16']:.1f} m/Myr below KP")
    
    print("\n3. KNICKPOINT AGES:")
    print("-" * 80)
    s = mc_results['stats_175']
    print(f"  175m: {s['age']['median']:.2f} "
          f"+{s['age']['p84']-s['age']['median']:.2f}/"
          f"-{s['age']['median']-s['age']['p16']:.2f} Ma")
    
    s = mc_results['stats_117']
    print(f"  117m: {s['age']['median']:.3f} "
          f"+{s['age']['p84']-s['age']['median']:.3f}/"
          f"-{s['age']['median']-s['age']['p16']:.3f} Ma "
          f"({s['age']['median']*1000:.0f} ka)")
    
    print("\n4. TIMING MISMATCH (from corrected scenario):")
    print("-" * 80)
    # Will calculate when all_results is available
    print("  [Run after processing to get timing ratios]")
    
    print("\n5. STRATIGRAPHIC CONTEXT:")
    print("-" * 80)
    print(f"  Navidad Fm: {NAVIDAD_AGE_MAX}-{NAVIDAD_AGE_MIN} Ma")
    print(f"  La Cueva Fm: {LA_CUEVA_AGE} Ma")

# Run this after Monte Carlo:
generate_poster_numbers({}, mc_results)

###############################################################################
# CELL 14: Complete Workflow Summary
###############################################################################

print("\n" + "="*80)
print("COMPLETE WORKFLOW SUMMARY")
print("="*80)
print("""
TO RUN THE COMPLETE ANALYSIS:

1. Load DEM data:
   import darray as d
   area = d.Area.load('path/to/area')
   fd = d.FlowDirectionD8.load('path/to/fd')
   elevation = d.Elevation.load('path/to/dem')

2. Extract chi-elevation data:
   chi_elev_data = extract_chi_elevation_data(
       area, fd, elevation, OUTLETS, SAMPLES
   )

3. Process all scenarios:
   all_results = process_all_scenarios(chi_elev_data, mc_results)

4. Generate plots:
   plot_scenario_comparison(all_results, mc_results)
   plot_monte_carlo_distributions(mc_results)

5. Print summaries:
   print_timing_summary(all_results)
   generate_poster_numbers(all_results, mc_results)

KEY OUTPUTS:
- scenario_comparison_complete.png (★ MAIN AGU FIGURE ★)
- monte_carlo_distributions.png (uncertainty quantification)
- Timing statistics for each scenario
""")
