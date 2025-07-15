"""Segmentation"""


import seaborn as sns
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from aind_hcr_data_loader.hcr_dataset import HCRDataset

def plot_cell_volume_and_diameter(cell_df, q1=None, q2=None, 
                                  resolution={'x': 0.24, 'y': 0.24, 'z': 1.0}, 
                                  figsize=(10, 4),
                                  fig = None,
                                  axes = None):
    """
    Plot cell volume distribution and estimated cell diameter based on volume.
    
    Parameters:
    -----------
    cell_df : pandas.DataFrame
        DataFrame containing cell information with a 'volume' column
    q1, q2 : float
        Quantiles for filtering and showing on plots (default: 0.05, 0.95)
    resolution : dict
        Resolution in microns per voxel for x, y, z dimensions
    figsize : tuple
        Figure size for the plot
    fig : matplotlib figure, optional
        Existing figure to use for plotting, if None a new figure is created
    axes : list of matplotlib axes, optional
        Existing axes to use for plotting, if None new axes are created
        
    Returns:
    --------
    fig : matplotlib figure
        The figure object containing the plots
    diameters : numpy.ndarray
        The calculated cell diameters in microns
    """
    import matplotlib.pyplot as plt
    
    # Need to import seaborn for KDE plot
    
    # Create figure with two subplots
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1, ax2 = axes

    # Plot 1: Cell volume distribution
    ax1.hist(cell_df["volume"], bins=75, color='lightblue', edgecolor='black')

    if q1 is not None:
        ax1.axvline(cell_df["volume"].quantile(q1), color='red', linestyle='dashed', 
                    linewidth=1, label=f'min_per: ({q1})')

    if q2 is not None:
        ax1.axvline(cell_df["volume"].quantile(q2), color='green', linestyle='dashed', 
                    linewidth=1, label=f'max_per: ({q2})')
    ax1.axvline(cell_df["volume"].median(), color='black', linestyle='dashed',
                linewidth=1, label=f'median: {int(cell_df["volume"].median())} voxels')
    ax1.set_title('Cell Volume Distribution')
    ax1.set_xlabel('Cell Volume (voxels)')
    ax1.set_ylabel('Frequency')
    # sci notation for x axis
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax1.legend()

    # Convert volume from voxels to cubic microns
    volume_conversion = resolution['x'] * resolution['y'] * resolution['z']
    volumes_um3 = cell_df["volume"] * volume_conversion

    # Calculate diameter assuming spherical cells (d = 2 * (3V/4π)^(1/3))
    diameters = 2 * ((3 * volumes_um3) / (4 * np.pi))**(1/3)

    # Plot 2: Estimated cell diameter
    sns.kdeplot(diameters, ax=ax2, fill=True, color='skyblue')
    if q1 is not None:
        ax2.axvline(diameters.quantile(q1), color='red', linestyle='dashed', 
                    linewidth=1, label=f'min_per: ({q1})')
    if q2 is not None:
        ax2.axvline(diameters.quantile(q2), color='green', linestyle='dashed', 
                    linewidth=1, label=f'max_per: ({q2})')
    ax2.axvline(diameters.median(), color='black', linestyle='dashed', 
                linewidth=1, label=f'median: {diameters.median():.2f} µm')
                
    ax2.set_title('Estimated Cell Diameter (assuming sphere)')
    ax2.set_xlabel('Estimated Diameter (µm)')
    ax2.set_ylabel('Density')
    ax2.legend()

    plt.tight_layout()

    print_stats = True
    if print_stats:
        print(f"Cell diameter statistics (µm):")
        print(f"Min: {diameters.min():.2f}")
        print(f"Max: {diameters.max():.2f}")
        print(f"Mean: {diameters.mean():.2f}")
        print(f"Median: {diameters.median():.2f}")
        print(f"5th percentile: {diameters.quantile(0.05):.2f}")
        print(f"95th percentile: {diameters.quantile(0.95):.2f}")
    return fig, diameters



def filter_cell_info(cell_info, q1=0.2, q2=0.95):
    n_cells = cell_info.shape[0]
    cell_info = cell_info[(cell_info["volume"] > cell_info["volume"].quantile(q1)) & (cell_info["volume"] < cell_info["volume"].quantile(q2))]
    print(f"Kept {cell_info.shape[0]} cells out of {n_cells} total cells, based on volume quantiles {q1} and {q2}.")
    return cell_info


def fig_centroids_filtered(cell_info, q1=0.2, q2=0.95, save=False, output_dir=None):
    """
    Create a figure with two subplots showing cell volume and diameter distributions
    before and after filtering.
    
    Parameters:
    -----------
    cell_info : pandas.DataFrame
        DataFrame containing cell information with volume data
    q1, q2 : float
        Lower and upper quantile thresholds for volume filtering
    save : bool
        Whether to save the figure to disk
    output_dir : Path or str
        Directory to save the figure if save=True
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The combined figure
    cell_df_filt : pandas.DataFrame
        The filtered cell dataframe
    """
    # Create the filtered dataframe
    cell_df_filt = filter_cell_info(cell_info, q1=q1, q2=q2)
    
    # Create a figure with subfigures
    fig = plt.figure(figsize=(12, 8))
    subfigs = fig.subfigures(2, 1, wspace=0.07, hspace=0.5)
    
    # Original data plot
    ax1 = subfigs[0].subplots(1, 2)
    _, diameters = plot_cell_volume_and_diameter(cell_info, q1=q1, q2=q2, fig=subfigs[0], axes=ax1)
    subfigs[0].suptitle("Original Cell Distribution", fontsize=14, y=1.25)
    
    # Filtered data plot
    ax2 = subfigs[1].subplots(1, 2)
    _, diameters = plot_cell_volume_and_diameter(cell_df_filt, q1=None, q2=None, fig=subfigs[1], axes=ax2)
    subfigs[1].suptitle("Filtered Cell Distribution", fontsize=14, y=1.25)
    
    # Add overall title
    #fig.suptitle(f"Cell Volume and Diameter Distributions (filtered: q1={q1}, q2={q2})", fontsize=16)
    
    # Save figure if requested
    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        fig_path = output_dir / f"cell_volume_diameter_q{q1}_{q2}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
    
    return fig, cell_df_filt

def qc_segmentation(dataset: HCRDataset, 
                    data_dir = Path("/root/capsule/data"), 
                    output_dir = Path("/root/capsule/stratch"),
                    verbose=True):
    """
    Run segmentation quality control analysis.

    Parameters:
    -----------
    dataset : HCRDataset
        The dataset object containing cell information
    data_dir : Path or str
        Path to data directory containing segmentation masks
    output_dir : Path or str
        Directory to save QC outputs
    verbose : bool
        Enable verbose output
    """

    # call fig_centroids_filtered
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    qc_dir = output_dir / "segmentation_qc"
    qc_dir.mkdir(exist_ok=True, parents=True)
    if verbose:
        print(f"Running segmentation QC for dataset: {dataset}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {qc_dir}")

    # add assert, dataset must have "R1" key
    assert "R1" in dataset.rounds, "Dataset must contain 'R1' round for segmentation QC."
    cell_info = dataset.rounds["R1"].get_cell_info()

    q1 = 0.2
    q2 = 0.95

    _, _ = fig_centroids_filtered(cell_info, q1=q1, q2=q2, save=True, output_dir=qc_dir)
    if verbose:
        print(f"Segmentation QC plots saved to: {qc_dir}")
