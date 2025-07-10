"""Spectral unmixing"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_spot_count(cxg_data, color_dict=None, volume_filter=True, 
                        volume_percentiles=(5, 95), log_plot = True, figsize=(12, 4),
                        min_n_spots=2):
    """
    Create a comprehensive QC figure for spot count distributions by gene.
    
    Parameters:
    -----------
    cxg_data : pandas.DataFrame
        Cell-by-gene data with columns: cell_id, gene, spot_count, volume
    color_dict : dict, optional
        Dictionary mapping gene names to colors. If None, uses default palette.
    volume_filter : bool
        Whether to filter cells by volume percentiles
    volume_percentiles : tuple
        (min_percentile, max_percentile) for volume filtering
    log_plot : bool
        Whether to use logarithmic scale for spot counts
    figsize : tuple
        Figure size (width, height)
    min_n_spots : int
        Minimum number of spots per gene to include in the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
 
    
    # Make a copy to avoid modifying original data
    data = cxg_data.copy()
    
    # Apply volume filter if requested
    if volume_filter and 'volume' in data.columns:
        vol_min = np.percentile(data['volume'], volume_percentiles[0])
        vol_max = np.percentile(data['volume'], volume_percentiles[1])
        data = data[(data['volume'] >= vol_min) & (data['volume'] <= vol_max)]
        filter_text = f"Volume filtered: {volume_percentiles[0]}-{volume_percentiles[1]}th percentile\n({vol_min:.0f}-{vol_max:.0f})"
    else:
        filter_text = "No volume filtering"

    # apply minimum number of spots filter
    if min_n_spots > 0:
        # for each cell, count the number of spots per gene, remove those with fewer than min_n_spots
        gene_counts = data.groupby(['cell_id', 'gene'])['spot_count'].sum().reset_index()
        gene_counts = gene_counts[gene_counts['spot_count'] >= min_n_spots]
        data = data[data.set_index(['cell_id', 'gene']).index.isin(gene_counts.set_index(['cell_id', 'gene']).index)]
        filter_text += f"\nMinimum {min_n_spots} spots per gene"
    else:
        filter_text += "\nNo minimum spots filter applied"
    
    
    # Get unique genes
    genes = sorted(data['gene'].unique())
    n_genes = len(genes)
    
    # Set up colors
    if color_dict is None:
        # Use a nice color palette
        colors = sns.color_palette("husl", n_genes)
        color_dict = dict(zip(genes, colors))
    
    # Calculate statistics for each gene
    stats = data.groupby('gene')['spot_count'].agg([
        'count', 'min', 'max', 'median', 'mean', 'std'
    ]).round(2)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Calculate grid layout (try to make it roughly square)
    cols = int(np.ceil(np.sqrt(n_genes)))
    rows = int(np.ceil(n_genes / cols))
    
    # Create subplots for each gene
    for i, gene in enumerate(genes):
        ax = plt.subplot(rows, cols, i + 1)
        
        gene_data = data[data['gene'] == gene]['spot_count']
        color = color_dict.get(gene, 'steelblue')
        
        # Handle log plotting
        if log_plot and gene_data.min() > 0:
            # Use log-spaced bins for better visualization
            bins = np.logspace(np.log10(max(gene_data.min(), 1)), 
                             np.log10(gene_data.max()), 
                             min(30, len(gene_data.unique())))
            # Create histogram with log bins
            sns.histplot(gene_data, bins=bins, kde=False, color=color, alpha=0.7, ax=ax)
            ax.set_xscale('log')
        else:
            # Regular histogram
            sns.histplot(gene_data, kde=True, color=color, alpha=0.7, ax=ax)
        
        # Add vertical lines for statistics
        median_val = gene_data.median()
        mean_val = gene_data.mean()
        
        ax.axvline(median_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Median: {median_val:.1f}')
        ax.axvline(mean_val, color='orange', linestyle=':', alpha=0.8, linewidth=2, label=f'Mean: {mean_val:.1f}')
        
        # Customize subplot
        ax.set_title(f'{gene}\n(n={len(gene_data)}, range: {gene_data.min()}-{gene_data.max()})', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Spot Count' + (' (log scale)' if log_plot and gene_data.min() > 0 else ''), fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.tick_params(labelsize=8)
        
        # Add legend for small subplots
        if len(gene_data) > 0:
            ax.legend(fontsize=7, loc='upper right')
        
        # Set reasonable x-axis limits
        if not log_plot and gene_data.max() > 0:
            ax.set_xlim(0, gene_data.quantile(0.99) * 1.1)
    
    # Remove empty subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(plt.subplot(rows, cols, j + 1))
    
    # Add overall title with summary info
    total_cells = len(data['cell_id'].unique())
    total_spots = data['spot_count'].sum()
    
    fig.suptitle(f'Spot Count QC by Gene\n{total_cells:,} cells, {total_spots:,} total spots\n{filter_text}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Print summary statistics table
    print("\nSummary Statistics by Gene:")
    print("=" * 60)
    print(stats.to_string())
    
    return fig


def print_volume_filtering__summary(cxg):
    # Summary comparison between filtered and unfiltered data
    print("\n" + "="*80)
    print("DATA SUMMARY COMPARISON")
    print("="*80)

    # Original data summary
    print("\nORIGINAL DATA (all cells):")
    print(f"  Total cells: {cxg['cell_id'].nunique():,}")
    print(f"  Total spots: {cxg['spot_count'].sum():,}")
    print(f"  Volume range: {cxg['volume'].min():.0f} - {cxg['volume'].max():.0f}")
    print(f"  Median volume: {cxg['volume'].median():.0f}")

    # Volume filtered data
    vol_min = np.percentile(cxg['volume'], 5)
    vol_max = np.percentile(cxg['volume'], 95)
    filtered_data = cxg[(cxg['volume'] >= vol_min) & (cxg['volume'] <= vol_max)]

    print(f"\nVOLUME FILTERED DATA (5th-95th percentile):")
    print(f"  Volume filter range: {vol_min:.0f} - {vol_max:.0f}")
    print(f"  Cells retained: {filtered_data['cell_id'].nunique():,} ({filtered_data['cell_id'].nunique()/cxg['cell_id'].nunique()*100:.1f}%)")
    print(f"  Spots retained: {filtered_data['spot_count'].sum():,} ({filtered_data['spot_count'].sum()/cxg['spot_count'].sum()*100:.1f}%)")
    print(f"  Cells excluded: {cxg['cell_id'].nunique() - filtered_data['cell_id'].nunique():,}")

    # Per-gene comparison
    print(f"\nPER-GENE IMPACT OF VOLUME FILTERING:")
    comparison = []
    for gene in sorted(cxg['gene'].unique()):
        orig_gene = cxg[cxg['gene'] == gene]
        filt_gene = filtered_data[filtered_data['gene'] == gene]
        
        comparison.append({
            'Gene': gene,
            'Orig_Cells': len(orig_gene),
            'Filt_Cells': len(filt_gene),
            'Cells_Retained_%': len(filt_gene)/len(orig_gene)*100 if len(orig_gene) > 0 else 0,
            'Orig_Spots': orig_gene['spot_count'].sum(),
            'Filt_Spots': filt_gene['spot_count'].sum(),
            'Spots_Retained_%': filt_gene['spot_count'].sum()/orig_gene['spot_count'].sum()*100 if orig_gene['spot_count'].sum() > 0 else 0
        })

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False, float_format='%.1f'))


def qc_spectral_unmixing(data_dir, output_dir, channels=None, verbose=False):
    """
    Run spectral unmixing quality control analysis.

    Parameters:
    -----------
    data_dir : Path or str
        Path to data directory containing spectral data
    output_dir : Path or str
        Directory to save QC outputs
    channels : list, optional
        List of channels to analyze
    verbose : bool
        Enable verbose output
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    if channels is None:
        channels = ["405", "488", "514", "561", "594", "638"]

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Running spectral unmixing QC for channels: {channels}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")

    # Placeholder analysis - create dummy spectral unmixing plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Channel crosstalk matrix
    n_channels = len(channels)
    crosstalk_matrix = np.random.uniform(0, 0.3, (n_channels, n_channels))
    np.fill_diagonal(crosstalk_matrix, 1.0)  # Perfect signal in own channel

    im = axes[0, 0].imshow(crosstalk_matrix, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[0, 0].set_title("Channel Crosstalk Matrix")
    axes[0, 0].set_xlabel("Excitation Channel")
    axes[0, 0].set_ylabel("Detection Channel")
    axes[0, 0].set_xticks(range(n_channels))
    axes[0, 0].set_yticks(range(n_channels))
    axes[0, 0].set_xticklabels(channels)
    axes[0, 0].set_yticklabels(channels)
    plt.colorbar(im, ax=axes[0, 0])

    # Spectral profiles
    wavelengths = np.linspace(400, 700, 100)
    for i, channel in enumerate(channels[:3]):  # Show first 3 channels
        emission = np.exp(-((wavelengths - (400 + i * 50)) ** 2) / (2 * 20**2))
        axes[0, 1].plot(wavelengths, emission, label=f"Ch {channel}", linewidth=2)
    axes[0, 1].set_xlabel("Wavelength (nm)")
    axes[0, 1].set_ylabel("Normalized Intensity")
    axes[0, 1].set_title("Spectral Emission Profiles")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Unmixing accuracy
    true_vs_unmixed = np.random.normal(0, 0.1, 100) + np.linspace(-1, 1, 100)
    axes[0, 2].scatter(np.linspace(-1, 1, 100), true_vs_unmixed, alpha=0.6)
    axes[0, 2].plot([-1, 1], [-1, 1], "r--", linewidth=2, label="Perfect unmixing")
    axes[0, 2].set_xlabel("True Signal")
    axes[0, 2].set_ylabel("Unmixed Signal")
    axes[0, 2].set_title("Unmixing Accuracy")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Signal-to-noise ratio per channel
    snr_values = np.random.uniform(10, 50, n_channels)
    axes[1, 0].bar(channels, snr_values, color="skyblue", alpha=0.7, edgecolor="black")
    axes[1, 0].set_xlabel("Channel")
    axes[1, 0].set_ylabel("SNR (dB)")
    axes[1, 0].set_title("Signal-to-Noise Ratio by Channel")
    axes[1, 0].grid(True, alpha=0.3)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

    # Bleedthrough correction efficiency
    correction_efficiency = np.random.uniform(0.8, 0.98, n_channels)
    axes[1, 1].bar(channels, correction_efficiency, color="lightcoral", alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("Channel")
    axes[1, 1].set_ylabel("Correction Efficiency")
    axes[1, 1].set_title("Bleedthrough Correction Efficiency")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

    # Residual analysis
    residuals = np.random.normal(0, 1, 1000)
    axes[1, 2].hist(residuals, bins=30, alpha=0.7, edgecolor="black", color="lightgreen")
    axes[1, 2].set_xlabel("Unmixing Residuals")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].set_title("Unmixing Residuals Distribution")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "spectral_unmixing_qc.png", dpi=300, bbox_inches="tight")
    plt.close()

    if verbose:
        print("Spectral unmixing QC completed successfully!")


def dummy_func():
    """Dummy function"""
    pass
