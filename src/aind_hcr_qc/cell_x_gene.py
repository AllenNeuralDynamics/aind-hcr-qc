import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns


def plot_cell_x_gene_simple(cxg, 
                            clip_range=(0, 50), 
                            sort_gene=None,
                            fig_size=(4, 6)):
    """
    Plot the cell x gene matrix as an image with inverted colormap.

    Parameters
    ----------
    cxg : pd.DataFrame
        Cell x gene matrix with genes as columns and cells as rows.
    clip_range : tuple
        Range to clip the values in the cell x gene matrix.
    sort_gene : str, optional
        Gene to sort the cell x gene matrix by. If None, sorts by the first gene.
        Default is None.
    fig_size : tuple
        Size of the figure to plot.
    
    """
    if not isinstance(cxg, pd.DataFrame):
        raise ValueError("Input cxg must be a pandas DataFrame.")


    cxg = cxg.copy()  # avoid modifying the original DataFrame
    # set color min/max
    cxg = cxg.fillna(0)  # fill NaN values with 0

    # make int
    cxg = cxg.astype(int)
    cxg = cxg.clip(lower=clip_range[0], upper=clip_range[1])

    # sort by gene if specified
    if sort_gene is not None and sort_gene not in cxg.columns:
        raise ValueError(f"Gene '{sort_gene}' not found in cell x gene matrix columns.")
    if sort_gene is not None:
        cxg = cxg.sort_values(by=sort_gene, ascending=False)

    fig, ax = plt.subplots(figsize=fig_size)

    ax.imshow(cxg, aspect='auto', cmap='gray_r', interpolation='none',)
    # show colorbar
    cbar = plt.colorbar(ax.imshow(cxg, aspect='auto', cmap='gray_r', interpolation='none'), ax=ax)
    cbar.set_label('Gene Expression Count', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)
    #cbar.set_clim(clip_range[0], clip_range[1])


    # add gene names from dataframe
    #plt.yticks(ticks=range(len(cxg.index)), labels=cxg.index)
    ax.set_xticks(ticks=range(len(cxg.columns)), labels=cxg.columns, rotation=90)

    # colorbar

    return fig


def plot_cell_x_gene_clustered(cxg, 
                              clip_range=(0, 50), 
                              sort_gene=None,
                              fig_size=(4, 6),
                              k=3,
                              add_cluster_labels=True,
                              cbar_label='Gene Expression Count'):
    """
    Plot the cell x gene matrix as an image with inverted colormap and K-means clustering.

    Parameters
    ----------
    cxg : pd.DataFrame
        Cell x gene matrix with genes as columns and cells as rows.
    clip_range : tuple
        Range to clip the values in the cell x gene matrix.
    sort_gene : str, optional
        Gene to sort the cell x gene matrix by. If None, performs clustering instead.
        Default is None.
    fig_size : tuple
        Size of the figure to plot.
    k : int
        Number of clusters for K-means clustering. Default is 3.
    add_cluster_labels : bool
        Whether to add green dashed lines and labels to indicate cluster boundaries.
        Default is True.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    cluster_labels : np.ndarray
        Array of cluster assignments for each cell.
    """
    if not isinstance(cxg, pd.DataFrame):
        raise ValueError("Input cxg must be a pandas DataFrame.")

    cxg = cxg.copy()  # avoid modifying the original DataFrame
    # set color min/max
    cxg = cxg.fillna(0)  # fill NaN values with 0

    # make int
    cxg = cxg.astype(int)
    cxg = cxg.clip(lower=clip_range[0], upper=clip_range[1])

    # Perform clustering or sorting
    if sort_gene is not None and sort_gene in cxg.columns:
        # Sort by specified gene
        cxg = cxg.sort_values(by=sort_gene, ascending=False)
        cluster_labels = None
    else:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cxg.values)
        
        # Sort by cluster labels first, then by total expression within each cluster
        cxg['cluster'] = cluster_labels
        cxg['total_expression'] = cxg.drop('cluster', axis=1).sum(axis=1)
        cxg = cxg.sort_values(['cluster', 'total_expression'], ascending=[True, False])
        
        # Update cluster labels to match the sorted order
        cluster_labels = cxg['cluster'].values
        
        # Remove helper columns
        cxg = cxg.drop(['cluster', 'total_expression'], axis=1)

    fig, ax = plt.subplots(figsize=fig_size)

    # Plot the heatmap
    im = ax.imshow(cxg, aspect='auto', cmap='gray_r', interpolation='none')
    
    # show colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    # add gene names from dataframe
    ax.set_xticks(ticks=range(len(cxg.columns)), labels=cxg.columns, rotation=90)

    # Add cluster labels if clustering was performed and requested
    if cluster_labels is not None and add_cluster_labels:
        # Find cluster boundaries
        cluster_changes = np.where(np.diff(cluster_labels) != 0)[0] + 0.5
        
        # Add horizontal dashed lines at cluster boundaries
        for boundary in cluster_changes:
            ax.axhline(y=boundary, color='green', linestyle='--', linewidth=2.0, alpha=0.9)
        
        # Add cluster labels
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            # Find the middle position of each cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_center = (cluster_indices[0] + cluster_indices[-1]) / 2
            
            # Add text label
            ax.text(-0.5, cluster_center, f'C{cluster_id}', 
                   color='green', fontweight='bold', fontsize=14,
                   verticalalignment='center', horizontalalignment='right')

    return fig, cluster_labels


def cell_mean_centroid_df(dataset, mean_csv, round_key):
    """
    Load cell mean flouresence data from a CSV file and merges with cell information.
    """
    
    # filter volume with 5 - 95 percentile
    cell_info = dataset.rounds[round_key].get_cell_info() # Double check if the right cells..., since R1 is default in loader
    
    # filter volume
    cell_info = cell_info[(cell_info["volume"] > cell_info["volume"].quantile(0.05)) & (cell_info["volume"] < cell_info["volume"].quantile(0.95))]
    filt_cells = cell_info.cell_id.values

    mean_df = pd.read_csv(mean_csv)
    # merge with cell_info
    mean_df = mean_df.merge(cell_info, on='cell_id', how='left')

    # filter by cell_id
    mean_df = mean_df[mean_df['cell_id'].isin(filt_cells)]

    print(f"Filtered mean_df shape: {mean_df.shape}")
    return mean_df



def plot_centroids_with_hist(df, 
                            orientation='XY', 
                            n_samples=None,
                            color_col=None,
                            cmap='viridis',
                            clip_range=(None, None),
                            xlims=(None, None),
                            ylims=(None, None),
                            random_state=42,
                            fig_size=(12, 8)):
    """
    Plot cell centroids with a vertical KDE histogram of the color column.

    TODO: maybe can move out of cellxgene?

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing centroid coordinates
    orientation : str
        One of 'XY', 'ZX', or 'ZY'
    n_samples : int, optional
        Number of random cells to plot
    color_col : str, optional
        Column to use for coloring and histogram
    cmap : str
        Colormap for scatter plot
    clip_range : tuple
        Range to clip the color column values
    random_state : int
        Random seed for reproducibility
    fig_size : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    """
    # Sample random cells if specified
    if n_samples is not None and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_state)

    # Clip color column if specified
    if color_col is not None:
        if color_col not in df.columns:
            raise ValueError(f"Color column '{color_col}' not found in DataFrame.")
        df = df.copy()
        df[color_col] = df[color_col].clip(lower=clip_range[0], upper=clip_range[1])
    
    # Define coordinate mappings
    coords = {
        'XY': ('x_centroid', 'y_centroid', 'XY Plane'),
        'ZX': ('x_centroid', 'z_centroid', 'ZX Plane'),
        'ZY': ('y_centroid', 'z_centroid', 'ZY Plane')
    }

    if orientation not in coords:
        raise ValueError("Orientation must be one of: 'XY', 'ZX', 'ZY'")

    x_coord, y_coord, plane = coords[orientation]

    # Create figure with subplots
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=fig_size, 
                                             gridspec_kw={'width_ratios': [3, 1]})

    # --- Scatter plot ---
    if color_col is not None:
        scatter = ax_scatter.scatter(df[x_coord], df[y_coord], 
                                   alpha=0.6, c=df[color_col], 
                                   cmap=cmap, s=8)
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_scatter)
        cbar.set_label(color_col, rotation=270, labelpad=15)
    else:
        ax_scatter.scatter(df[x_coord], df[y_coord], alpha=0.6, s=8)

    # Set scatter plot properties
    if xlims[0] is not None or xlims[1] is not None:
        ax_scatter.set_xlim(xlims[0], xlims[1])
    else:
        ax_scatter.set_xlim(df[x_coord].min() - 10, df[x_coord].max() + 10)
        
    
    if ylims[0] is not None or ylims[1] is not None:
        ax_scatter.set_ylim(ylims[0], ylims[1])
    else:
        ax_scatter.set_ylim(df[y_coord].min() - 10, df[y_coord].max() + 10)

    ax_scatter.set_aspect('equal', adjustable='box')
    ax_scatter.set_xlabel(f'{x_coord}')
    ax_scatter.set_ylabel(f'{y_coord}')
    ax_scatter.set_title(f'Cell Centroids - {plane}')
    
    # Reverse y-axis for consistency
    if orientation in ['ZX', 'ZY', 'XY']:
        ax_scatter.invert_yaxis()

    # --- Histogram/KDE plot ---
    if color_col is not None:
        # Get data for histogram - we want to show color values distributed across y-axis coordinates
        valid_data = df[[y_coord, color_col]].dropna()
        
        if len(valid_data) > 0:
            # Create binned averages of color values across y-coordinate bins
            y_bins = np.linspace(valid_data[y_coord].min(), valid_data[y_coord].max(), 30)
            y_centers = (y_bins[:-1] + y_bins[1:]) / 2
            
            # Calculate average color value in each y-coordinate bin
            binned_means = []
            for i in range(len(y_bins)-1):
                mask = (valid_data[y_coord] >= y_bins[i]) & (valid_data[y_coord] < y_bins[i+1])
                if mask.sum() > 0:
                    binned_means.append(valid_data[color_col][mask].sum())

                else:
                    binned_means.append(0)
            hist_color = plt.cm.get_cmap(cmap)(0.8)
            # Plot as horizontal bars showing average color value at each y position
            ax_hist.barh(y_centers, binned_means, height=(y_bins[1]-y_bins[0])*0.8,
                        alpha=0.7, color=hist_color, edgecolor='black', linewidth=0.5)
            
            # Also add a smoothed line
            from scipy.interpolate import interp1d
            from scipy.ndimage import gaussian_filter1d
            
            # Smooth the data
            # set color to top of cmap
            
            smoothed_means = gaussian_filter1d(binned_means, sigma=1.5)
            ax_hist.plot(smoothed_means, y_centers, color=hist_color, linewidth=2, alpha=0.8)
            
            ax_hist.set_ylabel(f'{y_coord}')
            ax_hist.set_xlabel(f'Summed {color_col} intensity')
            #ax_hist.set_title(f'{color_col} intensity vs {y_coord}')
            
            # Set y-limits to match the scatter plot
            ax_hist.set_ylim(ax_scatter.get_ylim())
            
            # Set x-limits for color values
            # if clip_range[0] is not None or clip_range[1] is not None:
            #     x_min = clip_range[0] if clip_range[0] is not None else valid_data[color_col].min()
            #     x_max = clip_range[1] if clip_range[1] is not None else valid_data[color_col].max()
            #     ax_hist.set_xlim(0, x_max)
            #     print(f"Histogram x-limits set to: {x_min} - {x_max}")
        else:
            ax_hist.text(0.5, 0.5, 'No valid data\nfor histogram', 
                        ha='center', va='center', transform=ax_hist.transAxes)
            ax_hist.set_title('No Data')
    else:
        # Hide histogram if no color column
        ax_hist.text(0.5, 0.5, 'No color column\nspecified', 
                    ha='center', va='center', transform=ax_hist.transAxes)
        ax_hist.set_title('No Color Data')

    # Add sample size information
    sample_text = f'n={len(df)}'
    if n_samples is not None:
        sample_text += f' (sampled from larger dataset)'
    fig.suptitle(f'{sample_text}', fontsize=10, y=0.02)

    plt.tight_layout()
    return fig