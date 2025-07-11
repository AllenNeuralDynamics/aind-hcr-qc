import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


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
                              add_cluster_labels=True):
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
    cbar.set_label('Gene Expression Count', rotation=270, labelpad=20)
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



def plot_centroids(df, 
                   orientation='XY', 
                   n_samples=None,
                   color_col= None,
                   cmap = 'viridis',
                   clip_range=(None, None),
                   random_state=42):
    """
    Plot cell centroids in different orientations

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing centroid coordinates
    orientation : str
        One of 'XY', 'ZX', or 'ZY'
    n_samples : int
        Number of random cells to plot
    random_state : int
        Random seed for reproducibility
    """
    # Sample random cells
    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=random_state)

    # if clip_range is specified, clip the color_col
    if color_col is not None:
        if color_col not in df.columns:
            raise ValueError(f"Color column '{color_col}' not found in DataFrame.")
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

    # Create scatter plot
    plt.figure(figsize=(10, 10))
    # use color col to set colorbar of scatter values


    
    plt.scatter(df[x_coord], df[y_coord], alpha=0.5, 
                c=df[color_col], cmap=cmap, s=5)
    plt.xlim(df[x_coord].min() - 10, df[x_coord].max() + 10)
    plt.ylim(df[y_coord].min() - 10, df[y_coord].max() + 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(f'{x_coord}')
    plt.ylabel(f'{y_coord}')
    plt.title(f'Cell Centroids - {plane} (n={n_samples})')

    # Reverse y-axis for ZX and ZY
    if orientation in ['ZX', 'ZY', 'XY']:
        plt.gca().invert_yaxis()

    # axis auto
    #plt.axis('auto')


    plt.show()