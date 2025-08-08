import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import gene_plotter
from typing import List, Optional
import pandas as pd

from aind_hcr_data_loader.hcr_dataset import HCRDataset




# -------------------------------------------------------------------------------------------------
# Multi Round
# -------------------------------------------------------------------------------------------------

def plot_single_cell_expression_all_rounds(
    plot_cell_id: int,
    dataset: HCRDataset,
    pyramid_level: str = "0",
    rounds: List[str] = None, 
    verbose: bool = False
) -> plt.Figure:
    """
    Plot single cell expression across multiple HCR rounds in a compact vertical layout.
    
    Creates a multi-panel figure showing all channels for a specific cell across different
    imaging rounds. Each round is displayed as a horizontal row of channel images with
    minimal spacing between rounds for easy comparison.
    
    Parameters
    ----------
    plot_cell_id : int
        The cell ID to visualize across rounds
     dataset : HCRDataset, optional
        The HCR dataset object containing the imaging data, by default None
    rounds : list of str
        List of round identifiers (e.g., ["R1", "R2", "R3"])
    pyramid_level : str, optional
        Zarr pyramid level for image resolution, by default "0" (full resolution)
    verbose : bool, optional
        Whether to print detailed processing information, by default False
        
    Returns
    -------
    plt.Figure
        The combined figure containing all rounds as subfigures
        
    Notes
    -----
    - Uses automatic intensity scaling (5th-95th percentile) for optimal visualization
    - Displays segmentation mask overlays with transparency
    - Trims images to square aspect ratio for consistent appearance
    - Uses tight layout with minimal spacing between rounds
    - Each round is displayed as a subfigure with channel titles showing gene names
    - Handles missing data gracefully with error messages
    
    Examples
    --------
    >>> fig = plot_single_cell_expression_all_rounds(
    ...     plot_cell_id=12345,
    ...     rounds=["R1", "R2", "R3", "R4"],
    ...     pyramid_level="0",
    ...     dataset=my_hcr_dataset,
    ...     verbose=True
    ... )
    >>> plt.show()
    
    See Also
    --------
    gene_plotter.plot_all_channels_cell : Individual round plotting function
    """
    if rounds is None:
        rounds = dataset.rounds.keys() if dataset else []
    if not isinstance(rounds, list):
        raise ValueError("Rounds must be a list of round identifiers (e.g., ['R1', 'R2', ...])")
    
    # Create a single parent figure
    fig = plt.figure(figsize=(20, 5*len(rounds)))
    
    # Create a GridSpec layout
    gs = gridspec.GridSpec(len(rounds), 1, figure=fig, hspace=0.05, wspace=0,
                           top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    # Create subfigures for each round
    for i, round_n in enumerate(rounds):
        # Create a subfigure from the gridspec
        subfig = fig.add_subfigure(gs[i, :])
        
        try:
            # Plot directly on the subfigure
            gene_plotter.plot_all_channels_cell(
                dataset=dataset,
                round_key=round_n,
                cell_id=plot_cell_id,
                pyramid_level=pyramid_level,
                vmin_vmax="auto",  # Use 5th-95th percentile
                plot_mask_outlines=True,
                trim_to_square=True,  # Default - trim to square
                figsize=None,  # Wide figure for single row
                verbose=verbose,
                fig=subfig  # Pass the subfigure
            )
        except Exception as e:
            print(f"Error plotting round {round_n} for cell {plot_cell_id}: {e}")
            # add a placeholder for the subfigure, say an empty plot
            subfig.add_subplot(111).text(0.5, 0.5, f"Error: {e}", fontsize=12, ha='center', va='center')
            plt.axis('off')
            plt.tight_layout()
            continue
        
        # Add title to the subfigure
        subfig.suptitle(f"Round {round_n}", fontsize=16, y=.98)

    # Add overall title
    fig.suptitle(f"Cell ID {plot_cell_id}", fontsize=18, y=1.05)
    # plt.subplots_adjust(
    #     top=0.95,
    #     bottom=0.05,
    #     left=0.02,
    #     right=0.98,
    #     hspace=0.1  # Small height spacing
    # )

    plt.tight_layout()
    return fig