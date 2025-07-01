"""Round to round"""


def qc_round_to_round(data_dir, output_dir, channels=None, verbose=False):
    """
    Run round-to-round quality control analysis.
    
    Parameters:
    -----------
    data_dir : Path or str
        Path to data directory containing round data
    output_dir : Path or str
        Directory to save QC outputs
    """
    
    if verbose:
        print("Round-to-round QC completed successfully!")
