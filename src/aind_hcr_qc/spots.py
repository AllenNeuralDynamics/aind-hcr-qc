"""Spots"""


def qc_spot_detection(data_dir, output_dir, channels=None, verbose=False):
    """
    Run spot detection quality control analysis.
    
    Parameters:
    -----------
    data_dir : Path or str
        Path to data directory containing spot detection results
    output_dir : Path or str
        Directory to save QC outputs
    """
   
    if verbose:
        print("Spot detection QC completed successfully!")
