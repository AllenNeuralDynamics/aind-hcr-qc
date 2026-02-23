#!/usr/bin/env python
"""
Batch run unmixing analysis for multiple HCR datasets.

This script generates comprehensive unmixing analysis figures for all rounds
across multiple mice datasets.
"""

from pathlib import Path
from aind_hcr_data_loader.hcr_dataset import create_hcr_dataset_from_config
from aind_hcr_qc.utils.utils import combine_pngs_to_pdf
import aind_hcr_qc.viz.spectral_unmixing as su


def main():
    """Run unmixing analysis for all specified mice."""
    
    # Define mice to process
    mice = [
        "755252",      # lavender
        "767022",      # rosemary
        "754803",      # old bay
        "767018",      # oregano
        "785054-v1",   
        "783551-v1"
    ]
    
    # Configuration
    config_path = "/root/capsule/code/MOUSE_HCR_CONFIG.json"
    output_dir = Path("/root/capsule/scratch/unmixing_figs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting unmixing analysis for {len(mice)} mice...")
    print(f"Output directory: {output_dir}")
    print("-" * 80)
    
    # Load all datasets
    print("\nLoading datasets...")
    datasets = {}
    for mouse in mice:
        try:
            datasets[mouse] = create_hcr_dataset_from_config(
                mouse, 
                config_path=config_path
            )
            print(f"  ✓ Loaded {mouse}")
        except Exception as e:
            print(f"  ✗ Failed to load {mouse}: {e}")
    
    print(f"\nSuccessfully loaded {len(datasets)}/{len(mice)} datasets")
    print("-" * 80)
    
    # Process each dataset
    total_figures = 0
    total_skipped = 0
    
    for mouse_id, ds in datasets.items():
        print(f"\nProcessing {mouse_id} ({ds.metadata.get('nickname', 'N/A')})...")
        
        # Load spots data
        try:
            print(f"  Loading spots data...")
            unmixed_spots_df = ds.load_all_rounds_spots_mp(table_type='unmixed_spots')
            mixed_spots_df = ds.load_all_rounds_spots_mp(table_type='mixed_spots')
            print(f"    Mixed spots: {len(mixed_spots_df):,}")
            print(f"    Unmixed spots: {len(unmixed_spots_df):,}")
        except Exception as e:
            print(f"  ✗ Failed to load spots for {mouse_id}: {e}")
            total_skipped += len(ds.rounds)
            continue
        
        # Process each round
        for round_key in ds.rounds.keys():
            try:
                filename = f"{ds.mouse_id}_{round_key}_unmixing_analysis"
                
                fig, axes, fate_df = su.fig_unmixing_comprehensive(
                    ds, 
                    round_key, 
                    mixed_spots_df, 
                    unmixed_spots_df
                )
                
                # Save figure
                output_path = output_dir / f"{filename}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"  ✓ Saved {round_key}: {filename}.png")
                
                # Close figure to free memory
                import matplotlib.pyplot as plt
                plt.close(fig)
                
                total_figures += 1
                
            except Exception as e:
                print(f"  ✗ Skipping {round_key} for {mouse_id}: {e}")
                total_skipped += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total figures generated: {total_figures}")
    print(f"Total rounds skipped: {total_skipped}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Combine all PNGs into a single PDF
    if total_figures > 0:
        print("\n" + "=" * 80)
        print("Combining all figures into PDF...")
        print("=" * 80)
        pdf_path = combine_pngs_to_pdf(
            input_dir=output_dir,
            output_path=output_dir / "all_unmixing_analysis.pdf",
            pattern="*_unmixing_analysis.png",
            sort=True,
            verbose=True
        )
        if pdf_path:
            print("\n" + "=" * 80)
            print(f"✓ PDF saved: {pdf_path}")
            print("=" * 80)


if __name__ == "__main__":
    main()
