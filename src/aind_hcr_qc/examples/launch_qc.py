#!/usr/bin/env python3
"""
QC Launcher Script for AIND HCR Data Quality Control

This script provides a unified interface to run various quality control analyses
on HCR (Hybridization Chain Reaction) data. Each QC type can be enabled via command-line flags.
"""

import argparse
import sys
from pathlib import Path

from aind_hcr_data_loader.hcr_dataset import create_hcr_dataset

import aind_hcr_qc.camera_alignment as ca
import aind_hcr_qc.round_to_round as r2r
import aind_hcr_qc.segmentation as seg
import aind_hcr_qc.spectral_unmixing as su
import aind_hcr_qc.spots as spots

# Import QC modules
import aind_hcr_qc.tile_alignment as ta


def setup_argument_parser():
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run HCR data quality control analyses", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output arguments
    parser.add_argument(
        "--config-file", type=Path, help="Path to configuration file (e.g., BigStitcher XML for tile alignment)"
    )
    parser.add_argument("--data-dir", type=Path, help="Path to data directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save QC outputs")
    parser.add_argument("--bucket-name", type=str, default="aind-open-data", help="S3 bucket name for data access")

    # QC type flags (all default to False)
    qc_group = parser.add_argument_group("QC Types")
    qc_group.add_argument("--tile-alignment", action="store_true", default=False, help="Run tile alignment QC analysis")
    qc_group.add_argument(
        "--camera-alignment", action="store_true", default=False, help="Run camera alignment QC analysis"
    )
    qc_group.add_argument("--segmentation", action="store_true", default=False, help="Run segmentation QC analysis")
    qc_group.add_argument("--round-to-round", action="store_true", default=False, help="Run round-to-round QC analysis")
    qc_group.add_argument(
        "--spectral-unmixing", action="store_true", default=False, help="Run spectral unmixing QC analysis"
    )
    qc_group.add_argument("--spot-detection", action="store_true", default=False, help="Run spot detection QC analysis")
    qc_group.add_argument("--all", action="store_true", default=False, help="Run all QC analyses")

    # Analysis parameters
    params_group = parser.add_argument_group("Analysis Parameters")
    params_group.add_argument("--pyramid-level", type=int, default=3, help="Pyramid level for tile-based analyses")
    params_group.add_argument(
        "--include-diagonals",
        action="store_true",
        default=False,
        help="Include diagonal tile pairs in tile alignment analysis",
    )
    params_group.add_argument(
        "--channels", nargs="+", default=["405", "488", "514", "561", "594", "638"], help="Channels to analyze"
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output")
    proc_group.add_argument(
        "--dry-run", action="store_true", default=False, help="Show what would be run without executing"
    )

    return parser


def qc_tile_alignment_wrapper(args):
    """Wrapper for tile alignment QC."""
    print("=" * 60)
    print("RUNNING TILE ALIGNMENT QC")
    print("=" * 60)

    if not args.config_file or not args.config_file.exists():
        raise FileNotFoundError(f"BigStitcher XML file not found: {args.config_file}")

    # Parse BigStitcher XML
    print(f"Parsing BigStitcher XML: {args.config_file}")
    stitched_xml = ta.parse_bigstitcher_xml(args.config_file)

    # Get adjacent tile pairs
    print("Finding adjacent tile pairs...")
    pairs = ta.get_all_adjacent_pairs(stitched_xml["tile_names"], include_diagonals=args.include_diagonals)
    print(f"Found {len(pairs)} adjacent tile pairs")

    # Run QC analysis
    ta.qc_tile_alignment(
        stitched_xml=stitched_xml,
        pairs=pairs,
        save_dir=args.output_dir,
        bucket_name=args.bucket_name,
        pyramid_level=args.pyramid_level,
    )
    print("Tile alignment QC completed successfully!")


def qc_camera_alignment_wrapper(args):
    """Wrapper for camera alignment QC."""
    print("=" * 60)
    print("RUNNING CAMERA ALIGNMENT QC")
    print("=" * 60)

    if not args.config_file or not args.config_file.exists():
        raise FileNotFoundError(f"Tile metrics JSON file not found: {args.config_file}")

    # Load tile metrics
    print(f"Loading tile metrics: {args.config_file}")
    metrics_data = ca.load_tile_metrics(args.config_file)

    # Create output directory
    output_dir = Path(args.output_dir) / "camera_alignment_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate translation distribution plots
    print("Generating translation distribution plots...")
    fig_dist, axes_dist = ca.plot_translation_distributions(metrics_data)
    fig_dist.savefig(output_dir / "translation_distributions.png", dpi=300, bbox_inches="tight")

    # Generate translation vector plots
    print("Generating translation vector plots...")
    fig_vec, ax_vec = ca.plot_translation_vectors(metrics_data)
    fig_vec.savefig(output_dir / "translation_vectors.png", dpi=300, bbox_inches="tight")

    print(f"Camera alignment QC plots saved to: {output_dir}")
    print("Camera alignment QC completed successfully!")


def qc_segmentation_wrapper(args):
    """Wrapper for segmentation QC."""
    print("=" * 60)
    print("RUNNING SEGMENTATION QC")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir) / "segmentation_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run segmentation QC
    seg.qc_segmentation(data_dir=args.data_dir, output_dir=output_dir, channels=args.channels, verbose=args.verbose)

    print("Segmentation QC completed successfully!")


def qc_round_to_round_wrapper(args):
    """Wrapper for round-to-round QC."""
    print("=" * 60)
    print("RUNNING ROUND-TO-ROUND QC")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir) / "round_to_round_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run round-to-round QC
    r2r.qc_round_to_round(data_dir=args.data_dir, output_dir=output_dir, channels=args.channels, verbose=args.verbose)

    print("Round-to-round QC completed successfully!")


def qc_spectral_unmixing_wrapper(args):
    """Wrapper for spectral unmixing QC."""
    print("=" * 60)
    print("RUNNING SPECTRAL UNMIXING QC")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir) / "spectral_unmixing_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run spectral unmixing QC
    su.qc_spectral_unmixing(data_dir=args.data_dir, output_dir=output_dir, channels=args.channels, verbose=args.verbose)

    print("Spectral unmixing QC completed successfully!")


def qc_spot_detection_wrapper(args):
    """Wrapper for spot detection QC."""
    print("=" * 60)
    print("RUNNING SPOT DETECTION QC")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir) / "spot_detection_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run spot detection QC
    spots.qc_spot_detection(data_dir=args.data_dir, output_dir=output_dir, channels=args.channels, verbose=args.verbose)

    print("Spot detection QC completed successfully!")


def main():
    """Main function to run QC analyses based on command-line arguments."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if any QC type is specified
    qc_types = [
        args.tile_alignment,
        args.camera_alignment,
        args.segmentation,
        args.round_to_round,
        args.spectral_unmixing,
        args.spot_detection,
        args.all,
    ]

    if not any(qc_types):
        print("ERROR: No QC type specified. Use --help for available options.")
        sys.exit(1)

    # Show configuration if verbose or dry run
    if args.verbose or args.dry_run:
        print("QC Configuration:")
        print(f"  Config file: {args.config_file}")
        print(f"  Data directory: {args.data_dir}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Bucket name: {args.bucket_name}")
        print(f"  Pyramid level: {args.pyramid_level}")
        print(f"  Channels: {args.channels}")
        print(f"  Include diagonals: {args.include_diagonals}")
        print()

    if args.dry_run:
        print("DRY RUN MODE - No analyses will be executed")
        if args.all or args.tile_alignment:
            print("Would run: Tile Alignment QC")
        if args.all or args.camera_alignment:
            print("Would run: Camera Alignment QC")
        if args.all or args.segmentation:
            print("Would run: Segmentation QC")
        if args.all or args.round_to_round:
            print("Would run: Round-to-Round QC")
        if args.all or args.spectral_unmixing:
            print("Would run: Spectral Unmixing QC")
        if args.all or args.spot_detection:
            print("Would run: Spot Detection QC")
        return

    # Run requested QC analyses
    try:
        if args.all or args.tile_alignment:
            qc_tile_alignment_wrapper(args)

        if args.all or args.camera_alignment:
            qc_camera_alignment_wrapper(args)

        if args.all or args.segmentation:
            qc_segmentation_wrapper(args)

        if args.all or args.round_to_round:
            qc_round_to_round_wrapper(args)

        if args.all or args.spectral_unmixing:
            qc_spectral_unmixing_wrapper(args)

        if args.all or args.spot_detection:
            qc_spot_detection_wrapper(args)

        print("\n" + "=" * 60)
        print("ALL QC ANALYSES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"\nERROR: QC analysis failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
