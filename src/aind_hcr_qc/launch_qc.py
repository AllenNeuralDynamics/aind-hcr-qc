#!/usr/bin/env python3
"""
QC Launcher Script for AIND HCR Data Quality Control

This script provides a unified interface to run various quality control analyses
on HCR (Hybridization Chain Reaction) data. Each QC type can be enabled via command-line flags.


TODO:
+ can it detect pipeline or capsule run?
"""

import argparse
import sys
from pathlib import Path

from aind_hcr_data_loader.hcr_dataset import create_hcr_dataset

import aind_hcr_qc.camera_alignment as ca
import aind_hcr_qc.round_to_round as r2r
import aind_hcr_qc.segmentation as seg
import aind_hcr_qc.spectral_unmixing as su
import aind_hcr_qc.spot_detection as sd

# Import QC modules
import aind_hcr_qc.tile_alignment as ta


def str2bool(v):
    """Convert string representation to boolean value.

    Args:
        v: Input value to convert to boolean. Can be string or bool.
            Accepted string values (case insensitive):
            - True: 'yes', 'true', 't', 'y', '1'
            - False: 'no', 'false', 'f', 'n', '0'

    Returns:
        bool: The converted boolean value

    Raises:
        argparse.ArgumentTypeError: If input cannot be converted to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_argument_parser():
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run HCR data quality control analyses", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # add --dataset argument to specify the dataset
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for QC analysis")
    # Input/output arguments
    # parser.add_argument(
    #     "--config-file",
    #     type=Path,
    #     help="Path to configuration file (e.g., BigStitcher XML for tile alignment)"
    # )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to data directory",
        default="/root/capsule/data",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory to save QC outputs", default="../results"
    )
    parser.add_argument("--bucket-name", type=str, default="aind-open-data", help="S3 bucket name for data access")

    # QC type flags (all default to False)
    qc_group = parser.add_argument_group("QC Types")
    qc_group.add_argument("--tile-alignment", type=str2bool, default=False, help="Run tile alignment QC analysis")
    qc_group.add_argument("--camera-alignment", type=str2bool, default=False, help="Run camera alignment QC analysis")
    qc_group.add_argument("--segmentation", type=str2bool, default=False, help="Run segmentation QC analysis")
    qc_group.add_argument("--round-to-round", type=str2bool, default=False, help="Run round-to-round QC analysis")
    qc_group.add_argument("--spectral-unmixing", type=str2bool, default=False, help="Run spectral unmixing QC analysis")
    qc_group.add_argument("--spot-detection", type=str2bool, default=False, help="Run spot detection QC analysis")
    qc_group.add_argument("--all", type=str2bool, default=False, help="Run all QC analyses")

    # Additional arguments
    # pyramid level for tile alignment
    parser.add_argument("--pyramid-level", type=int, default=0, help="pyramid level")
    return parser


def qc_tile_alignment_wrapper(args):
    """Wrapper for tile alignment QC."""
    print("=" * 60)
    print("RUNNING TILE ALIGNMENT QC")
    print("=" * 60)

    # Capsule: get HCRDataset.
    dataset = create_hcr_dataset({"round_x": args.dataset}, Path(args.data_dir))
    pc_xml = dataset.rounds["round_x"].tile_alignment_files.pc_xml
    ip_xml = dataset.rounds["round_x"].tile_alignment_files.ip_xml

    # Pipeline: TODO

    # check if file exists make list
    xmls = []
    folder_names = []
    if pc_xml.exists():
        xmls.append(pc_xml)
        folder_names.append("PC")
    if ip_xml.exists():
        xmls.append(ip_xml)
        folder_names.append("PC_IP")
    if not xmls:
        raise FileNotFoundError("No tile alignment XML files found for the specified dataset.")

    # Parse BigStitcher XML
    for xml_path, folder_name in zip(xmls, folder_names):
        print(f"Parsing BigStitcher XML: {xml_path}")
        stitched_xml = ta.parse_bigstitcher_xml(xml_path)

        # Get adjacent tile pairs
        pairs = ta.get_all_adjacent_pairs(stitched_xml["tile_names"], include_diagonals=False)
        print(f"Found {len(pairs)} adjacent tile pairs")

        output_dir = Path(args.output_dir) / "tile_alignment_qc" / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        # Run QC analysis
        ta.qc_tile_alignment(
            stitched_xml=stitched_xml,
            pairs=pairs,
            save_dir=output_dir,
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
    sd.qc_spot_detection(data_dir=args.data_dir, output_dir=output_dir, channels=args.channels, verbose=args.verbose)

    print("Spot detection QC completed successfully!")


def main():
    """Main function to run QC analyses based on command-line arguments."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir = Path(args.output_dir) / args.dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # check dataset folder is in data directory (needed for capsule mode)
    dataset_dir = Path(args.data_dir) / args.dataset
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

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
