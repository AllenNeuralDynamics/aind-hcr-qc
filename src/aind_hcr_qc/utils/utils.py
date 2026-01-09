"""General utils"""
from __future__ import annotations
from functools import wraps
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import operator
import re

from aind_hcr_qc.constants import Z1_CHANNEL_CMAP_VIBRANT

# -------------------------------------------------------------------------------------------------
# Project specific utils
# -------------------------------------------------------------------------------------------------


def get_gene_channel_colors(dataset):
    """
    Get a dictionary mapping genes to their channel colors.

    Example:
    {'Rn28s': '#9E9E9E',
    'GFP': '#4CAF50',
    'Gad2': '#2196F3'}

    Parameters
    ----------
    dataset : HCRDataset
        The HCR dataset containing round information.

    Returns
    -------
    dict
        A dictionary mapping genes to their channel colors.
        488 channel is green, so show what gene is in that channel as green.
        Example: {'Gad2': '#4CAF50', 'SST': 'purple', ...}
    """

    if not dataset.rounds:
        raise ValueError("Dataset has no rounds. Cannot get gene channel colors.")

    gene_channel_colors = {}
    for round_n in dataset.rounds:
        pm = dataset.rounds[round_n].processing_manifest
        for channel, gene in pm["gene_dict"].items():
            gene_channel_colors[gene["gene"]] = Z1_CHANNEL_CMAP_VIBRANT[channel]

    return gene_channel_colors


# Removed `_get_channel_colors()` function as it is redundant.


def apply_filters_to_df(df, filters):
    """
    Apply filters to a DataFrame.
    filters: dict, e.g. {'over_thresh': True, 'r': '>.5', 'chan': '488'}
    Supports:
      - bool: exact match
      - string: exact match
      - string with operator: '>', '<', '>=', '<=', '==', '!=' (e.g. '>.5')
    """

    ops = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    df_filtered = df
    for col, val in filters.items():
        if isinstance(val, bool):
            df_filtered = df_filtered[df_filtered[col] == val]
        elif isinstance(val, (int, float)):
            df_filtered = df_filtered[df_filtered[col] == val]
        elif isinstance(val, str):
            # Check for operator pattern
            m = re.match(r"(>=|<=|==|!=|>|<)\s*(.+)", val)
            if m:
                op_str, num_str = m.groups()
                op_func = ops[op_str]
                try:
                    num = float(num_str)
                except ValueError:
                    num = num_str
                df_filtered = df_filtered[op_func(df_filtered[col], num)]
            else:
                # Exact string match
                df_filtered = df_filtered[df_filtered[col] == val]
        else:
            raise ValueError(f"Unsupported filter value: {val}")
    return df_filtered


def filter_dict_to_string(filters):
    """
    Convert filter dictionary to abbreviated string for filenames.
    Example: {'over_thresh': True, 'valid_spot': False} -> 'ot-T_vs-F'
    """
    if not filters:
        return "unfiltered"

    abbreviations = {"over_thresh": "ot", "valid_spot": "vs"}

    parts = []
    for key, value in filters.items():
        abbr = abbreviations.get(key, key[:2])
        val_str = "T" if value else "F" if isinstance(value, bool) else str(value)
        parts.append(f"{abbr}-{val_str}")

    return "_".join(parts)



# -------------------------------------------------------------------------------------------------
# Plot utils
# -------------------------------------------------------------------------------------------------


_SAVE_KEYS = {
    "save", "output_dir", "filename", "formats", "dpi",
    "bbox_inches", "pad_inches", "transparent", "facecolor",
    "overwrite", "timestamp", "show", "close", "metadata"
}

def _slugify(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:200] or "figure"

def saveable_plot(defaults: dict | None = None):
    """
    Add flexible saving to any plotting function.

    Added kwargs:
      - save: bool (default False)
      - output_dir: str|Path (default 'figures')
      - filename: str|None (default: function name)
      - formats: tuple|list (default ('png',))
      - dpi: int (default 300)
      - bbox_inches: str|None (default 'tight')
      - pad_inches: float (default 0.02)
      - transparent: bool (default False)
      - facecolor: str|None (default 'white')
      - overwrite: bool (default False)
      - timestamp: bool (default False)
      - show: bool (default True)
      - close: bool (default False)
      - metadata: dict|None (default None)
    """
    defaults = defaults or {}
    dflt = {
        "save": False,
        "output_dir": "figures",
        "filename": None,
        "formats": ("png",),
        "dpi": 300,
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "transparent": False,
        "facecolor": "white",
        "overwrite": False,
        "timestamp": False,
        "show": True,
        "close": False,
        "metadata": None,
    }
    dflt.update(defaults)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract saving kwargs
            save_opts = {k: kwargs.pop(k, dflt[k]) for k in _SAVE_KEYS if k in dflt}
            ret = func(*args, **kwargs)

            # Resolve figure
            if ret is None:
                fig = plt.gcf()
            elif isinstance(ret, tuple):
                fig = next((x for x in ret if hasattr(x, "savefig")), plt.gcf())
            else:
                fig = ret if hasattr(ret, "savefig") else plt.gcf()

            if save_opts["save"]:
                out_dir = Path(save_opts["output_dir"]).expanduser().resolve()
                out_dir.mkdir(parents=True, exist_ok=True)

                base = save_opts["filename"] or func.__name__
                base = _slugify(base)
                if save_opts["timestamp"]:
                    base = f"{base}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

                formats = save_opts["formats"]
                if isinstance(formats, str):
                    formats = [formats]

                for ext in formats:
                    ext = ext.lstrip(".")
                    path = out_dir / f"{base}.{ext}"
                    if path.exists() and not save_opts["overwrite"]:
                        # Auto-increment to avoid clobbering
                        i = 1
                        while True:
                            candidate = out_dir / f"{base}_{i}.{ext}"
                            if not candidate.exists():
                                path = candidate
                                break
                            i += 1
                    fig.savefig(
                        path,
                        dpi=save_opts["dpi"],
                        bbox_inches=save_opts["bbox_inches"],
                        pad_inches=save_opts["pad_inches"],
                        transparent=save_opts["transparent"],
                        facecolor=save_opts["facecolor"],
                        metadata=save_opts["metadata"],
                    )

            if not save_opts["show"]:
                plt.close(fig)  # prevent UI popups in batch runs
            elif save_opts["close"]:
                # Show then close, if you prefer explicit lifecycle
                plt.show()
                plt.close(fig)

            return ret
        return wrapper
    return decorator


# -------------------------------------------------------------------------------------------------
# Image file utilities
# -------------------------------------------------------------------------------------------------


def _natural_sort_key(s):
    """
    Sort strings with numbers naturally (1, 2, 10 instead of 1, 10, 2).
    
    Parameters
    ----------
    s : str or Path
        String or Path to sort
        
    Returns
    -------
    list
        List of string and int components for natural sorting
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


def combine_pngs_to_pdf(
    input_dir: str | Path,
    output_path: str | Path | None = None,
    pattern: str = "*.png",
    sort: bool = True,
    verbose: bool = True
) -> Path | None:
    """
    Combine all PNG files in a directory into a single multi-page PDF.
    
    Parameters
    ----------
    input_dir : str | Path
        Directory containing PNG files
    output_path : str | Path | None
        Output PDF path. If None, saves as 'combined.pdf' in input_dir
    pattern : str
        Glob pattern for matching files (default: "*.png")
    sort : bool
        Whether to sort files naturally by name (default: True)
    verbose : bool
        Whether to print progress messages (default: True)
    
    Returns
    -------
    Path | None
        Path to the created PDF file, or None if no PNG files found
        
    Examples
    --------
    >>> # Combine all PNG files in a directory
    >>> combine_pngs_to_pdf("/path/to/figures")
    
    >>> # Combine with specific pattern and custom output
    >>> combine_pngs_to_pdf(
    ...     input_dir="/path/to/figures",
    ...     output_path="/path/to/output.pdf",
    ...     pattern="*_analysis.png"
    ... )
    
    Notes
    -----
    Requires PIL/Pillow package. Images with RGBA mode will be converted
    to RGB with white background. All images are converted to RGB mode
    as required by PDF format.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PIL/Pillow is required for combine_pngs_to_pdf. "
            "Install with: pip install Pillow"
        )
    
    input_dir = Path(input_dir)
    
    # Find all PNG files
    png_files = list(input_dir.glob(pattern))
    
    if not png_files:
        if verbose:
            print(f"No PNG files found matching '{pattern}' in {input_dir}")
        return None
    
    # Sort files naturally
    if sort:
        png_files = sorted(png_files, key=_natural_sort_key)
    
    if verbose:
        print(f"Found {len(png_files)} PNG files")
    
    # Convert all images to RGB (PDF requires RGB mode)
    images = []
    for png_path in png_files:
        try:
            img = Image.open(png_path)
            # Convert RGBA to RGB if needed
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                images.append(rgb_img)
            elif img.mode != 'RGB':
                images.append(img.convert('RGB'))
            else:
                images.append(img)
            if verbose:
                print(f"  ✓ {png_path.name}")
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed to load {png_path.name}: {e}")
            continue
    
    if not images:
        if verbose:
            print("No valid images to combine")
        return None
    
    # Set output path
    if output_path is None:
        output_path = input_dir / "combined.pdf"
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as PDF
    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            resolution=300.0,
            quality=95
        )
        if verbose:
            print(f"\n✓ Saved PDF: {output_path}")
            print(f"  Pages: {len(images)}")
        return output_path
    except Exception as e:
        if verbose:
            print(f"\n✗ Failed to save PDF: {e}")
        return None
