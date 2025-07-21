"""General utils"""

import operator
import re

from aind_hcr_qc.constants import CHANNEL_COLORS

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
            gene_channel_colors[gene["gene"]] = CHANNEL_COLORS[channel]

    return gene_channel_colors


def _get_channel_colors():
    return CHANNEL_COLORS


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
