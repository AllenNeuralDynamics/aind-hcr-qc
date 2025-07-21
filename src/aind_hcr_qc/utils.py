"""General utils"""

import operator
import re


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
