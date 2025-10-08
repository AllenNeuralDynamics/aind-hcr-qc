# constants.py

# more vibrant channel colors
Z1_CHANNEL_CMAP_VIBRANT = {
    "405": "#9E9E9E",  # Grey
    "488": "#4CAF50",  # Soft green
    "514": "#F44336",  # Vibrant red
    "561": "#2196F3",  # Sky blue
    "594": "#00BCD4",  # Teal
    "638": "#9C27B0",  # Rich purple
}

# pleasant and soft channel colors
Z1_CHANNEL_CMAP_SOFT = {
    '405': '#A9A9A9',   # darkgrey
    '488': '#3CB373',   # mediumseagreen
    '514': '#FF6347',   # tomato
    '561': '#4169E1',   # royalblue
    '594': '#00BFFF',   # deepskyblue
    '638': '#9370DB'    # mediumpurple
}


VOXEL_SIZE = [0.245055, 0.245055, 1.0]  # [X, Y, Z]
EASI_FISH_THRESHOLD = 0.01

# For data paths, thresholds, etc.
DEFAULT_VOLUME_PERCENTILES = (5, 95)
DEFAULT_SPOT_THRESHOLD = 10
